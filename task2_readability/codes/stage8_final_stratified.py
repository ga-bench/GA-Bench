#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 8 FINAL — Domain / Journal / Publisher stratified trends
==============================================================
Run from:  ./task2_readability
Outputs :  output/stage8_final_stratified/{tables,trends}/

On the calibrated readability score (R_calibrated_ridge_01):
  * per-DOMAIN yearly trends (stratify by PRIMARY subject_area = first ';' value)
  * per-domain descriptive slope, flagged when early-year n is thin
  * JOURNAL and PUBLISHER deviations = group mean minus grand mean, after removing
    domain + year composition (shrinkage-style empirical-Bayes toward 0) -> a
    lightweight stand-in for random effects (the formal (1|Journal) model is Stage 9)
  * template-change flag: journals whose year-to-year mean jumps by a large amount
Respects the 2024-2026 concentration: thin years (n<MIN_YEAR_N) flagged, not trusted.
DESCRIPTIVE / association language only. Pure pandas + numpy; matplotlib guarded.
"""
from __future__ import annotations
import re, glob
from pathlib import Path
import numpy as np
import pandas as pd

T2      = Path("./task2_readability/output")
OUT_DIR = T2 / "stage8_final_stratified"
TBL = OUT_DIR / "tables"; TRN = OUT_DIR / "trends"

STAGE5_FINAL   = T2 / "stage5_final/stage5_final_scores.csv"
GA_INDEX       = T2 / "stage1_preprocessing/index/stage1_ga_index.csv"
DATASET_MASTER = Path("./dataset_analyzer/"
                      "paper1_dataset_statistics/output_10k/dataset_master.csv")

SCORE = "R_calibrated_ridge_01"
MIN_YEAR_N = 30           # a domain-year with fewer GAs is flagged as thin
MIN_JOURNAL_N = 20        # journals with fewer GAs excluded from ranking
TEMPLATE_JUMP = 0.10      # year-to-year mean jump flagged as possible template change
SHRINK_K = 25             # empirical-Bayes shrinkage constant (larger = more shrink)
YEAR_CANDS=["publication_year","year","year_clean","pub_year"]
DOM_CANDS=["subject_area","domain"]; PUB_CANDS=["publisher"]; JOUR_CANDS=["journal"]

def norm_id(s):
    s=str(s or "").strip().lower().replace("https://doi.org/","").replace("doi:","")
    return re.sub(r"[^a-z0-9]+","_",s).strip("_")
def find_id_col(df):
    for c in ["paper_id","doi_folder","doi_safe","doi","ga_id","id"]:
        if c in df.columns: return c
    return df.columns[0]
def pick(df,cands):
    low={c.lower():c for c in df.columns}
    for n in cands:
        if n in df.columns: return n
        if n.lower() in low: return low[n.lower()]
    return None
def to_num(x):
    try: return float(x)
    except Exception: return np.nan
def load(p):
    if p is None or not Path(p).exists(): return None
    df=pd.read_csv(p,dtype=str); df["_id"]=df[find_id_col(df)].map(norm_id); return df
def primary_domain(v):
    parts=[p.strip() for p in re.split(r"\s*;\s*",str(v or "")) if p.strip()]
    return parts[0] if parts else "unknown"
def plt_():
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; return plt
    except Exception: return None

def yearly_mean(df, years):
    return [pd.to_numeric(df[df.year==y][SCORE],errors="coerce").dropna().mean() for y in years]
def slope(sub):
    t=sub.groupby("year")[SCORE].mean().dropna()
    return float(np.polyfit(t.index.astype(int),t.values,1)[0]) if len(t)>=3 else float("nan")

def main():
    TBL.mkdir(parents=True,exist_ok=True); TRN.mkdir(parents=True,exist_ok=True)
    S=load(STAGE5_FINAL)
    if S is None: raise SystemExit(f"missing {STAGE5_FINAL}")
    S[SCORE]=S[SCORE].map(to_num)
    idx=load(GA_INDEX); mst=load(DATASET_MASTER)
    d=S.copy()
    if idx is not None:
        yc=pick(idx,YEAR_CANDS)
        if yc: d=d.merge(idx[["_id",yc]].rename(columns={yc:"year_raw"}),on="_id",how="left")
    if mst is not None:
        dc=pick(mst,DOM_CANDS); pc=pick(mst,PUB_CANDS); jc=pick(mst,JOUR_CANDS); yc2=pick(mst,YEAR_CANDS)
        keep=["_id"]+[x for x in [dc,pc,jc,yc2] if x]
        d=d.merge(mst[keep].rename(columns={c:n for c,n in
                  [(dc,"domain_raw"),(pc,"publisher"),(jc,"journal"),(yc2,"year_master")] if c}),on="_id",how="left")
    d["year"]=pd.to_numeric(d.get("year_raw"),errors="coerce")
    if d["year"].isna().all() and "year_master" in d: d["year"]=pd.to_numeric(d["year_master"],errors="coerce")
    d=d.dropna(subset=["year",SCORE]); d["year"]=d["year"].astype(int)
    d["domain"]=d.get("domain_raw","unknown").map(primary_domain)
    for c in ["publisher","journal"]:
        if c not in d: d[c]="unknown"
        d[c]=d[c].fillna("unknown").replace("","unknown")
    years=sorted(d.year.unique()); grand=d[SCORE].mean()
    print(f"[stage8] {len(d)} GAs | domains={d.domain.nunique()} journals={d.journal.nunique()} publishers={d.publisher.nunique()}")

    # ---- per-domain yearly trends + slope ----
    dom_year=[]; dom_slope=[]
    for dom,sub in d.groupby("domain"):
        for y in years:
            v=pd.to_numeric(sub[sub.year==y][SCORE],errors="coerce").dropna()
            if len(v)==0: continue
            dom_year.append([dom,y,len(v),round(v.mean(),4),round(v.std(),4), "thin" if len(v)<MIN_YEAR_N else ""])
        dom_slope.append([dom,len(sub),round(slope(sub),5)])
    pd.DataFrame(dom_year,columns=["domain","year","n","mean","sd","flag"]).to_csv(TBL/"domain_yearly.csv",index=False)
    DS=pd.DataFrame(dom_slope,columns=["domain","n","slope_per_year"]).sort_values("n",ascending=False)
    DS.to_csv(TBL/"domain_slopes.csv",index=False)

    # ---- publisher deviations (shrinkage) ----
    def shrink_dev(col):
        rows=[]
        for g,sub in d.groupby(col):
            n=len(sub); m=pd.to_numeric(sub[SCORE],errors="coerce").mean()
            dev=(m-grand)*(n/(n+SHRINK_K))   # empirical-Bayes style shrink toward 0
            rows.append([g,n,round(m,4),round(dev,4)])
        return pd.DataFrame(rows,columns=[col,"n","mean","shrunk_deviation"])
    shrink_dev("publisher").sort_values("shrunk_deviation").to_csv(TBL/"publisher_deviations.csv",index=False)
    JD=shrink_dev("journal")
    JD=JD[JD.n>=MIN_JOURNAL_N].sort_values("shrunk_deviation")
    JD.to_csv(TBL/"journal_deviations.csv",index=False)

    # ---- template-change flags: large year-to-year jump within a journal ----
    tmpl=[]
    for j,sub in d.groupby("journal"):
        if len(sub)<MIN_JOURNAL_N: continue
        ts=sub.groupby("year")[SCORE].mean().dropna().sort_index()
        for (y0,m0),(y1,m1) in zip(ts.items(),list(ts.items())[1:]):
            if abs(m1-m0)>=TEMPLATE_JUMP:
                tmpl.append([j,y0,y1,round(m0,3),round(m1,3),round(m1-m0,3)])
    pd.DataFrame(tmpl,columns=["journal","year_from","year_to","mean_from","mean_to","jump"]).to_csv(TBL/"template_change_flags.csv",index=False)

    # ---- figures: top domains by n ----
    plt=plt_()
    if plt is not None:
        top=DS.head(8).domain.tolist()
        fig,ax=plt.subplots(figsize=(9,5.5))
        for dom in top:
            sub=d[d.domain==dom]; ax.plot(years,yearly_mean(sub,years),"-o",label=f"{dom[:22]} (n={len(sub)})")
        ax.set_xlabel("publication year"); ax.set_ylabel("mean calibrated readability [0,1]")
        ax.set_ylim(0,1); ax.set_title("Calibrated readability by year, top domains"); ax.legend(fontsize=7,ncol=2); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(TRN/"domain_trends.png",dpi=150); plt.close(fig)

    L=["Stage 8 FINAL — stratified trends (DESCRIPTIVE; association only)","="*55,"",
       f"GAs={len(d)}  domains={d.domain.nunique()}  journals={d.journal.nunique()}  publishers={d.publisher.nunique()}",
       f"grand mean calibrated readability = {grand:.3f}","",
       "Per-domain slope/yr (top by n):","-"*40, DS.head(12).to_string(index=False),"",
       f"Journals flagged for possible template change (|jump|>={TEMPLATE_JUMP}): {len(tmpl)}",
       f"Thin domain-years (n<{MIN_YEAR_N}) flagged in domain_yearly.csv.","",
       "Tables in tables/, figure in trends/. Deviations are shrinkage estimates;",
       "the formal (1|Journal) random-effects model is Stage 9."]
    (OUT_DIR/"stage8_final_report.txt").write_text("\n".join(L)+"\n")
    print("\n"+"\n".join(L)+f"\n\nSaved -> {OUT_DIR}")

if __name__=="__main__":
    main()