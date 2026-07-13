#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 7 FINAL — Temporal Trend Analysis (calibrated readability)
================================================================
Run from:  ./task2_readability
Outputs :  output/stage7_final_temporal_trends/{tables,trends}/

Consumes the Stage-5 FINAL calibrated scores and reports, per publication year:
  * mean / median / SD / 95% CI for calibrated readability (R_calibrated_ridge_01),
    the unweighted baseline (R_overall_unweighted), and each component
  * % low- and % high-readability GAs (fixed global tertile thresholds)
  * RAW yearly trends and CONFOUND-ADJUSTED yearly trends
  * image-quality variables (resolution, OCR confidence) by year, plotted alongside
    readability, to check whether apparent trends track technical artifacts.

ADJUSTED = OLS residualization: score regressed on confounds ONLY (domain + publisher
+ log-resolution + OCR confidence; YEAR EXCLUDED), adjusted = grand_mean + residual.
Descriptive; the inferential mixed model is Stage 9. ASSOCIATION language only.

Pure pandas + numpy; matplotlib guarded.
"""
from __future__ import annotations
import re, glob
from pathlib import Path
import numpy as np
import pandas as pd

T2      = Path("./task2_readability/output")
OUT_DIR = T2 / "stage7_final_temporal_trends"
TBL = OUT_DIR / "tables"; TRN = OUT_DIR / "trends"

STAGE5_FINAL = T2 / "stage5_final/stage5_final_scores.csv"
GA_INDEX     = T2 / "stage1_preprocessing/index/stage1_ga_index.csv"
DATASET_MASTER = Path("./dataset_analyzer/"
                      "paper1_dataset_statistics/output_10k/dataset_master.csv")  # domain/publisher/journal
def autofind(*pats):
    for p in pats:
        h = sorted(glob.glob(str(T2 / p), recursive=True))
        if h: return Path(h[0])
    return None
STAGE2_CSV = autofind("**/*stage2*ocr*feature*.csv", "**/*stage2*feature*.csv")

SCORES = [("R_calibrated_ridge_01", "calibrated"), ("R_overall_unweighted", "unweighted"),
          ("R_text", "text"), ("R_visual", "visual"), ("R_semantic", "semantic")]
YEAR_CANDS = ["publication_year", "year", "year_clean", "pub_year"]
RES_CANDS  = ["pixel_count", "image_pixel_count", "resolution", "width", "image_width"]
OCR_CANDS  = ["mean_ocr_confidence", "ocr_confidence_mean", "avg_ocr_confidence", "ocr_confidence"]
DOM_CANDS  = ["subject_area", "domain"]; PUB_CANDS = ["publisher"]; JOUR_CANDS = ["journal"]

def norm_id(s):
    s=str(s or "").strip().lower().replace("https://doi.org/","").replace("doi:","")
    return re.sub(r"[^a-z0-9]+","_",s).strip("_")
def find_id_col(df):
    for c in ["paper_id","doi_folder","doi_safe","doi","ga_id","id"]:
        if c in df.columns: return c
    return df.columns[0]
def pick(df, cands):
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
def plt_():
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; return plt
    except Exception: return None

def residualize(df, ycol, confs_cat, confs_num):
    """OLS residualization on confounds (year excluded). adjusted = grand_mean + resid."""
    d = df.dropna(subset=[ycol]).copy()
    parts = [np.ones((len(d),1))]
    for c in confs_cat:
        if c in d and d[c].nunique() > 1:
            parts.append(pd.get_dummies(d[c].fillna("NA"), drop_first=True).values.astype(float))
    for c in confs_num:
        if c in d:
            v = pd.to_numeric(d[c], errors="coerce"); v = v.fillna(v.mean())
            parts.append(((v - v.mean())/(v.std() or 1)).values.reshape(-1,1))
    X = np.hstack(parts); y = pd.to_numeric(d[ycol], errors="coerce").values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    d[ycol+"_adj"] = y.mean() + resid
    return d[["_id", ycol+"_adj"]]

def yearly(df, col, years):
    rows=[]
    for y in years:
        v = pd.to_numeric(df[df.year==y][col], errors="coerce").dropna()
        if len(v)==0: rows.append([y,0,np.nan,np.nan,np.nan,np.nan]); continue
        ci = 1.96*v.std()/np.sqrt(len(v)) if len(v)>1 else np.nan
        rows.append([y,len(v),v.mean(),v.median(),v.std(),ci])
    return pd.DataFrame(rows, columns=["year","n","mean","median","sd","ci95"])

def line(fig_name, title, ylab, series, ylim=None):
    plt=plt_()
    if plt is None: return
    fig,ax=plt.subplots(figsize=(8.5,5))
    for lab,x,yv,*st in series: ax.plot(x,yv,st[0] if st else "-o",label=lab)
    ax.set_xlabel("publication year"); ax.set_ylabel(ylab); ax.set_title(title)
    if ylim: ax.set_ylim(*ylim)
    ax.legend(fontsize=8,ncol=2); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(TRN/(fig_name+".png"),dpi=150); plt.close(fig)

def main():
    TBL.mkdir(parents=True,exist_ok=True); TRN.mkdir(parents=True,exist_ok=True)
    S=load(STAGE5_FINAL)
    if S is None: raise SystemExit(f"missing {STAGE5_FINAL}")
    for c,_ in SCORES:
        if c in S: S[c]=S[c].map(to_num)

    idx=load(GA_INDEX); mst=load(DATASET_MASTER); s2=load(STAGE2_CSV)
    d=S.copy()
    # year + resolution from GA index
    if idx is not None:
        yc=pick(idx,YEAR_CANDS); rc=pick(idx,RES_CANDS)
        cols=["_id"]+[x for x in [yc,rc] if x]
        m=idx[cols].rename(columns={yc:"year_raw"} if yc else {})
        if rc: m=m.rename(columns={rc:"resolution"})
        d=d.merge(m,on="_id",how="left")
    # domain/publisher/journal from dataset_master
    if mst is not None:
        dc=pick(mst,DOM_CANDS); pc=pick(mst,PUB_CANDS); jc=pick(mst,JOUR_CANDS); yc2=pick(mst,YEAR_CANDS)
        keep=["_id"]+[x for x in [dc,pc,jc,yc2] if x]
        mm=mst[keep].rename(columns={c:n for c,n in [(dc,"domain"),(pc,"publisher"),(jc,"journal"),(yc2,"year_master")] if c})
        d=d.merge(mm,on="_id",how="left")
    # OCR confidence from Stage 2
    if s2 is not None:
        oc=pick(s2,OCR_CANDS)
        if oc: d=d.merge(s2[["_id",oc]].rename(columns={oc:"ocr_conf"}),on="_id",how="left")
    # resolve year
    yr = d.get("year_raw"); 
    d["year"] = pd.to_numeric(d.get("year_raw"), errors="coerce")
    if d["year"].isna().all() and "year_master" in d:
        d["year"]=pd.to_numeric(d["year_master"],errors="coerce")
    d["year"]=d["year"].apply(lambda v: int(v) if pd.notna(v) and 1900<=v<=2100 else np.nan)
    d=d.dropna(subset=["year"]); d["year"]=d["year"].astype(int)
    if "resolution" in d: d["log_resolution"]=np.log1p(pd.to_numeric(d["resolution"],errors="coerce"))
    print(f"[stage7] {len(d)} GAs with year; years {sorted(d.year.unique())}")

    years=sorted(d.year.unique())
    # global tertile thresholds on calibrated score
    cal=pd.to_numeric(d["R_calibrated_ridge_01"],errors="coerce")
    t_lo,t_hi=cal.quantile(0.33),cal.quantile(0.67)

    # per-score yearly tables
    for col,label in SCORES:
        if col not in d: continue
        yearly(d,col,years).assign(score=label).to_csv(TBL/f"yearly_{label}.csv",index=False)
    # %low/high per year (calibrated)
    pct=[]
    for y in years:
        v=pd.to_numeric(d[d.year==y]["R_calibrated_ridge_01"],errors="coerce").dropna()
        if len(v)==0: continue
        pct.append([y,len(v),100*(v<t_lo).mean(),100*(v>t_hi).mean()])
    pd.DataFrame(pct,columns=["year","n","pct_low","pct_high"]).to_csv(TBL/"yearly_low_high_pct.csv",index=False)

    # adjusted (confound-residualized) for calibrated + unweighted
    adj_series=[]
    for col,label in [("R_calibrated_ridge_01","calibrated"),("R_overall_unweighted","unweighted")]:
        if col not in d: continue
        adj=residualize(d,col,["domain","publisher"],["log_resolution","ocr_conf"])
        dd=d[["_id","year"]].merge(adj,on="_id",how="inner")
        ya=yearly(dd,col+"_adj",years); ya.assign(score=label+"_adjusted").to_csv(TBL/f"yearly_{label}_adjusted.csv",index=False)
        adj_series.append((label,ya))

    # image quality by year
    iq=[]
    for y in years:
        sub=d[d.year==y]
        iq.append([y,len(sub),
                   pd.to_numeric(sub.get("resolution"),errors="coerce").mean() if "resolution" in sub else np.nan,
                   pd.to_numeric(sub.get("ocr_conf"),errors="coerce").mean() if "ocr_conf" in sub else np.nan])
    IQ=pd.DataFrame(iq,columns=["year","n","mean_resolution","mean_ocr_conf"]); IQ.to_csv(TBL/"yearly_image_quality.csv",index=False)

    # ---- figures ----
    def ym(col): 
        t=yearly(d,col,years); return t.year.values,t["mean"].values
    if "R_calibrated_ridge_01" in d and "R_overall_unweighted" in d:
        xc,yc=ym("R_calibrated_ridge_01"); xu,yu=ym("R_overall_unweighted")
        line("calibrated_vs_unweighted","Readability by year: calibrated vs unweighted","mean readability [0,1]",
             [("calibrated",xc,yc),("unweighted",xu,yu)],ylim=(0,1))
    comp=[(c,l) for c,l in SCORES if l in {"text","visual","semantic"} and c in d]
    if comp:
        line("components_by_year","Readability components by year","mean [0,1]",
             [(l,*ym(c)) for c,l in comp],ylim=(0,1))
    # raw vs adjusted calibrated
    if adj_series:
        xc,yc=ym("R_calibrated_ridge_01")
        ser=[("calibrated raw",xc,yc)]
        for label,ya in adj_series:
            if label=="calibrated": ser.append(("calibrated adjusted",ya.year.values,ya["mean"].values))
        line("calibrated_raw_vs_adjusted","Calibrated readability: raw vs confound-adjusted","mean [0,1]",ser,ylim=(0,1))
    # image quality overlay
    plt=plt_()
    if plt is not None and not IQ.empty:
        fig,ax=plt.subplots(figsize=(8.5,5)); xc,yc=ym("R_calibrated_ridge_01")
        ax.plot(xc,yc,"-o",color="#1C7293",label="calibrated readability"); ax.set_ylim(0,1)
        ax.set_xlabel("publication year"); ax.set_ylabel("readability [0,1]",color="#1C7293")
        ax2=ax.twinx(); ax2.plot(IQ.year,IQ.mean_ocr_conf,"--s",color="#B85042",label="OCR confidence")
        ax2.set_ylabel("mean OCR confidence",color="#B85042")
        ax.set_title("Readability vs image quality (OCR) by year"); fig.tight_layout()
        fig.savefig(TRN/"readability_vs_ocr_by_year.png",dpi=150); plt.close(fig)

    # descriptive slope note
    def slope(col):
        t=yearly(d,col,years).dropna(subset=["mean"])
        if len(t)<3: return float("nan")
        return float(np.polyfit(t.year, t["mean"],1)[0])
    rpt=["Stage 7 FINAL — temporal trends (DESCRIPTIVE; association only)","="*55,"",
         f"GAs with year: {len(d)}   years: {years}","",
         f"calibrated  raw yearly-mean slope/yr : {slope('R_calibrated_ridge_01'):+.5f}",
         f"unweighted  raw yearly-mean slope/yr : {slope('R_overall_unweighted'):+.5f}",
         "",
         "Adjusted trends, %low/high, components, and image-quality tables in tables/.",
         "Figures in trends/. Slopes are descriptive associations, not causal; the",
         "inferential mixed model with controls is Stage 9."]
    (OUT_DIR/"stage7_final_report.txt").write_text("\n".join(rpt)+"\n")
    print("\n"+"\n".join(rpt)+f"\n\nSaved -> {OUT_DIR}")

if __name__=="__main__":
    main()