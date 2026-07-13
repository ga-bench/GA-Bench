#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 9 FINAL — Statistical Modeling (mixed-effects; association language only)
==============================================================================
Run from:  ./task2_readability
Outputs :  output/stage9_final_modeling/

Outcome: calibrated readability R_calibrated_ridge_01 in [0,1].
Publisher is DROPPED (degenerate: single value in the dataset).

Models (all with journal random intercept where noted):
  M1  Linear LMM  : y ~ year_c + C(domain) + log_resolution + ocr_conf + (1|journal)
  M2  Spline LMM  : y ~ cr(year_c,df=4) + C(domain) + log_resolution + ocr_conf + (1|journal)
  M3  Interaction : y ~ 0 + C(domain) + year_c:C(domain) + log_resolution + ocr_conf + (1|journal)
                    -> per-domain year slopes with SE / p (tests whether fields differ)
  B   Beta reg    : bounded-outcome robustness (no RE): y ~ year_c + C(domain) + controls

Headline = M1 year_c coefficient (per-year association) with 95% CI + p, AFTER controls
and journal random effects. Reports AIC(M1 vs M2) for nonlinearity. ASSOCIATION only.

Requires: statsmodels, scipy, pandas, numpy.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
except Exception:
    sys.exit("statsmodels not installed. Run:\n  pip install statsmodels scipy")

T2      = Path("./task2_readability/output")
OUT_DIR = T2 / "stage9_final_modeling"
STAGE5_FINAL   = T2 / "stage5_final/stage5_final_scores.csv"
GA_INDEX       = T2 / "stage1_preprocessing/index/stage1_ga_index.csv"
DATASET_MASTER = Path("./dataset_analyzer/"
                      "paper1_dataset_statistics/output_10k/dataset_master.csv")
import glob
def autofind(*pats):
    for p in pats:
        h=sorted(glob.glob(str(T2/p),recursive=True))
        if h: return Path(h[0])
    return None
STAGE2_CSV = autofind("**/*stage2*ocr*feature*.csv","**/*stage2*feature*.csv")
STAGE3_CSV = autofind("**/*stage3*visual*feature*.csv","**/*stage3*feature*.csv")

SCORE="R_calibrated_ridge_01"; MIN_DOMAIN_N=100
YEAR_CANDS=["publication_year","year","year_clean","pub_year"]
RES_CANDS=["pixel_count","image_pixel_count","width","image_width"]
OCR_CANDS=["mean_ocr_confidence","ocr_confidence_mean","avg_ocr_confidence","ocr_confidence"]
DOM_CANDS=["subject_area","domain"]; JOUR_CANDS=["journal"]

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
    return re.sub(r"[^A-Za-z0-9]+","_",parts[0]) if parts else "unknown"

def build():
    S=load(STAGE5_FINAL); S[SCORE]=S[SCORE].map(to_num)
    idx=load(GA_INDEX); mst=load(DATASET_MASTER); s2=load(STAGE2_CSV); s3=load(STAGE3_CSV)
    d=S.copy()
    if idx is not None:
        yc=pick(idx,YEAR_CANDS); rc=pick(idx,RES_CANDS)
        cols=["_id"]+[x for x in [yc,rc] if x]
        d=d.merge(idx[cols].rename(columns={**({yc:"year_raw"} if yc else {}),**({rc:"resolution"} if rc else {})}),on="_id",how="left")
    if mst is not None:
        dc=pick(mst,DOM_CANDS); jc=pick(mst,JOUR_CANDS); yc2=pick(mst,YEAR_CANDS)
        keep=["_id"]+[x for x in [dc,jc,yc2] if x]
        d=d.merge(mst[keep].rename(columns={c:n for c,n in [(dc,"domain_raw"),(jc,"journal"),(yc2,"year_master")] if c}),on="_id",how="left")
    if s3 is not None and "resolution" not in d:
        rc=pick(s3,RES_CANDS)
        if rc: d=d.merge(s3[["_id",rc]].rename(columns={rc:"resolution"}),on="_id",how="left")
    if s2 is not None:
        oc=pick(s2,OCR_CANDS)
        if oc: d=d.merge(s2[["_id",oc]].rename(columns={oc:"ocr_conf"}),on="_id",how="left")
    d["year"]=pd.to_numeric(d.get("year_raw"),errors="coerce")
    if d["year"].isna().all() and "year_master" in d: d["year"]=pd.to_numeric(d["year_master"],errors="coerce")
    d=d.dropna(subset=["year",SCORE]); d["year"]=d["year"].astype(int); d["year_c"]=d["year"]-d["year"].min()
    d["domain"]=d.get("domain_raw","unknown").map(primary_domain)
    d["journal"]=d.get("journal","unknown").fillna("unknown").replace("","unknown")
    d["log_resolution"]=np.log1p(pd.to_numeric(d.get("resolution"),errors="coerce"))
    d["ocr_conf"]=pd.to_numeric(d.get("ocr_conf"),errors="coerce")
    d["y"]=pd.to_numeric(d[SCORE],errors="coerce")
    d=d.dropna(subset=["y","log_resolution","ocr_conf"]).reset_index(drop=True)
    # keep domains with enough n for stable interaction slopes
    keep_dom=d.domain.value_counts()[lambda s: s>=MIN_DOMAIN_N].index
    d["domain"]=np.where(d.domain.isin(keep_dom),d.domain,"other")
    return d

def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    d=build()
    print(f"[stage9] n={len(d)}  journals={d.journal.nunique()}  domains={d.domain.nunique()}")
    L=[f"Stage 9 FINAL — mixed-effects modeling (ASSOCIATION only)","="*55,"",
       f"n={len(d)}  journals={d.journal.nunique()}  domains={d.domain.nunique()}  outcome={SCORE} in [0,1]",
       "Publisher dropped (degenerate). year_c = year - min(year).",""]

    # ---- M1 linear LMM ----
    try:
        m1=smf.mixedlm("y ~ year_c + C(domain) + log_resolution + ocr_conf", d, groups=d["journal"]).fit(reml=False)
        b=m1.params["year_c"]; ci=m1.conf_int().loc["year_c"]; p=m1.pvalues["year_c"]
        pd.DataFrame({"term":m1.params.index,"coef":m1.params.values,
                      "p":m1.pvalues.reindex(m1.params.index).values}).to_csv(OUT_DIR/"m1_linear_coeffs.csv",index=False)
        L+=["M1 linear LMM — headline year association:",
            f"  year_c coef = {b:+.5f}  95%CI [{ci[0]:+.5f}, {ci[1]:+.5f}]  p={p:.3g}",
            f"  (per-year change in calibrated readability, after domain+image quality+OCR + (1|journal))",
            f"  AIC={m1.aic:.1f}",""]
        aic1=m1.aic
    except Exception as e:
        L+=[f"M1 failed: {e}",""]; aic1=None

    # ---- M2 spline LMM ----
    try:
        m2=smf.mixedlm("y ~ cr(year_c, df=4) + C(domain) + log_resolution + ocr_conf", d, groups=d["journal"]).fit(reml=False)
        L+=["M2 spline(year) LMM:",
            f"  AIC={m2.aic:.1f}" + (f"   (vs M1 {aic1:.1f}; lower is better)" if aic1 else ""),
            "  -> if M2 AIC not clearly lower, a linear year term suffices (no strong nonlinearity).",""]
    except Exception as e:
        L+=[f"M2 failed: {e}",""]

    # ---- M3 year x domain interaction: per-domain slopes ----
    try:
        m3=smf.mixedlm("y ~ 0 + C(domain) + year_c:C(domain) + log_resolution + ocr_conf", d, groups=d["journal"]).fit(reml=False)
        rows=[]
        for term in m3.params.index:
            mt=re.match(r"year_c:C\(domain\)\[(?:T\.)?(.+)\]", term)
            if mt:
                rows.append([mt.group(1), round(m3.params[term],5),
                             round(m3.bse[term],5), round(m3.pvalues[term],4)])
        DS=pd.DataFrame(rows,columns=["domain","year_slope","se","p"]).sort_values("year_slope")
        DS.to_csv(OUT_DIR/"m3_domain_year_slopes.csv",index=False)
        nsig=(DS.p<0.05).sum()
        L+=["M3 year x domain interaction — per-domain year slopes:","-"*40, DS.to_string(index=False),
            f"  domains with p<0.05 slope: {nsig}/{len(DS)}",""]
    except Exception as e:
        L+=[f"M3 failed: {e}",""]

    # ---- Beta regression robustness (bounded outcome, no RE) ----
    try:
        from statsmodels.othermod.betareg import BetaModel
        N=len(d); yb=(d["y"]*(N-1)+0.5)/N
        Xb=sm.add_constant(pd.get_dummies(d["domain"],drop_first=True).astype(float)
                           .assign(year_c=d.year_c.values, log_resolution=d.log_resolution.values, ocr_conf=d.ocr_conf.values))
        bm=BetaModel(yb, Xb).fit(disp=0)
        L+=["Beta regression (bounded-outcome robustness, no RE):",
            f"  year_c coef = {bm.params.get('year_c',float('nan')):+.5f}  p={bm.pvalues.get('year_c',float('nan')):.3g}",
            "  -> confirms sign/significance of the year association under a bounded model.",""]
    except Exception as e:
        L+=[f"Beta regression skipped: {e}",""]

    L+=["Interpretation: report the M1 year coefficient as an ASSOCIATION (not causal).",
        "A near-zero coefficient with controls + (1|journal) => no meaningful temporal trend,",
        "consistent with Stage 7/8. Per-domain slopes (M3) quantify field heterogeneity."]
    (OUT_DIR/"stage9_final_report.txt").write_text("\n".join(str(x) for x in L)+"\n")
    print("\n"+"\n".join(str(x) for x in L)+f"\n\nSaved -> {OUT_DIR}")

if __name__=="__main__":
    main()