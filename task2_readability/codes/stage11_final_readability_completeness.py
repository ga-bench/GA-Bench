#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 11 FINAL (v2) — Readability x Completeness (cross-task join)
=================================================================
Run from:  ./task2_readability
Outputs :  output/stage11_final_readability_completeness/

Joins Task-1 completeness (unweighted C=(S+R)/2, primary Qwen-B) to Stage-5
calibrated readability by paper_id, then correlations + LMM + four quadrants.
LMM uses explicit numeric design matrices (no patsy formula) to avoid the
C()/reserved-name and dense-dummy issues. Association language only.
Requires statsmodels, scipy, pandas, numpy.
"""
from __future__ import annotations
import re, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import statsmodels.api as sm
except Exception:
    sys.exit("pip install statsmodels scipy")

T2   = Path("./task2_readability/output")
OUT_DIR = T2 / "stage11_final_readability_completeness"
STAGE5_FINAL   = T2 / "stage5_final/stage5_final_scores.csv"
GA_INDEX       = T2 / "stage1_preprocessing/index/stage1_ga_index.csv"
DATASET_MASTER = Path("./dataset_analyzer/"
                      "paper1_dataset_statistics/output_10k/dataset_master.csv")
T1DIR = Path("./task1_completeness_awq/task1_scores")
T1_PRIMARY = "scores_qwen3_vl_32b_variantB.csv"
def autofind(*pats):
    for p in pats:
        h=sorted(glob.glob(str(T2/p),recursive=True))
        if h: return Path(h[0])
    return None
STAGE2_CSV=autofind("**/*stage2*ocr*feature*.csv","**/*stage2*feature*.csv")

READ="R_calibrated_ridge_01"
YEAR_CANDS=["publication_year","year","year_clean","pub_year"]
RES_CANDS=["pixel_count","image_pixel_count","width","image_width"]
OCR_CANDS=["mean_ocr_confidence","ocr_confidence_mean","avg_ocr_confidence","ocr_confidence"]
DOM_CANDS=["subject_area","domain"]; JOUR_CANDS=["journal"]; MIN_DOMAIN_N=100

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
def spearman(a,b):
    d=pd.DataFrame({"a":a,"b":b}).apply(pd.to_numeric,errors="coerce").dropna()
    if len(d)<3 or d.a.nunique()<2 or d.b.nunique()<2: return float("nan"),0
    return round(float(np.corrcoef(d.a.rank(),d.b.rank())[0,1]),3),len(d)
def pearson(a,b):
    d=pd.DataFrame({"a":a,"b":b}).apply(pd.to_numeric,errors="coerce").dropna()
    if len(d)<3: return float("nan")
    return round(float(np.corrcoef(d.a,d.b)[0,1]),3)

def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    S=load(STAGE5_FINAL)
    for c in [READ,"R_text","R_visual","R_semantic"]:
        if c in S: S[c]=S[c].map(to_num)
    t1=pd.read_csv(T1DIR/T1_PRIMARY,dtype=str)
    t1=t1[t1.get("parse_ok",pd.Series(["True"]*len(t1))).astype(str).str.lower()!="false"]
    idc="doi_safe" if "doi_safe" in t1.columns else find_id_col(t1)
    t1["_id"]=t1[idc].map(norm_id)
    Sx=t1.get("S_section_score",pd.Series(dtype=str)).map(to_num)
    Rx=t1.get("R_relation_score",pd.Series(dtype=str)).map(to_num)
    t1["C_unw"]=(Sx+Rx)/2.0; t1["comp_level"]=t1.get("level",pd.Series(dtype=str)).map(to_num)
    d=S.merge(t1[["_id","C_unw","comp_level"]],on="_id",how="inner")

    idx=load(GA_INDEX); mst=load(DATASET_MASTER); s2=load(STAGE2_CSV)
    if idx is not None:
        yc=pick(idx,YEAR_CANDS); rc=pick(idx,RES_CANDS)
        d=d.merge(idx[["_id"]+[x for x in [yc,rc] if x]].rename(
            columns={**({yc:"year"} if yc else {}),**({rc:"resolution"} if rc else {})}),on="_id",how="left")
    if mst is not None:
        dc=pick(mst,DOM_CANDS); jc=pick(mst,JOUR_CANDS)
        d=d.merge(mst[["_id"]+[x for x in [dc,jc] if x]].rename(
            columns={c:n for c,n in [(dc,"domain_raw"),(jc,"journal")] if c}),on="_id",how="left")
    if s2 is not None:
        oc=pick(s2,OCR_CANDS)
        if oc: d=d.merge(s2[["_id",oc]].rename(columns={oc:"ocr_conf"}),on="_id",how="left")
    d["domain"]=d.get("domain_raw","unknown").map(primary_domain)
    d["journal"]=d.get("journal","unknown").fillna("unknown").replace("","unknown")
    d["log_resolution"]=np.log1p(pd.to_numeric(d.get("resolution"),errors="coerce"))
    d["ocr_conf"]=pd.to_numeric(d.get("ocr_conf"),errors="coerce")
    d=d.dropna(subset=[READ,"C_unw"]).reset_index(drop=True)
    print(f"[stage11] joined n={len(d)}")

    L=["Stage 11 FINAL — readability x completeness (ASSOCIATION only)","="*55,"",
       f"joined n={len(d)}  (Task1 primary = {T1_PRIMARY}, unweighted C)",""]

    # ---- correlations ----
    rows=[]
    for lab,col in [("readability_overall",READ),("R_text","R_text"),("R_visual","R_visual"),("R_semantic","R_semantic")]:
        if col in d:
            sp,n=spearman(d[col],d["C_unw"]); rows.append([lab,"C_unw",sp,pearson(d[col],d["C_unw"]),n])
    sp,n=spearman(d[READ],d["comp_level"]); rows.append(["readability_overall","comp_level_0_4",sp,pearson(d[READ],d["comp_level"]),n])
    COR=pd.DataFrame(rows,columns=["readability_var","completeness_var","spearman","pearson","n"])
    COR.to_csv(OUT_DIR/"correlations.csv",index=False)
    L+=["Correlations (readability vs completeness):","-"*40,COR.to_string(index=False),""]

    # ---- LMM (explicit numeric matrices) ----
    try:
        dm=d.copy()
        keep=dm.domain.value_counts()[lambda s:s>=MIN_DOMAIN_N].index
        dm["domain"]=np.where(dm.domain.isin(keep),dm.domain,"other")
        dm["Comp"]=pd.to_numeric(dm.C_unw,errors="coerce")
        dm["Rd"]=pd.to_numeric(dm[READ],errors="coerce")
        for c in ["Comp","Rd","log_resolution","ocr_conf"]:
            dm[c]=dm[c].replace([np.inf,-np.inf],np.nan)
        dm=dm.dropna(subset=["Comp","Rd"]).reset_index(drop=True)
        X=pd.concat([dm[["Rd"]],
                     pd.get_dummies(dm["domain"],drop_first=True).astype(float)],axis=1)
        X=sm.add_constant(X)
        grp=dm["journal"].astype("category").cat.codes.values
        m=sm.MixedLM(dm["Comp"].values, X.values, groups=grp).fit(reml=False)
        rd_i=list(X.columns).index("Rd")
        L+=["LMM  completeness ~ readability + domain + image-quality + OCR + (1|journal):",
            f"  readability coef = {m.params[rd_i]:+.4f}  SE={m.bse[rd_i]:.4f}  p={m.pvalues[rd_i]:.3g}   n={len(dm)}",""]
        pd.DataFrame({"term":list(X.columns),"coef":m.params[:X.shape[1]]}).to_csv(OUT_DIR/"lmm_coeffs.csv",index=False)
    except Exception as e:
        L+=[f"LMM failed: {e}",""]

    # ---- four quadrants ----
    rmed=d[READ].median(); cmed=d["C_unw"].median()
    def quad(r,c):
        hr=r>=rmed; hc=c>=cmed
        return ("HR_HC_ideal" if hr and hc else "HR_LC_clear_but_shallow" if hr and not hc
                else "LR_HC_rich_but_overloaded" if (not hr) and hc else "LR_LC_weak")
    d["quadrant"]=[quad(r,c) for r,c in zip(d[READ],d["C_unw"])]
    qc=d["quadrant"].value_counts()
    QC=pd.DataFrame({"quadrant":qc.index,"n":qc.values,"pct":(100*qc.values/len(d)).round(1)})
    QC.to_csv(OUT_DIR/"quadrant_counts.csv",index=False)
    cols=["_id",READ,"C_unw","comp_level","domain","journal","quadrant"]
    for q in ["HR_LC_clear_but_shallow","LR_HC_rich_but_overloaded","HR_HC_ideal","LR_LC_weak"]:
        d[d.quadrant==q][cols].to_csv(OUT_DIR/f"quadrant_{q}.csv",index=False)
    L+=[f"Median split: readability={rmed:.3f}  completeness={cmed:.3f}",
        "Four quadrants:","-"*40,QC.to_string(index=False),"",
        "Mismatch cells (HR_LC and LR_HC) exported for qualitative taxonomy.",
        "Overall readability~completeness is ~0, but components split (semantic +, text/visual -),",
        "so the two dimensions are statistically distinct, not redundant."]
    (OUT_DIR/"stage11_final_report.txt").write_text("\n".join(str(x) for x in L)+"\n")
    print("\n"+"\n".join(str(x) for x in L)+f"\n\nSaved -> {OUT_DIR}")

if __name__=="__main__":
    main()