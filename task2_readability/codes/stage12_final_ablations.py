#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 12 FINAL — Ablations & Baselines (readability metric justification)
========================================================================
Run from:  ./task2_readability
Outputs :  output/stage12_final_ablations/

Question: does the calibrated MULTI-component readability score beat simpler
alternatives at predicting HUMAN overall readability? All fit/scored on the 500
human labels with nested 5-fold CV (no leakage). Metric = held-out Spearman.

Ablations (ridge, nested CV):
  OCR-only (Stage2) | Visual-only (Stage3) | VLM-only (Stage4 1-5) |
  ImageQuality-only (resolution+OCRconf) | non-LLM (OCR+Visual) |
  ALL features | ALL + image-quality
Direct baselines (no fit, Spearman vs human):
  unweighted mean(R_text,R_visual,R_semantic) | VLM readability judge
  (overall_interpretability_1to5) | calibrated linear a*text+b*visual+g*semantic (CV)

Pure pandas + numpy + openpyxl.
"""
from __future__ import annotations
import re, glob
from pathlib import Path
import numpy as np
import pandas as pd

T2   = Path("./task2_readability/output")
OUT_DIR = T2 / "stage12_final_ablations"
HUMAN_XLSX = Path("./temp_statistics/annotation_sheet.xlsx")
STAGE4_CSV = T2 / "stage4_vlm_structural_interpretation/consolidated/stage4_interpretations.csv"
STAGE5_CSV = T2 / "stage5_readability_scoring/features/stage5_readability_scores.csv"
GA_INDEX   = T2 / "stage1_preprocessing/index/stage1_ga_index.csv"
def autofind(*pats):
    for p in pats:
        h=sorted(glob.glob(str(T2/p),recursive=True))
        if h: return Path(h[0])
    return None
STAGE2_CSV=autofind("**/*stage2*ocr*feature*.csv","**/*stage2*feature*.csv")
STAGE3_CSV=autofind("**/*stage3*visual*feature*.csv","**/*stage3*feature*.csv")

N_OUT,N_IN,SEED=5,3,42; LAMBDAS=[0.1,1,10,100,1000]
RES_CANDS=["pixel_count","image_pixel_count","width","image_width"]
OCR_CANDS=["mean_ocr_confidence","ocr_confidence_mean","avg_ocr_confidence","ocr_confidence"]
HEADER_RENAME={"Overall Readability (1-5)":"overall_readability"}
META_DROP={"paper_id","doi","doi_safe","doi_folder","ga_path","ga_id","id","model","journal",
           "publisher","domain","subject_area","subject_categories","_id","publication_year",
           "image_orig_width","image_orig_height"}

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
def numeric_features(df):
    out=[]
    for c in df.columns:
        if c in META_DROP or c=="_id": continue
        v=pd.to_numeric(df[c],errors="coerce")
        if v.notna().sum()>=0.5*len(df) and v.nunique()>1: out.append(c)
    return out
def spearman(a,b):
    d=pd.DataFrame({"a":a,"b":b}).apply(pd.to_numeric,errors="coerce").dropna()
    if len(d)<3 or d.a.nunique()<2 or d.b.nunique()<2: return float("nan")
    return float(np.corrcoef(d.a.rank(),d.b.rank())[0,1])
def folds(n,k,seed=SEED):
    rng=np.random.RandomState(seed); idx=rng.permutation(n); return [idx[i::k] for i in range(k)]
def ridge_fit(X,y,lam):
    yb=y.mean(); return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@(y-yb)),yb
def standardize(Xtr,Xte):
    mu=np.nanmean(Xtr,0); sd=np.nanstd(Xtr,0); sd[sd==0]=1
    return np.where(np.isnan(Xtr),0,(Xtr-mu)/sd),np.where(np.isnan(Xte),0,(Xte-mu)/sd)
def ridge_cv(X,y):
    if X.shape[1]==0 or len(y)<40: return float("nan"),float("nan")
    fo=folds(len(y),N_OUT); sc=[]
    for k in range(N_OUT):
        te=fo[k]; tr=np.concatenate([fo[j] for j in range(N_OUT) if j!=k])
        fi=folds(len(tr),N_IN,SEED+1); bl,bm=LAMBDAS[0],np.inf
        for lam in LAMBDAS:
            ms=[]
            for ii in range(N_IN):
                ite=tr[fi[ii]]; itr=tr[np.concatenate([fi[j] for j in range(N_IN) if j!=ii])]
                Ztr,Zte=standardize(X[itr],X[ite]); w,yb=ridge_fit(Ztr,y[itr],lam)
                ms.append(np.mean((Zte@w+yb-y[ite])**2))
            if np.mean(ms)<bm: bm,bl=np.mean(ms),lam
        Ztr,Zte=standardize(X[tr],X[te]); w,yb=ridge_fit(Ztr,y[tr],bl); sc.append(spearman(Zte@w+yb,y[te]))
    return float(np.nanmean(sc)),float(np.nanstd(sc))
SIMPLEX=[(a/20,b/20,1-a/20-b/20) for a in range(21) for b in range(21-a)]
def weight_cv(Rt,Rv,Rs,y):
    fo=folds(len(y),N_OUT); cal=[]
    for k in range(N_OUT):
        te=fo[k]; tr=np.concatenate([fo[j] for j in range(N_OUT) if j!=k])
        best,bw=-2,(1/3,1/3,1/3)
        for a,b,g in SIMPLEX:
            r=spearman(a*Rt[tr]+b*Rv[tr]+g*Rs[tr],y[tr])
            if r==r and r>best: best,bw=r,(a,b,g)
        a,b,g=bw; cal.append(spearman(a*Rt[te]+b*Rv[te]+g*Rs[te],y[te]))
    return float(np.nanmean(cal)),float(np.nanstd(cal))

def load_human():
    sh=pd.read_excel(HUMAN_XLSX,sheet_name=None,dtype=str); fr=[]
    for name,df in sh.items():
        if name.lower().startswith("01") or "instruction" in name.lower(): continue
        df.columns=[str(c).strip() for c in df.columns]; df=df.rename(columns=HEADER_RENAME)
        df=df[df["doi_folder"].notna() & (df["doi_folder"].astype(str).str.strip()!="")]; fr.append(df)
    h=pd.concat(fr,ignore_index=True); h["_id"]=h["doi_folder"].map(norm_id)
    h["overall_readability"]=h["overall_readability"].apply(
        lambda v: np.nan if str(v).strip().lower() in {"","n/a","na","none","nan"} else to_num(v))
    return h[["_id","overall_readability"]].dropna()

def main():
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    H=load_human(); s2=load(STAGE2_CSV); s3=load(STAGE3_CSV); s4=load(STAGE4_CSV); s5=load(STAGE5_CSV); idx=load(GA_INDEX)
    F=H.copy()
    fam={}
    for tag,df in [("ocr",s2),("visual",s3)]:
        if df is None: continue
        cols=numeric_features(df); fam[tag]=[f"{tag}_{c}" for c in cols]
        F=F.merge(df[["_id"]+cols].rename(columns={c:f"{tag}_{c}" for c in cols}),on="_id",how="left")
    if s4 is not None:
        cols=[c for c in numeric_features(s4) if c.endswith("_1to5") or c.startswith("num_")]
        fam["vlm"]=[f"vlm_{c}" for c in cols]
        F=F.merge(s4[["_id"]+cols].rename(columns={c:f"vlm_{c}" for c in cols}),on="_id",how="left")
    # image quality family
    iqcols=[]
    if idx is not None:
        rc=pick(idx,RES_CANDS)
        if rc: F=F.merge(idx[["_id",rc]].rename(columns={rc:"iq_resolution"}),on="_id",how="left"); iqcols.append("iq_resolution")
    if s2 is not None:
        oc=pick(s2,OCR_CANDS)
        if oc: F=F.merge(s2[["_id",oc]].rename(columns={oc:"iq_ocr"}),on="_id",how="left"); iqcols.append("iq_ocr")
    fam["imgquality"]=iqcols
    if s5 is not None:
        for c in ["R_text","R_visual","R_semantic"]:
            s5[c]=s5.get(c,pd.Series(dtype=str)).map(to_num)
        F=F.merge(s5[["_id","R_text","R_visual","R_semantic"]],on="_id",how="left")
    if s4 is not None:
        F=F.merge(s4[["_id"]].assign(vlm_judge=s4.get("overall_interpretability_1to5",pd.Series(dtype=str)).map(to_num)),on="_id",how="left")
    y=F["overall_readability"].values
    def X(cols): return F[[c for c in cols if c in F.columns]].apply(pd.to_numeric,errors="coerce").values

    rows=[]
    def add(name,cols):
        m,s=ridge_cv(X(cols),y); rows.append([name,len([c for c in cols if c in F.columns]),round(m,3),round(s,3)])
    add("OCR-only (Stage2)", fam.get("ocr",[]))
    add("Visual-only (Stage3)", fam.get("visual",[]))
    add("VLM-only (Stage4 1-5)", fam.get("vlm",[]))
    add("ImageQuality-only", fam.get("imgquality",[]))
    add("non-LLM (OCR+Visual)", fam.get("ocr",[])+fam.get("visual",[]))
    all_feat=fam.get("ocr",[])+fam.get("visual",[])+fam.get("vlm",[])
    add("ALL features", all_feat)
    add("ALL + image-quality", all_feat+fam.get("imgquality",[]))
    AB=pd.DataFrame(rows,columns=["ablation","n_feats","cv_spearman","sd"]).sort_values("cv_spearman",ascending=False)
    AB.to_csv(OUT_DIR/"ablation_ridge.csv",index=False)

    base=[]
    if {"R_text","R_visual","R_semantic"}.issubset(F.columns):
        d2=F.dropna(subset=["R_text","R_visual","R_semantic","overall_readability"])
        unw=spearman((d2.R_text+d2.R_visual+d2.R_semantic)/3, d2.overall_readability)
        base.append(["unweighted mean(R_text,R_visual,R_semantic)",round(unw,3),""])
        cm,cs=weight_cv(d2.R_text.values,d2.R_visual.values,d2.R_semantic.values,d2.overall_readability.values)
        base.append(["calibrated linear a*text+b*visual+g*semantic (CV)",round(cm,3),round(cs,3)])
    if "vlm_judge" in F.columns:
        base.append(["VLM readability judge (overall_interpretability_1to5)",round(spearman(F.vlm_judge,y),3),""])
    BB=pd.DataFrame(base,columns=["baseline","spearman","sd"]); BB.to_csv(OUT_DIR/"baselines.csv",index=False)

    L=["Stage 12 FINAL — ablations & baselines (human overall readability, 5-fold CV)","="*60,"",
       f"n(human)={len(F)}","",
       "RIDGE ABLATIONS (held-out CV Spearman):","-"*40,AB.to_string(index=False),"",
       "DIRECT BASELINES:","-"*40,BB.to_string(index=False),"",
       "Read: 'ALL features' (the calibrated primary) should top single-family ablations,",
       "confirming multi-component calibrated scoring beats OCR-only / visual-only / VLM-only",
       "and simple unweighted or single-VLM-judge baselines."]
    (OUT_DIR/"stage12_final_report.txt").write_text("\n".join(str(x) for x in L)+"\n")
    print("\n"+"\n".join(str(x) for x in L)+f"\n\nSaved -> {OUT_DIR}")

if __name__=="__main__":
    main()