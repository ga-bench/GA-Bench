#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 FINAL — Human-Calibrated Readability (deploy over full dataset)
======================================================================
Run from:  ./task2_readability
Outputs :  task2_readability/output/stage5_final/

Fits the calibrated readability score on the 500 human-labeled papers ONLY
(no leakage: humans never saw the other ~9,500), then applies it to every GA.

Primary  : feature-level RIDGE  (Stage2 OCR + Stage3 visual + Stage4 Qwen 1-5)
           -> R_calibrated_ridge   (best human agreement)
Secondary: interpretable LINEAR  R = a*R_text + b*R_visual + g*R_semantic
           -> R_calibrated_linear  (matches proposal's Stage-6 form)
Baseline : R_overall_unweighted = mean(R_text,R_visual,R_semantic)   (unchanged)

Honest validation = nested 5-fold CV Spearman on the 500 (reported).
Deployment model  = refit on all 500 (lambda / weights picked by 5-fold), saved to JSON.
Qwen primary; InternVL dropped. Pure pandas + numpy + openpyxl.
"""
from __future__ import annotations
import re, glob, json
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
T2      = Path("./task2_readability/output")
OUT_DIR = T2 / "stage5_final"
HUMAN_XLSX = Path("./temp_statistics/annotation_sheet.xlsx")

def autofind(*pats):
    for p in pats:
        h = sorted(glob.glob(str(T2 / p), recursive=True))
        if h: return Path(h[0])
    return None
STAGE2_CSV = autofind("**/*stage2*ocr*feature*.csv", "**/*stage2*feature*.csv")
STAGE3_CSV = autofind("**/*stage3*visual*feature*.csv", "**/*stage3*feature*.csv")
STAGE4_CSV = T2 / "stage4_vlm_structural_interpretation/consolidated/stage4_interpretations.csv"  # Qwen
STAGE5_CSV = T2 / "stage5_readability_scoring/features/stage5_readability_scores.csv"             # Qwen

N_OUT, N_IN, SEED = 5, 3, 42
LAMBDAS = [0.1, 1, 10, 100, 1000]
ID_CANDIDATES = ["paper_id", "doi_folder", "doi_safe", "doi", "ga_id", "GA_ID", "id"]
HEADER_RENAME = {"Overall Readability (1-5)": "overall_readability"}
META_DROP = {"paper_id","doi","doi_safe","doi_folder","ga_path","ga_id","id","model","journal",
             "publisher","domain","subject_area","subject_categories","_id","publication_year",
             "image_orig_width","image_orig_height"}

# =============================================================================
# HELPERS
# =============================================================================
def norm_id(s):
    s = str(s or "").strip().lower().replace("https://doi.org/","").replace("doi:","")
    return re.sub(r"[^a-z0-9]+","_",s).strip("_")
def find_id_col(df):
    for c in ID_CANDIDATES:
        if c in df.columns: return c
    raise SystemExit(f"no id col {list(df.columns)[:6]}")
def to_num(x):
    try: return float(x)
    except Exception: return np.nan
def spearman(a,b):
    d=pd.DataFrame({"a":a,"b":b}).apply(pd.to_numeric,errors="coerce").dropna()
    if len(d)<3 or d.a.nunique()<2 or d.b.nunique()<2: return float("nan")
    return float(np.corrcoef(d.a.rank(),d.b.rank())[0,1])
def folds(n,k,seed=SEED):
    rng=np.random.RandomState(seed); idx=rng.permutation(n); return [idx[i::k] for i in range(k)]
def load_csv(p):
    if p is None or not Path(p).exists(): return None
    df=pd.read_csv(p,dtype=str); df["_id"]=df[find_id_col(df)].map(norm_id); return df
def numeric_features(df):
    out=[]
    for c in df.columns:
        if c in META_DROP or c=="_id": continue
        v=pd.to_numeric(df[c],errors="coerce")
        if v.notna().sum()>=0.5*len(df) and v.nunique()>1: out.append(c)
    return out

def ridge_fit(X,y,lam):
    yb=y.mean(); w=np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@(y-yb)); return w,yb
def pick_lambda(X,y):
    fi=folds(len(y),N_IN,SEED+1); best,bl=np.inf,LAMBDAS[0]
    for lam in LAMBDAS:
        ms=[]
        for ii in range(N_IN):
            te=fi[ii]; tr=np.concatenate([fi[j] for j in range(N_IN) if j!=ii])
            mu=np.nanmean(X[tr],0); sd=np.nanstd(X[tr],0); sd[sd==0]=1
            Ztr=np.where(np.isnan(X[tr]),0,(X[tr]-mu)/sd); Zte=np.where(np.isnan(X[te]),0,(X[te]-mu)/sd)
            w,yb=ridge_fit(Ztr,y[tr],lam); ms.append(np.mean((Zte@w+yb-y[te])**2))
        if np.mean(ms)<best: best,bl=np.mean(ms),lam
    return bl
def nested_cv_spearman(X,y):
    fo=folds(len(y),N_OUT); sc=[]
    for k in range(N_OUT):
        te=fo[k]; tr=np.concatenate([fo[j] for j in range(N_OUT) if j!=k])
        lam=pick_lambda(X[tr],y[tr])
        mu=np.nanmean(X[tr],0); sd=np.nanstd(X[tr],0); sd[sd==0]=1
        Ztr=np.where(np.isnan(X[tr]),0,(X[tr]-mu)/sd); Zte=np.where(np.isnan(X[te]),0,(X[te]-mu)/sd)
        w,yb=ridge_fit(Ztr,y[tr],lam); sc.append(spearman(Zte@w+yb,y[te]))
    return float(np.nanmean(sc)), float(np.nanstd(sc))

SIMPLEX=[(a/20,b/20,1-a/20-b/20) for a in range(21) for b in range(21-a)]
def fit_weights(Rt,Rv,Rs,y):
    best,bw=-2,(1/3,1/3,1/3)
    for a,b,g in SIMPLEX:
        r=spearman(a*Rt+b*Rv+g*Rs,y)
        if r==r and r>best: best,bw=r,(a,b,g)
    return bw, best
def weight_cv(Rt,Rv,Rs,y):
    fo=folds(len(y),N_OUT); cal=[]
    for k in range(N_OUT):
        te=fo[k]; tr=np.concatenate([fo[j] for j in range(N_OUT) if j!=k])
        (a,b,g),_=fit_weights(Rt[tr],Rv[tr],Rs[tr],y[tr]); cal.append(spearman(a*Rt[te]+b*Rv[te]+g*Rs[te],y[te]))
    return float(np.nanmean(cal)), float(np.nanstd(cal))

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

# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[paths]\n s2={STAGE2_CSV}\n s3={STAGE3_CSV}\n s4={STAGE4_CSV}\n s5={STAGE5_CSV}")
    H=load_human(); s2=load_csv(STAGE2_CSV); s3=load_csv(STAGE3_CSV)
    s4=load_csv(STAGE4_CSV); s5=load_csv(STAGE5_CSV)
    if s5 is None: raise SystemExit("Stage5 (component) scores required")

    # ----- build full-dataset feature frame keyed by _id -----
    base = s5[["_id"]].drop_duplicates().copy()
    for c in ["R_text","R_visual","R_semantic"]:
        base[c]=s5.set_index("_id")[c].map(to_num).reindex(base["_id"]).values if c in s5 else np.nan
    base["R_overall_unweighted"]=base[["R_text","R_visual","R_semantic"]].mean(axis=1)

    F = base.copy()
    fam = {}
    for tag, df in [("s2",s2),("s3",s3),("s4",s4)]:
        if df is None: continue
        cols=numeric_features(df)
        if tag=="s4": cols=[c for c in cols if c.endswith("_1to5") or c.startswith("num_")]
        fam[tag]=[f"{tag}_{c}" for c in cols]
        F=F.merge(df[["_id"]+cols].rename(columns={c:f"{tag}_{c}" for c in cols}), on="_id", how="left")
    feat_cols=[c for fam_c in fam.values() for c in fam_c if c in F.columns]
    print(f"[features] {len(feat_cols)} across {len(F)} GAs")

    # ----- fit set = human-labeled subset -----
    fit = F.merge(H, on="_id", how="inner").reset_index(drop=True)
    y = fit["overall_readability"].values
    Xfit = fit[feat_cols].apply(pd.to_numeric, errors="coerce").values
    print(f"[fit] {len(fit)} human-labeled papers")

    # ----- RIDGE: honest CV + deployment refit -----
    cv_m, cv_sd = nested_cv_spearman(Xfit, y)
    lam = pick_lambda(Xfit, y)
    mu = np.nanmean(Xfit,0); sd = np.nanstd(Xfit,0); sd[sd==0]=1
    Zfit = np.where(np.isnan(Xfit),0,(Xfit-mu)/sd)
    w, yb = ridge_fit(Zfit, y, lam)

    # ----- LINEAR abg: CV + deployment -----
    lin_cols = {"R_text","R_visual","R_semantic"}.issubset(fit.columns)
    if lin_cols:
        d2 = fit.dropna(subset=["R_text","R_visual","R_semantic"])
        Rt,Rv,Rs,yy = d2.R_text.values,d2.R_visual.values,d2.R_semantic.values,d2.overall_readability.values
        lin_cv_m, lin_cv_sd = weight_cv(Rt,Rv,Rs,yy)
        (a,b,g), lin_full = fit_weights(Rt,Rv,Rs,yy)
    else:
        lin_cv_m=lin_cv_sd=a=b=g=float("nan")

    # ----- APPLY to full dataset -----
    Xall = F[feat_cols].apply(pd.to_numeric, errors="coerce").values
    Zall = np.where(np.isnan(Xall),0,(Xall-mu)/sd)
    ridge_pred = Zall @ w + yb
    F["R_calibrated_ridge_1to5"] = np.clip(ridge_pred, 1, 5)
    rp = F["R_calibrated_ridge_1to5"]; F["R_calibrated_ridge_01"] = (rp-1)/4.0
    if lin_cols:
        lin = a*F["R_text"]+b*F["R_visual"]+g*F["R_semantic"]
        F["R_calibrated_linear_01"] = lin
    out_cols=["_id","R_text","R_visual","R_semantic","R_overall_unweighted",
              "R_calibrated_ridge_1to5","R_calibrated_ridge_01"] + (["R_calibrated_linear_01"] if lin_cols else [])
    F[out_cols].to_csv(OUT_DIR/"stage5_final_scores.csv", index=False)

    # ----- save deployment model -----
    model={"formula_primary":"ridge over Stage2+Stage3+Stage4(Qwen) features, target=human overall 1-5",
           "ridge_lambda":lam, "ridge_intercept":float(yb),
           "feature_columns":feat_cols,
           "feature_mean":mu.tolist(), "feature_std":sd.tolist(), "ridge_weights":w.tolist(),
           "linear_weights":{"alpha_text":a,"beta_visual":b,"gamma_semantic":g},
           "fit_n":int(len(fit)), "seed":SEED}
    (OUT_DIR/"stage5_final_model.json").write_text(json.dumps(model,indent=2))

    rpt=[f"Stage 5 FINAL calibrated readability — deployed over {len(F)} GAs","="*60,"",
         f"Fit on {len(fit)} human-labeled papers (no leakage).","",
         "PRIMARY  ridge:",
         f"  nested 5-fold CV Spearman = {cv_m:.3f} +/- {cv_sd:.3f}",
         f"  deployment lambda = {lam}   (#features={len(feat_cols)})","",
         "SECONDARY linear a*text+b*visual+g*semantic:",
         f"  5-fold CV Spearman = {lin_cv_m:.3f} +/- {lin_cv_sd:.3f}",
         f"  deploy weights a,b,g = {a:.2f}, {b:.2f}, {g:.2f}","",
         "Outputs: stage5_final_scores.csv (per-GA), stage5_final_model.json",
         "Columns: R_overall_unweighted (baseline), R_calibrated_ridge_* (primary),",
         "         R_calibrated_linear_01 (interpretable secondary)."]
    (OUT_DIR/"stage5_final_report.txt").write_text("\n".join(rpt)+"\n")
    print("\n"+"\n".join(rpt)+f"\n\nSaved -> {OUT_DIR}")

if __name__=="__main__":
    main()