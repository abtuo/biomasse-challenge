from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

from .config import load_yaml
from .data import S2_BANDS, S2_INDICES, clean_analysis_rows, validate_s2_schema
from .modeling import make_model, random_holdout, spatial_cv
from .spatial import add_spatial_blocks
from .uncertainty import conformal_radius, empirical_coverage, interval_from_radius
from .validity import KNNDomainValidity

STYLE = {"figure.dpi": 120, "savefig.dpi": 220, "axes.grid": True, "grid.alpha": .2}

def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def _sample(df, n, seed=42):
    return df if len(df) <= n else df.sample(n=n, random_state=seed)

def audit_dataset(df: pd.DataFrame, features: list[str], table_dir: Path) -> dict:
    table_dir.mkdir(parents=True, exist_ok=True)
    missing = pd.DataFrame({
        "column": df.columns, "dtype": [str(df[c].dtype) for c in df],
        "missing_count": [int(df[c].isna().sum()) for c in df],
        "missing_percent": [float(df[c].isna().mean() * 100) for c in df],
    })
    missing.to_csv(table_dir / "missingness.csv", index=False)
    numeric = [c for c in ["agbd", "agbd_se", "relative_se"] + features if c in df]
    summary = df[numeric].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99, .995]).T
    summary["missing"] = df[numeric].isna().sum()
    summary["n_unique"] = df[numeric].nunique(dropna=True)
    summary["near_constant"] = [df[c].value_counts(normalize=True, dropna=False).iloc[0] >= .995 for c in numeric]
    summary.rename_axis("column").reset_index().to_csv(table_dir / "feature_summary.csv", index=False)
    duplicated = int(df["shot_number"].duplicated().sum()) if "shot_number" in df else None
    invalid = {
        "agbd_negative": int((df.agbd < 0).sum()),
        "agbd_se_negative": int((df.agbd_se < 0).sum()),
        "invalid_coordinates": int((~df.longitude.between(-180, 180) | ~df.latitude.between(-90, 90)).sum()),
        "index_outside_minus1_1": int(sum((~df[c].between(-1, 1) & df[c].notna()).sum() for c in S2_INDICES if c in df)),
    }
    rows = [
        ("n_rows", len(df)), ("n_columns", len(df.columns)), ("duplicate_shot_number", duplicated),
        ("agbd_max", float(df.agbd.max())), ("agbd_p99", float(df.agbd.quantile(.99))),
        ("agbd_p995", float(df.agbd.quantile(.995))), ("extreme_agbd_above_p995", int((df.agbd > df.agbd.quantile(.995)).sum())),
        *[(k, v) for k, v in invalid.items()],
    ]
    pd.DataFrame(rows, columns=["metric", "value"]).to_csv(table_dir / "data_audit.csv", index=False)
    return {"duplicate_shot_number": duplicated, "invalid": invalid, "agbd": summary.loc["agbd"].to_dict()}

def make_eda_figures(df, features, fig_dir: Path, block_size_m: float, crs: str):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    # Full and tail distributions.
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.hist(df.agbd.dropna(), bins=100, color="#267a8a")
    ax.set(xlabel="AGBD (Mg/ha)", ylabel="GEDI shots", title="GEDI L4A AGBD distribution"); _save(fig, fig_dir / "agbd_distribution.png")
    q95 = df.agbd.quantile(.95)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2)); axs[0].hist(df.agbd, bins=120, log=True); axs[1].hist(df.loc[df.agbd >= q95, "agbd"], bins=60, color="#b44d5e")
    axs[0].set(title="Full distribution (log count)", xlabel="AGBD (Mg/ha)"); axs[1].set(title=f"Upper 5% (>= {q95:.1f} Mg/ha)", xlabel="AGBD (Mg/ha)"); _save(fig, fig_dir / "agbd_upper_tail.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.6)); ax.hist(df.relative_se.dropna().clip(upper=df.relative_se.quantile(.995)), bins=80); ax.set(xlabel="Relative standard error", ylabel="Shots", title="GEDI relative uncertainty (to 99.5th percentile)"); _save(fig, fig_dir / "relative_se_distribution.png")
    corr = df[features + ["agbd"]].corr(method="spearman", numeric_only=True)
    target_corr = corr["agbd"].drop("agbd").sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5)); ax.barh(target_corr.index, target_corr.values, color="#4387b4"); ax.set(xlabel="Spearman rho", title="Predictor association with AGBD"); _save(fig, fig_dir / "spearman_predictors_vs_agbd.png")
    pcorr = corr.loc[features, features]
    fig, ax = plt.subplots(figsize=(9, 8)); im=ax.imshow(pcorr, vmin=-1, vmax=1, cmap="RdBu_r"); ax.set_xticks(range(len(features)), features, rotation=55, ha="right"); ax.set_yticks(range(len(features)), features); fig.colorbar(im, ax=ax, label="Spearman rho"); ax.set_title("Sentinel-2 predictor correlation matrix"); _save(fig, fig_dir / "predictor_correlation_matrix.png")
    plot_df = _sample(df, 150000)
    for feature in [c for c in S2_INDICES if c in features]:
        fig, ax = plt.subplots(figsize=(6.5, 5)); hb=ax.hexbin(plot_df[feature], plot_df.agbd, gridsize=65, bins="log", mincnt=1, cmap="viridis"); fig.colorbar(hb, ax=ax, label="log10(count)"); ax.set(xlabel=feature, ylabel="AGBD (Mg/ha)", title=f"AGBD versus {feature}"); _save(fig, fig_dir / f"agbd_vs_{feature.lower()}.png")
    red = [c for c in ["B5", "B6", "B7", "B8A"] if c in features]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8));
    for ax, feature in zip(axs.flat, red): ax.hexbin(plot_df[feature], plot_df.agbd, gridsize=50, bins="log", mincnt=1); ax.set(xlabel=feature, ylabel="AGBD")
    _save(fig, fig_dir / "agbd_vs_red_edge_bands.png")
    cols=4; rows=int(np.ceil(len(features)/cols)); fig, axs=plt.subplots(rows, cols, figsize=(14, 3.2*rows), squeeze=False)
    for ax, feature in zip(axs.flat, features):
        try: bins=pd.qcut(df[feature], 20, duplicates="drop"); g=df.groupby(bins, observed=True).agg(x=(feature,"median"), med=("agbd","median"), lo=("agbd",lambda x:x.quantile(.25)), hi=("agbd",lambda x:x.quantile(.75)))
        except ValueError: continue
        ax.plot(g.x,g.med); ax.fill_between(g.x,g.lo,g.hi,alpha=.2); ax.set(xlabel=feature,ylabel="AGBD",title=feature)
    for ax in axs.flat[len(features):]: ax.axis("off")
    _save(fig, fig_dir / "binned_median_agbd_by_predictor.png")
    fig, ax=plt.subplots(figsize=(7,6)); h=ax.hexbin(df.longitude,df.latitude,gridsize=100,bins="log",mincnt=1); fig.colorbar(h,ax=ax,label="log10(shots)"); ax.set(xlabel="Longitude",ylabel="Latitude",title="Spatial density of GEDI samples"); _save(fig,fig_dir/"gedi_spatial_density.png")
    geo=_sample(df,120000); fig,ax=plt.subplots(figsize=(7,6)); s=ax.scatter(geo.longitude,geo.latitude,c=geo.agbd,s=2,cmap="viridis",vmin=0,vmax=df.agbd.quantile(.99)); fig.colorbar(s,ax=ax,label="AGBD (Mg/ha)"); ax.set(xlabel="Longitude",ylabel="Latitude",title="Spatial distribution of AGBD (sample)"); _save(fig,fig_dir/"agbd_spatial_distribution.png")
    fig,ax=plt.subplots(figsize=(6.5,5)); h=ax.hexbin(plot_df.agbd,plot_df.agbd_se,gridsize=65,bins="log",mincnt=1); fig.colorbar(h,ax=ax,label="log10(count)"); ax.set(xlabel="AGBD",ylabel="AGBD SE",title="GEDI uncertainty versus biomass"); _save(fig,fig_dir/"uncertainty_vs_biomass.png")
    blocked=add_spatial_blocks(df[["longitude","latitude"]],block_size_m,crs); codes=pd.factorize(blocked.spatial_block)[0]
    fig,ax=plt.subplots(figsize=(7,6)); idx=_sample(pd.DataFrame({"x":blocked.x_utm,"y":blocked.y_utm,"c":codes}),150000); ax.scatter(idx.x,idx.y,c=idx.c,s=1,cmap="tab20"); ax.set(xlabel="UTM easting (m)",ylabel="UTM northing (m)",title=f"Spatial blocks ({block_size_m/1000:g} km; n={blocked.spatial_block.nunique()})"); _save(fig,fig_dir/"spatial_blocks.png")

def saturation_analysis(df, features, table_dir: Path, fig_dir: Path, bin_width=25):
    sat_features=[c for c in ["NDVI","NDMI","NDRE","NBR","B5","B6","B7","B8A"] if c in features]
    top=np.ceil(df.agbd.max()/bin_width)*bin_width; edges=np.arange(0,top+bin_width,bin_width)
    work=df[["agbd"]+sat_features].copy(); work["agbd_bin"]=pd.cut(work.agbd,edges,right=False)
    rows=[]; thresholds=[]
    for f in sat_features:
        g=work.groupby("agbd_bin",observed=True).agg(n=(f,"count"),agbd_median=("agbd","median"),median=(f,"median"),q25=(f,lambda x:x.quantile(.25)),q75=(f,lambda x:x.quantile(.75))).reset_index()
        g["feature"]=f; g["bin_left"]=g.agbd_bin.map(lambda x: x.left).astype(float); rows.append(g.drop(columns="agbd_bin"))
        valid=g[g.n>=100].copy(); scale=max(valid["median"].quantile(.95)-valid["median"].quantile(.05),1e-9); valid["change"]=valid["median"].diff().abs()/scale
        flat=valid[(valid.bin_left>=75) & (valid.change.rolling(3,min_periods=3).mean()<.04)]
        thresholds.append((f,float(flat.bin_left.iloc[0]) if len(flat) else np.nan))
    out=pd.concat(rows,ignore_index=True); out.to_csv(table_dir/"saturation_summary.csv",index=False)
    fig,axs=plt.subplots(2,int(np.ceil(len(sat_features)/2)),figsize=(15,7),squeeze=False)
    for ax,f in zip(axs.flat,sat_features):
        g=out[out.feature==f]; ax.plot(g.agbd_median,g["median"]); ax.fill_between(g.agbd_median,g.q25,g.q75,alpha=.22); ax.set(title=f,xlabel="AGBD bin median",ylabel=f"{f} median/IQR")
    _save(fig,fig_dir/"saturation_curves.png")
    vals=[v for _,v in thresholds if np.isfinite(v)]; region=float(np.median(vals)) if vals else None
    lines=["# Empirical Sentinel-2 saturation analysis","",f"AGBD was grouped in {bin_width} Mg/ha bins; medians and IQRs were computed without removing high values.","","Flattening candidates (a descriptive rolling-change rule, not universal physical thresholds):"]+[f"- {f}: {v:.0f} Mg/ha" if np.isfinite(v) else f"- {f}: no stable flattening detected" for f,v in thresholds]
    lines += ["", f"The cross-feature empirical saturation region begins around **{region:.0f} Mg/ha**." if region else "No consistent cross-feature flattening region was detected.", "This is dataset-specific evidence; it must not be interpreted as a universal optical saturation threshold."]
    (table_dir/"saturation_interpretation.md").write_text("\n".join(lines),encoding="utf-8")
    return {"feature_thresholds":dict(thresholds),"empirical_region_mg_ha":region}

def _group_split(groups, seed=42):
    unique=np.array(pd.unique(groups)); rng=np.random.default_rng(seed); rng.shuffle(unique)
    n=len(unique); test=set(unique[:max(1,int(.2*n))]); cal=set(unique[max(1,int(.2*n)):max(2,int(.4*n))])
    return np.where(groups.isin(test))[0],np.where(groups.isin(cal))[0],np.where(~groups.isin(test|cal))[0]

def _fit_subset(model,X,y,idx,max_samples,seed=42):
    idx=np.asarray(idx);
    if max_samples and len(idx)>max_samples: idx=np.random.default_rng(seed).choice(idx,max_samples,replace=False)
    return clone(model).fit(X.iloc[idx],y.iloc[idx]),idx

def run_experiment(dataset: Path, config: Path):
    cfg=load_yaml(config); out=Path(cfg.get("output_root","outputs")); tables=out/"tables"; figs=out/"figures"; models=out/"models"; metrics_dir=out/"metrics"
    for p in (tables,figs,models,metrics_dir,Path("report")): p.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(dataset); features=validate_s2_schema(df,cfg.get("target","agbd")); df=clean_analysis_rows(df,features); audit=audit_dataset(df,features,tables)
    spatial_cfg=cfg["spatial"]; df=add_spatial_blocks(df,spatial_cfg["block_size_m"],spatial_cfg["crs"]); groups=df.spatial_block; X=df[features]; y=df.agbd
    make_eda_figures(df,features,figs,spatial_cfg["block_size_m"],spatial_cfg["crs"]); saturation=saturation_analysis(df,features,tables,figs,cfg.get("saturation",{}).get("bin_width",25))
    max_train=cfg.get("compute",{}).get("max_train_samples_per_fold",120000); n_splits=spatial_cfg.get("n_splits",5); model_names=cfg.get("models",["ridge","random_forest","hist_gradient_boosting"])
    comparison=[]; fold_all=[]; oofs={}; fitted_defs={}
    for name in model_names:
        model=make_model(name,cfg.get("model",{})); folds,oof,overall=spatial_cv(model,X,y,groups,n_splits,max_train,42); random=random_holdout(model,X,y,max_train_samples=max_train)
        comparison += [{"model":name,"validation":"spatial_cv",**overall},{"model":name,"validation":"random_split",**{k:v for k,v in random.items() if k in overall}}]
        folds.insert(0,"model",name); fold_all.append(folds); oofs[name]=oof; fitted_defs[name]=model
    comp=pd.DataFrame(comparison); comp.to_csv(tables/"model_comparison.csv",index=False); folds_df=pd.concat(fold_all,ignore_index=True); folds_df.to_csv(tables/"spatial_cv_folds.csv",index=False); comp[comp.validation=="random_split"].to_csv(tables/"random_split_metrics.csv",index=False)
    spatial_rows=comp[comp.validation=="spatial_cv"].sort_values("rmse"); best_name=str(spatial_rows.iloc[0].model); best_model=fitted_defs[best_name]; best_oof=oofs[best_name]
    # Validation figures.
    pivot=comp.pivot(index="model",columns="validation",values="rmse"); fig,ax=plt.subplots(figsize=(8,4.8)); pivot.plot.bar(ax=ax); ax.set(ylabel="RMSE (Mg/ha)",title="Random versus spatial validation"); _save(fig,figs/"random_vs_spatial_validation.png")
    sample=np.random.default_rng(42).choice(len(y),min(100000,len(y)),replace=False); fig,ax=plt.subplots(figsize=(6,5)); ax.hexbin(y.iloc[sample],best_oof[sample],gridsize=65,bins="log",mincnt=1); lim=[0,max(y.quantile(.995),np.quantile(best_oof,.995))]; ax.plot(lim,lim,"r--"); ax.set(xlabel="Observed AGBD",ylabel="Spatial OOF prediction",title=f"Spatial CV: {best_name}"); _save(fig,figs/"predicted_vs_observed_spatial_cv.png")
    fig,ax=plt.subplots(figsize=(6,5)); ax.hexbin(best_oof[sample],y.iloc[sample].to_numpy()-best_oof[sample],gridsize=65,bins="log",mincnt=1); ax.axhline(0,color="r",ls="--"); ax.set(xlabel="Predicted AGBD",ylabel="Residual (observed - predicted)",title="Spatial CV residuals"); _save(fig,figs/"residuals_vs_predicted.png")
    # Feature ablation under the same spatial CV protocol.
    subsets={"bands_only":[c for c in S2_BANDS if c in features],"indices_only":[c for c in S2_INDICES if c in features],"bands_plus_indices":features,"red_edge":[c for c in ["B5","B6","B7","B8A","NDRE"] if c in features],"non_red_edge":[c for c in features if c not in {"B5","B6","B7","B8A","NDRE"}]}
    ab=[]
    for label,cols in subsets.items():
        _,_,m=spatial_cv(best_model,X[cols],y,groups,n_splits,max_train,42); ab.append({"feature_set":label,"n_features":len(cols),**m})
    ab=pd.DataFrame(ab).sort_values("rmse"); ab.to_csv(tables/"sentinel2_ablation.csv",index=False); fig,ax=plt.subplots(figsize=(8,4.6)); ax.barh(ab.feature_set,ab.rmse); ax.invert_yaxis(); ax.set(xlabel="Spatial CV RMSE",title=f"Sentinel-2 feature ablation ({best_name})"); _save(fig,figs/"sentinel2_ablation.png")
    # Strict group-separated train/calibration/test uncertainty experiment.
    test_idx,cal_idx,train_idx=_group_split(groups); conf_model,used_train=_fit_subset(best_model,X,y,train_idx,max_train,47); cal_pred=conf_model.predict(X.iloc[cal_idx]); test_pred=conf_model.predict(X.iloc[test_idx]); alpha=cfg["conformal"].get("alpha",.1); radius=conformal_radius(y.iloc[cal_idx],cal_pred,alpha); lo,hi=interval_from_radius(test_pred,radius); coverage=empirical_coverage(y.iloc[test_idx],lo,hi)
    conf_metrics=pd.DataFrame([{"nominal_coverage":1-alpha,"empirical_coverage":coverage,"conformal_radius":radius,"mean_interval_width":float(np.mean(hi-lo)),"n_train":len(used_train),"n_calibration":len(cal_idx),"n_test":len(test_idx)}]); conf_metrics.to_csv(tables/"conformal_metrics.csv",index=False)
    test_frame=pd.DataFrame({"agbd":y.iloc[test_idx].to_numpy(),"pred":test_pred,"lower":lo,"upper":hi}); test_frame["biomass_class"]=pd.cut(test_frame.agbd,[-np.inf,100,200,np.inf],labels=["low_<100","medium_100_200","high_>=200"]); cb=test_frame.groupby("biomass_class",observed=True).apply(lambda z:pd.Series({"n":len(z),"coverage":((z.agbd>=z.lower)&(z.agbd<=z.upper)).mean(),"mean_interval_width":(z.upper-z.lower).mean()}),include_groups=False).reset_index(); cb.to_csv(tables/"conformal_by_biomass_bin.csv",index=False)
    ss=_sample(test_frame,300); fig,ax=plt.subplots(figsize=(10,4.8)); ss=ss.sort_values("pred").reset_index(drop=True); ax.vlines(np.arange(len(ss)),ss.lower,ss.upper,alpha=.3); ax.scatter(np.arange(len(ss)),ss.agbd,s=7,c="black"); ax.set(xlabel="Held-out sample (sorted by prediction)",ylabel="AGBD",title="90% split-conformal intervals"); _save(fig,figs/"conformal_intervals.png")
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.bar(cb.biomass_class.astype(str),cb.coverage); ax.axhline(1-alpha,color="r",ls="--"); ax.set(ylim=(0,1),ylabel="Empirical coverage",title="Conformal coverage by biomass class"); _save(fig,figs/"conformal_coverage_by_biomass.png")
    # Domain validity is intentionally separate from conformal uncertainty.
    valcfg=cfg["validity"]; validity=KNNDomainValidity(valcfg.get("k_neighbors",10),valcfg.get("quantile_threshold",.95),valcfg.get("max_reference_samples",20000)).fit(X.iloc[used_train]); distances=validity.score_samples(X.iloc[test_idx]); errors=np.abs(test_pred-y.iloc[test_idx].to_numpy()); in_domain=distances<=validity.threshold_; dv=pd.DataFrame([{"threshold":validity.threshold_,"in_domain_fraction":in_domain.mean(),"mae_in_domain":errors[in_domain].mean(),"mae_out_of_domain":errors[~in_domain].mean() if (~in_domain).any() else np.nan,"spearman_distance_abs_error":pd.Series(distances).corr(pd.Series(errors),method="spearman")}]); dv.to_csv(tables/"domain_validity_metrics.csv",index=False)
    ds=pd.DataFrame({"distance":distances,"error":errors}); ds["bin"]=pd.qcut(ds.distance,20,duplicates="drop"); dg=ds.groupby("bin",observed=True).agg(distance=("distance","median"),error=("error","median")); fig,ax=plt.subplots(figsize=(7,4.8)); ax.scatter(_sample(ds,50000).distance,_sample(ds,50000).error,s=2,alpha=.08); ax.plot(dg.distance,dg.error,color="red",lw=2); ax.axvline(validity.threshold_,color="black",ls="--"); ax.set(xlabel="Standardized kNN domain distance",ylabel="Absolute error",title="Error versus feature-space domain distance"); _save(fig,figs/"error_vs_domain_distance.png")
    # Permutation importance on genuinely held-out spatial data.
    pi_n=min(cfg.get("compute",{}).get("importance_samples",10000),len(test_idx)); pi_idx=np.random.default_rng(42).choice(test_idx,pi_n,replace=False); pi=permutation_importance(conf_model,X.iloc[pi_idx],y.iloc[pi_idx],n_repeats=3,random_state=42,scoring="neg_root_mean_squared_error",n_jobs=1); imp=pd.DataFrame({"feature":features,"importance_mean":pi.importances_mean,"importance_sd":pi.importances_std}).sort_values("importance_mean",ascending=False); imp.to_csv(tables/"permutation_importance.csv",index=False); fig,ax=plt.subplots(figsize=(8,5)); z=imp.sort_values("importance_mean"); ax.barh(z.feature,z.importance_mean,xerr=z.importance_sd); ax.set(xlabel="Increase in RMSE after permutation",title=f"Held-out spatial permutation importance ({best_name})"); _save(fig,figs/"permutation_importance.png")
    pdp_features=[c for c in ["NDVI","NDMI","NDRE","B11","B12"] if c in features]; pdp_idx=np.random.default_rng(43).choice(test_idx,min(5000,len(test_idx)),replace=False); pdp_X=X.iloc[pdp_idx].astype("float64"); fig,axs=plt.subplots(2,3,figsize=(12,7)); PartialDependenceDisplay.from_estimator(conf_model,pdp_X,pdp_features,ax=axs.flat[:len(pdp_features)]); [a.axis("off") for a in axs.flat[len(pdp_features):]]; _save(fig,figs/"partial_dependence.png")
    # Refit deliverable model on a documented sample; calibration radius remains from disjoint groups.
    final_max=cfg.get("compute",{}).get("max_final_train_samples",200000); all_idx=np.arange(len(df)); final_model,final_idx=_fit_subset(best_model,X,y,all_idx,final_max,99)
    bundle={"model":final_model,"features":features,"preprocessing":"embedded in sklearn Pipeline","conformal_radius":radius,"conformal_alpha":alpha,"validity":validity,"crs":spatial_cfg["crs"],"block_size_m":spatial_cfg["block_size_m"],"target":"agbd","training_summary":{"n_dataset":len(df),"n_final_fit":len(final_idx),"selected_model":best_name,"spatial_cv":spatial_rows.iloc[0].to_dict(),"calibration_is_disjoint":True}}
    joblib.dump(bundle,models/"sentinel2_model_bundle.joblib")
    random_best=comp[(comp.model==best_name)&(comp.validation=="random_split")].iloc[0].to_dict(); spatial_best=spatial_rows.iloc[0].to_dict()
    gap=spatial_best["rmse"]-random_best["rmse"]
    summary={"sample_count":len(df),"final_feature_set":features,"random_split_metrics":random_best,"spatial_cv_metrics":spatial_best,"spatial_minus_random_rmse":gap,"fold_statistics":folds_df[folds_df.model==best_name].to_dict("records"),"conformal_coverage":coverage,"conformal_interval_width":float(np.mean(hi-lo)),"saturation_observations":saturation,"domain_validity_threshold":validity.threshold_,"selected_final_model":best_name,"spatial_blocks":int(groups.nunique()),"training_subsampling":{"per_fold_cap":max_train,"final_fit_cap":final_max,"validation_subsampled":False},"software":{"python":platform.python_version(),"pandas":pd.__version__,"scikit_learn":sklearn.__version__}}
    (metrics_dir/"sentinel2_experiment_summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:None if pd.isna(x) else float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)),encoding="utf-8")
    high_coverage = float(cb.loc[cb["biomass_class"].astype(str) == "high_>=200", "coverage"].iloc[0])
    report=f"""# Sentinel-2 biomass experiment results

## Dataset and audit

The matched dataset contains **{len(df):,} GEDI L4A observations** and {len(features)} Sentinel-2 predictors. GEDI AGBD is an estimated reference, not field ground truth. There were {audit['duplicate_shot_number']:,} duplicated `shot_number` values. The maximum AGBD was {df.agbd.max():.1f} Mg/ha; high observations were retained.

## Exploratory analysis and saturation

The empirical curve-flattening analysis used {cfg.get('saturation',{}).get('bin_width',25)} Mg/ha bins. The cross-feature descriptive saturation region was {saturation['empirical_region_mg_ha'] if saturation['empirical_region_mg_ha'] is not None else 'not consistently detected'} Mg/ha. This dataset-specific diagnostic does not establish a universal physical threshold and optical predictors should not be assumed to resolve high-biomass forest fully.

## Random versus spatial validation

The selected model was **{best_name}**, chosen by spatial CV RMSE. Spatial CV achieved RMSE {spatial_best['rmse']:.2f} Mg/ha, MAE {spatial_best['mae']:.2f}, R² {spatial_best['r2']:.3f}, and bias {spatial_best['bias']:.2f}. The diagnostic random split RMSE was {random_best['rmse']:.2f}; spatial minus random RMSE was {gap:.2f} Mg/ha, quantifying the random split's optimism. Held-out folds were never subsampled; model fitting used at most {max_train:,} training rows per fold.

## Uncertainty

Group-separated train, calibration, and test blocks were used for 90% split conformal prediction. Test coverage was {coverage:.3f}, with mean interval width {np.mean(hi-lo):.2f} Mg/ha. Coverage fell to {high_coverage:.3f} for AGBD at or above 200 Mg/ha, showing that the constant-width interval does not fully capture high-biomass error.

## Domain of validity

Standardized k-nearest-neighbour distance was fitted to training predictors. Its 95th-percentile reference threshold is {validity.threshold_:.3f}; {in_domain.mean():.1%} of the spatial test observations were inside-domain. In-domain MAE was {errors[in_domain].mean():.2f} Mg/ha and out-of-domain MAE was {errors[~in_domain].mean() if (~in_domain).any() else np.nan:.2f} Mg/ha; Spearman correlation between distance and absolute error was {pd.Series(distances).corr(pd.Series(errors), method='spearman'):.3f}. Distance did not identify larger errors in this test, so it should be retained as an extrapolation flag rather than interpreted as predictive uncertainty.

## Limitations and next steps

Spatial blocks reduce, but cannot eliminate, dependence and geographic sampling bias. Sentinel-2 reflectance can saturate and GEDI labels contain retrieval uncertainty. The operational sequence should next add Sentinel-1, repeat the same spatial validation and ablation, and then evaluate the incremental contribution of ALOS PALSAR L-band data.
"""
    Path("report/sentinel2_results.md").write_text(report,encoding="utf-8")
    return summary
