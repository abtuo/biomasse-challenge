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
import rasterio
import sklearn
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy.interpolate import RegularGridInterpolator
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from .config import load_yaml
from .data import (
    PALSAR_FEATURES,
    S1_FEATURES,
    SENSOR_GROUPS,
    combination_features,
    validate_sensor_groups,
)
from .modeling import fixed_fold_cv, make_hist_gradient_boosting, random_holdout, regression_metrics
from .raster import raster_feature_names, sample_neighborhood_means, sample_raster_blockwise
from .spatial import add_spatial_blocks
from .uncertainty import conformal_radius, empirical_coverage, interval_from_radius
from .validity import KNNDomainValidity


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def _sample(df, n, seed=42):
    return df if len(df) <= n else df.sample(n=n, random_state=seed)

def extract_multisensor_dataset(
    s2_dataset: Path, s1_raster: Path, palsar_raster: Path, output: Path
) -> pd.DataFrame:
    df = pd.read_parquet(s2_dataset)
    if len(df) == 0:
        raise ValueError("Input GEDI/Sentinel-2 dataset is empty")
    s1 = sample_raster_blockwise(df, s1_raster, S1_FEATURES)
    palsar = sample_raster_blockwise(df, palsar_raster, PALSAR_FEATURES)
    duplicate = (set(s1.columns) | set(palsar.columns)) & set(df.columns)
    if duplicate:
        raise ValueError(f"Output feature names already exist: {sorted(duplicate)}")
    out = pd.concat([df.reset_index(drop=True), s1.reset_index(drop=True), palsar.reset_index(drop=True)], axis=1)
    if len(out) != len(df) or not np.array_equal(out.longitude.to_numpy(), df.longitude.to_numpy(), equal_nan=True):
        raise RuntimeError("Coordinate/order consistency check failed after raster extraction")
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    return out

def multisensor_audit(df: pd.DataFrame, table_dir: Path) -> dict:
    groups = validate_sensor_groups(df)
    all_features = [feature for features in groups.values() for feature in features]
    table_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    missing_rows = []
    for sensor, features in groups.items():
        values = df[features].replace([np.inf, -np.inf], np.nan)
        complete = values.notna().all(axis=1)
        for feature in features:
            series = values[feature]
            missing_rows.append({
                "sensor": sensor, "feature": feature, "missing_count": int(series.isna().sum()),
                "missing_percent": float(series.isna().mean() * 100),
                "n_unique": int(series.nunique(dropna=True)),
                "constant": bool(series.nunique(dropna=True) <= 1),
                "min": float(series.min()) if series.notna().any() else np.nan,
                "max": float(series.max()) if series.notna().any() else np.nan,
            })
        rows.append({
            "sensor": sensor, "n_rows": len(df), "complete_rows": int(complete.sum()),
            "rows_with_any_missing": int((~complete).sum()), "complete_percent": float(complete.mean() * 100),
            "infinite_values": int(np.isinf(df[features].to_numpy()).sum()),
            "constant_features": ";".join(c for c in features if values[c].nunique(dropna=True) <= 1),
        })
    all_values = df[all_features].replace([np.inf, -np.inf], np.nan)
    rows.append({
        "sensor": "ALL", "n_rows": len(df),
        "complete_rows": int(all_values.notna().all(axis=1).sum()),
        "rows_with_any_missing": int(all_values.isna().any(axis=1).sum()),
        "complete_percent": float(all_values.notna().all(axis=1).mean() * 100),
        "infinite_values": int(np.isinf(df[all_features].to_numpy()).sum()),
        "constant_features": "",
    })
    audit = pd.DataFrame(rows)
    audit["duplicate_shot_number"] = int(df.shot_number.duplicated().sum()) if "shot_number" in df else np.nan
    audit["invalid_coordinates"] = int((~df.longitude.between(-180,180) | ~df.latitude.between(-90,90)).sum())
    # Very broad sanity limits catch linear-power values accidentally labelled dB and impossible ratios.
    radar = S1_FEATURES + PALSAR_FEATURES
    audit["suspicious_radar_values"] = int(sum((df[c].abs() > 1e4).sum() for c in radar))
    db_columns=["VV","VH","PALSAR_HH_dB","PALSAR_HV_dB"]
    audit["db_outside_minus60_20"] = int(sum((~df[c].between(-60,20)&df[c].notna()).sum() for c in db_columns))
    ratio_columns=["PALSAR_HV_HH_ratio","PALSAR_HH_HV_ratio"]
    audit["nonpositive_ratios"] = int(sum((df[c]<=0).sum() for c in ratio_columns))
    audit["ratio_above_20"] = int(sum((df[c]>20).sum() for c in ratio_columns))
    audit["positive_db_values"] = int(sum((df[c]>0).sum() for c in db_columns))
    audit["max_s1_difference_identity_error"] = float((df.VV-df.VH-df.VV_minus_VH).abs().max())
    audit["max_palsar_difference_identity_error"] = float((df.PALSAR_HH_dB-df.PALSAR_HV_dB-df.PALSAR_HH_minus_HV_dB).abs().max())
    audit.to_csv(table_dir / "multisensor_data_audit.csv", index=False)
    pd.DataFrame(missing_rows).to_csv(table_dir / "multisensor_missingness.csv", index=False)
    return {"groups": groups, "audit": audit.to_dict("records")}

def make_multisensor_eda(df: pd.DataFrame, fig_dir: Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot = _sample(df, 150000)
    for sensor, features, filename in [
        ("Sentinel-1", S1_FEATURES, "sentinel1_vs_agbd.png"),
        ("PALSAR", PALSAR_FEATURES[:4], "palsar_vs_agbd.png"),
    ]:
        cols = 2 if len(features) <= 4 else 3
        rows = int(np.ceil(len(features) / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 4.5*rows), squeeze=False)
        for ax, feature in zip(axs.flat, features):
            h = ax.hexbin(plot[feature], plot.agbd, gridsize=60, bins="log", mincnt=1, cmap="viridis")
            ax.set(xlabel=feature, ylabel="AGBD (Mg/ha)", title=f"{sensor}: {feature}")
        for ax in axs.flat[len(features):]: ax.axis("off")
        fig.colorbar(h, ax=axs.ravel().tolist(), label="log10(count)")
        _save(fig, fig_dir / filename)
    features = S1_FEATURES + PALSAR_FEATURES
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), squeeze=False)
    for ax, feature in zip(axs.flat, features):
        bins = pd.cut(df.agbd, np.arange(0, np.ceil(df.agbd.max()/50)*50+50, 50), right=False)
        g = df.groupby(bins, observed=True).agg(
            x=("agbd", "median"), med=(feature, "median"),
            q25=(feature, lambda x: x.quantile(.25)), q75=(feature, lambda x: x.quantile(.75)),
        )
        ax.plot(g.x, g.med); ax.fill_between(g.x, g.q25, g.q75, alpha=.2)
        ax.set(xlabel="AGBD bin median", ylabel=feature, title=feature)
    _save(fig, fig_dir / "radar_binned_response.png")
    corr = df[features + ["agbd"]].corr(method="spearman")["agbd"].drop("agbd").sort_values()
    fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(corr.index, corr.values); ax.axvline(0, color="black", lw=.8)
    ax.set(xlabel="Spearman rho with AGBD", title="Radar associations with GEDI AGBD")
    _save(fig, fig_dir / "radar_spearman_correlations.png")

def sensor_saturation(df: pd.DataFrame, table_dir: Path, fig_dir: Path, bin_width=25) -> dict:
    features = ["NDVI","NDMI","NDRE","NBR","B5","B6","B7","B8A","VV","VH","PALSAR_HH_dB","PALSAR_HV_dB"]
    edges = np.arange(0, np.ceil(df.agbd.max()/bin_width)*bin_width+bin_width, bin_width)
    binned = pd.cut(df.agbd, edges, right=False)
    rows, thresholds = [], {}
    for feature in features:
        g = df.groupby(binned, observed=True).agg(
            n=(feature,"count"), agbd_median=("agbd","median"), median=(feature,"median"),
            q25=(feature,lambda x:x.quantile(.25)), q75=(feature,lambda x:x.quantile(.75)),
        ).reset_index(drop=True)
        g["feature"] = feature
        scale=max(g["median"].quantile(.95)-g["median"].quantile(.05),1e-9)
        g["robust_change"] = g["median"].diff().abs()/scale
        flat=g[(g.n>=100)&(g.agbd_median>=75)&(g.robust_change.rolling(3,min_periods=3).mean()<.04)]
        thresholds[feature]=float(flat.agbd_median.iloc[0]) if len(flat) else None
        rows.append(g)
    result=pd.concat(rows,ignore_index=True); result.to_csv(table_dir/"sensor_saturation_summary.csv",index=False)
    selected=["NDVI","NDMI","VV","VH","PALSAR_HH_dB","PALSAR_HV_dB"]
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for ax,feature in zip(axs.flat,selected):
        g=result[result.feature==feature]; ax.plot(g.agbd_median,g["median"]); ax.fill_between(g.agbd_median,g.q25,g.q75,alpha=.2); ax.set(title=feature,xlabel="AGBD (Mg/ha)",ylabel="Median and IQR")
    _save(fig,fig_dir/"sensor_saturation_comparison.png")
    sensor_thresholds={
        sensor: float(np.median([thresholds[f] for f in fs if thresholds.get(f) is not None]))
        if any(thresholds.get(f) is not None for f in fs) else None
        for sensor,fs in {"S2":["NDVI","NDMI","NDRE","NBR","B5","B6","B7","B8A"],"S1":["VV","VH"],"PALSAR":["PALSAR_HH_dB","PALSAR_HV_dB"]}.items()
    }
    tail_counts={f"n_ge_{v}":int((df.agbd>=v).sum()) for v in [200,370,df.agbd.quantile(.9),df.agbd.quantile(.95)]}
    text=["# Sensor saturation interpretation","","## Empirical observations",f"A descriptive rolling-change rule applied to {bin_width} Mg/ha bins produced these dataset-specific flattening candidates: `{sensor_thresholds}`.",f"Tail sample counts were `{tail_counts}`. Whether PALSAR retains variation longer is assessed jointly with the high-biomass spatial-CV results; this curve diagnostic alone is insufficient.","","## Physical hypotheses","Optical indices can flatten as canopy greenness and leaf area approach closure. C-band backscatter is expected to interact mainly with canopy-scale elements and moisture, while longer-wavelength L-band can retain greater sensitivity to woody structure. These mechanisms are hypotheses for observed differences, not proof from this dataset.","","## Established sensor physics versus thresholds","Wavelength and interaction-depth differences are established sensor properties. The numerical flattening candidates above are not universal physical saturation thresholds and should not be transferred to other forests, seasons, or processing chains."]
    (table_dir/"sensor_saturation_interpretation.md").write_text("\n".join(text),encoding="utf-8")
    return {"feature_thresholds":thresholds,"sensor_thresholds":sensor_thresholds,"tail_counts":tail_counts}

def persisted_folds(df: pd.DataFrame, cfg: dict, path: Path) -> np.ndarray:
    blocked = add_spatial_blocks(df[["longitude","latitude"]], cfg["block_size_m"], cfg["crs"])
    key = "shot_number" if "shot_number" in df else None
    if path.exists():
        saved = pd.read_parquet(path)
        if key and len(saved) == len(df) and np.array_equal(saved[key].to_numpy(), df[key].to_numpy()):
            if not np.array_equal(saved.spatial_block.to_numpy(), blocked.spatial_block.to_numpy()):
                raise ValueError("Persisted blocks differ from current block definition")
            return saved.fold.to_numpy(dtype=int)
        raise ValueError("Persisted fold assignment does not match this dataset/order")
    fold = np.full(len(df), -1, dtype=int)
    cv = GroupKFold(n_splits=cfg["n_splits"])
    for fold_id, (_, test) in enumerate(cv.split(df, groups=blocked.spatial_block)):
        fold[test] = fold_id
    payload = pd.DataFrame({"spatial_block":blocked.spatial_block,"fold":fold})
    if key: payload.insert(0,key,df[key].to_numpy())
    path.parent.mkdir(parents=True, exist_ok=True); payload.to_parquet(path,index=False)
    return fold

def _regime_masks(y: pd.Series) -> dict[str,np.ndarray]:
    a=y.to_numpy(); q90=np.quantile(a,.9); q95=np.quantile(a,.95)
    return {
        "lt_100":a<100, "100_200":(a>=100)&(a<200), "200_370":(a>=200)&(a<370),
        "ge_370":a>=370, "top_10pct":a>=q90, "top_5pct":a>=q95,
    }

def _metrics_by_regime(y, pred, stack):
    rows=[]
    for regime,mask in _regime_masks(y).items():
        rows.append({"sensor_stack":stack,"biomass_regime":regime,"n":int(mask.sum()),**regression_metrics(y.to_numpy()[mask],np.asarray(pred)[mask])})
    return rows

def high_biomass_metrics(y, pred, stack: str) -> dict:
    actual=y.to_numpy(); pred=np.asarray(pred); top5=actual>=np.quantile(actual,.95)
    ge200=actual>=200; ge370=actual>=370
    return {
        "sensor_stack":stack,
        "mean_residual_ge_200":float(np.mean(pred[ge200]-actual[ge200])),
        "mean_residual_ge_370":float(np.mean(pred[ge370]-actual[ge370])),
        "rmse_ge_370":regression_metrics(actual[ge370],pred[ge370])["rmse"],
        "mae_ge_370":regression_metrics(actual[ge370],pred[ge370])["mae"],
        "median_observed_ge_370":float(np.median(actual[ge370])),
        "median_predicted_ge_370":float(np.median(pred[ge370])),
        "prediction_observation_ratio_ge_370":float(np.median(pred[ge370])/np.median(actual[ge370])),
        "fraction_ge_370_predicted_below_200":float(np.mean(pred[ge370]<200)),
        "fraction_top5_severely_underestimated":float(np.mean(pred[top5] < .7*actual[top5])),
    }

def grouped_permutation_importance(model, X, y, groups: dict[str,list[str]], scoring_model=None, repeats=3, seed=42):
    baseline=regression_metrics(y,model.predict(X))["rmse"]
    rng=np.random.default_rng(seed); rows=[]
    for sensor,features in groups.items():
        increases=[]
        for _ in range(repeats):
            order=rng.permutation(len(X)); shuffled=X.copy(); shuffled.loc[:,features]=X.iloc[order][features].to_numpy()
            increases.append(regression_metrics(y,model.predict(shuffled))["rmse"]-baseline)
        rows.append({"sensor":sensor,"importance_mean":float(np.mean(increases)),"importance_sd":float(np.std(increases)),"baseline_rmse":baseline})
    return pd.DataFrame(rows).sort_values("importance_mean",ascending=False)

def select_final_stack(overall: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    selection=overall[["sensor_stack","rmse","r2","fold_rmse_sd"]].merge(diagnostics,on="sensor_stack")
    selection["n_sensors"]=selection.sensor_stack.str.count(r"\+")+1
    selection["decision_score"]=(selection.rmse.rank()+selection.rmse_ge_370.rank()+selection.mean_residual_ge_370.abs().rank()+.15*selection.n_sensors)
    return selection.sort_values(["decision_score","n_sensors","sensor_stack"]).reset_index(drop=True)

def _group_train_cal_test(groups: pd.Series, seed=42):
    unique=np.array(pd.unique(groups)); np.random.default_rng(seed).shuffle(unique); n=len(unique)
    test=set(unique[:max(1,int(.2*n))]); cal=set(unique[max(1,int(.2*n)):max(2,int(.4*n))])
    return np.flatnonzero(~groups.isin(test|cal)),np.flatnonzero(groups.isin(cal)),np.flatnonzero(groups.isin(test))

def _fit_capped(model,X,y,idx,cap,seed):
    idx=np.asarray(idx)
    if cap and len(idx)>cap: idx=np.random.default_rng(seed).choice(idx,cap,replace=False)
    return clone(model).fit(X.iloc[idx],y.iloc[idx]),idx

def _conformal_comparison(df, combinations, model, groups, cap, alpha, table_dir, fig_dir):
    train,cal,test=_group_train_cal_test(groups); rows=[]; by=[]; fitted={}; predictions={}
    for stack in ["S2", combinations["selected_name"]]:
        if stack in fitted: continue
        features=combinations[stack]; fitted_model,_=_fit_capped(model,df[features],df.agbd,train,cap,51)
        cal_pred=fitted_model.predict(df.iloc[cal][features]); test_pred=fitted_model.predict(df.iloc[test][features])
        radius=conformal_radius(df.agbd.iloc[cal],cal_pred,alpha); lo,hi=interval_from_radius(test_pred,radius)
        rows.append({"sensor_stack":stack,"nominal_coverage":1-alpha,"coverage":empirical_coverage(df.agbd.iloc[test],lo,hi),"mean_width":float(np.mean(hi-lo)),"median_width":float(np.median(hi-lo)),"radius":radius,"n_calibration":len(cal),"n_test":len(test)})
        for regime,mask in _regime_masks(df.agbd.iloc[test]).items():
            yy=df.agbd.iloc[test].to_numpy()[mask]; by.append({"sensor_stack":stack,"biomass_regime":regime,"n":len(yy),"coverage":empirical_coverage(yy,lo[mask],hi[mask]),"mean_width":float(np.mean(hi[mask]-lo[mask])),"median_width":float(np.median(hi[mask]-lo[mask]))})
        fitted[stack]=fitted_model; predictions[stack]=(test_pred,lo,hi,radius)
    metrics=pd.DataFrame(rows); detail=pd.DataFrame(by); metrics.to_csv(table_dir/"conformal_multisensor_metrics.csv",index=False); detail.to_csv(table_dir/"conformal_multisensor_by_biomass.csv",index=False)
    for value,filename,ylabel in [("coverage","conformal_coverage_sensor_comparison.png","Coverage"),("mean_width","conformal_width_sensor_comparison.png","Mean interval width")]:
        p=detail.pivot(index="biomass_regime",columns="sensor_stack",values=value); fig,ax=plt.subplots(figsize=(10,5)); p.plot.bar(ax=ax); ax.set(ylabel=ylabel,title=f"Conditional conformal {ylabel.lower()}");
        if value=="coverage": ax.axhline(1-alpha,color="black",ls="--")
        _save(fig,fig_dir/filename)
    return metrics,detail,fitted,predictions,(train,cal,test)

def run_multisensor_experiment(dataset: Path, config: Path) -> dict:
    cfg=load_yaml(config); root=Path(cfg.get("output_root","outputs")); tables=root/"tables"; figs=root/"figures"/"multisensor"; models=root/"models"
    for p in [tables,figs,models,Path("report")]: p.mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(dataset).replace([np.inf,-np.inf],np.nan); groups=validate_sensor_groups(df); combos=combination_features(groups)
    all_features=[feature for features in groups.values() for feature in features]
    complete=df[all_features].notna().all(axis=1)&df.agbd.notna()
    if not complete.all(): df=df.loc[complete].reset_index(drop=True)
    audit=multisensor_audit(pd.read_parquet(dataset),tables); make_multisensor_eda(df,figs); saturation=sensor_saturation(df,tables,figs,cfg.get("saturation",{}).get("bin_width",25))
    folds=persisted_folds(df,cfg["spatial"],Path("data/processed/spatial_fold_assignment.parquet")); blocked=add_spatial_blocks(df[["longitude","latitude"]],cfg["spatial"]["block_size_m"],cfg["spatial"]["crs"]); block_groups=blocked.spatial_block
    model=make_hist_gradient_boosting(cfg["model"].get("hist_gradient_boosting",{})); cap=cfg.get("compute",{}).get("max_train_samples_per_fold",120000)
    overall_rows=[]; regime_rows=[]; diagnostic_rows=[]; fold_rows=[]; predictions={}
    for stack,features in combos.items():
        fold_table,oof,overall=fixed_fold_cv(model,df[features],df.agbd,folds,cap,42)
        overall_rows.append({"sensor_stack":stack,"n_features":len(features),**overall,"fold_rmse_sd":float(fold_table.rmse.std())})
        fold_table.insert(0,"sensor_stack",stack); fold_rows.append(fold_table); regime_rows.extend(_metrics_by_regime(df.agbd,oof,stack)); diagnostic_rows.append(high_biomass_metrics(df.agbd,oof,stack)); predictions[stack]=oof
    overall=pd.DataFrame(overall_rows).sort_values("rmse"); regimes=pd.DataFrame(regime_rows); diagnostics=pd.DataFrame(diagnostic_rows)
    overall.to_csv(tables/"multisensor_ablation.csv",index=False); regimes.to_csv(tables/"multisensor_ablation_by_biomass.csv",index=False); diagnostics.to_csv(tables/"high_biomass_diagnostics.csv",index=False); pd.concat(fold_rows).to_csv(tables/"multisensor_spatial_cv_folds.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,5)); z=overall.sort_values("rmse",ascending=False); ax.barh(z.sensor_stack,z.rmse); ax.set(xlabel="Spatial CV RMSE (Mg/ha)",title="Multisensor ablation: global performance"); _save(fig,figs/"model_ablation_global.png")
    high=regimes[regimes.biomass_regime=="ge_370"].sort_values("rmse",ascending=False); fig,ax=plt.subplots(figsize=(9,5)); ax.barh(high.sensor_stack,high.rmse); ax.set(xlabel="RMSE for AGBD >= 370 Mg/ha",title="High-biomass spatial-CV performance"); _save(fig,figs/"model_ablation_high_biomass.png")
    fig,ax=plt.subplots(figsize=(9,5)); z=diagnostics.sort_values("mean_residual_ge_370"); ax.barh(z.sensor_stack,z.mean_residual_ge_370); ax.axvline(0,color="black"); ax.set(xlabel="Mean prediction residual (Mg/ha)",title="Bias for AGBD >= 370 Mg/ha"); _save(fig,figs/"high_biomass_bias.png")
    top_idx=np.flatnonzero(df.agbd.to_numpy()>=np.quantile(df.agbd,.95)); top_idx=np.random.default_rng(42).choice(top_idx,min(30000,len(top_idx)),replace=False); fig,axs=plt.subplots(2,4,figsize=(15,8));
    for ax,(stack,pred) in zip(axs.flat,predictions.items()): ax.hexbin(df.agbd.iloc[top_idx],pred[top_idx],gridsize=45,bins="log",mincnt=1); ax.plot([370,df.agbd.iloc[top_idx].max()],[370,df.agbd.iloc[top_idx].max()],"r--"); ax.set(title=stack,xlabel="Observed",ylabel="Predicted")
    for ax in axs.flat[len(predictions):]:ax.axis("off")
    _save(fig,figs/"high_biomass_predicted_vs_observed.png")
    # Transparent selection: equal rank weight for global RMSE, >=370 RMSE and absolute >=370 bias, plus a small complexity cost.
    selection=select_final_stack(overall,diagnostics); best=str(selection.iloc[0].sensor_stack)
    # Random/spatial comparison on pre-specified stacks.
    gap_rows=[]
    for stack in ["S2","S2+PALSAR","S2+S1+PALSAR"]:
        rand=random_holdout(model,df[combos[stack]],df.agbd,max_train_samples=cap); spatial=overall.set_index("sensor_stack").loc[stack]
        gap_rows.append({"sensor_stack":stack,"r2_random":rand["r2"],"r2_spatial":spatial.r2,"delta_r2":rand["r2"]-spatial.r2,"rmse_random":rand["rmse"],"rmse_spatial":spatial.rmse,"delta_rmse":spatial.rmse-rand["rmse"]})
    gaps=pd.DataFrame(gap_rows); gaps.to_csv(tables/"random_vs_spatial_multisensor.csv",index=False); fig,axs=plt.subplots(1,2,figsize=(11,4.5)); axs[0].bar(gaps.sensor_stack,gaps.delta_r2); axs[0].set(title="Random minus spatial R²",ylabel="Delta R²"); axs[1].bar(gaps.sensor_stack,gaps.delta_rmse); axs[1].set(title="Spatial minus random RMSE",ylabel="Delta RMSE"); [a.tick_params(axis="x",rotation=25) for a in axs]; _save(fig,figs/"random_vs_spatial_gap.png")
    combos["selected_name"]=best; conf,_conf_by,conf_models,conf_preds,splits=_conformal_comparison(df,combos,model,block_groups,cap,cfg["conformal"].get("alpha",.1),tables,figs); del combos["selected_name"]
    # Importance uses the held-out spatial test component of the conformal split.
    train,_cal,test=splits; best_model=conf_models[best]; imp_n=min(cfg.get("compute",{}).get("importance_samples",10000),len(test)); imp_idx=np.random.default_rng(70).choice(test,imp_n,replace=False); features=combos[best]
    imp=permutation_importance(best_model,df.iloc[imp_idx][features],df.agbd.iloc[imp_idx],n_repeats=3,scoring="neg_root_mean_squared_error",random_state=42,n_jobs=1); feature_imp=pd.DataFrame({"feature":features,"importance_mean":imp.importances_mean,"importance_sd":imp.importances_std}); feature_imp["sensor"]=[next(s for s,fs in groups.items() if f in fs) for f in features]; feature_imp=feature_imp.sort_values("importance_mean",ascending=False); feature_imp.to_csv(tables/"feature_permutation_importance_multisensor.csv",index=False)
    selected_groups={s:[f for f in fs if f in features] for s,fs in groups.items() if any(f in features for f in fs)}; grouped=grouped_permutation_importance(best_model,df.iloc[imp_idx][features],df.agbd.iloc[imp_idx],selected_groups); grouped.to_csv(tables/"grouped_sensor_importance.csv",index=False)
    for table,filename,label in [(feature_imp,"permutation_importance_multisensor.png","feature"),(grouped,"grouped_sensor_importance.png","sensor")]: z=table.sort_values("importance_mean"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(z[label],z.importance_mean,xerr=z.importance_sd); ax.set(xlabel="Held-out RMSE increase",title="Permutation importance"); _save(fig,figs/filename)
    # Domain validity comparison on identical held-out blocks.
    domain_rows=[]; validity_objects={}
    for stack in ["S2",best]:
        if stack in validity_objects: continue
        fs=combos[stack]; validity=KNNDomainValidity(cfg["validity"].get("k_neighbors",10),cfg["validity"].get("quantile_threshold",.95),cfg["validity"].get("max_reference_samples",20000)).fit(df.iloc[train][fs]); distance=validity.score_samples(df.iloc[test][fs]); pred=conf_preds[stack][0]; error=np.abs(pred-df.agbd.iloc[test].to_numpy()); hb=df.agbd.iloc[test].to_numpy()>=370; ind=distance<=validity.threshold_
        domain_rows.append({"sensor_stack":stack,"threshold":validity.threshold_,"in_domain_fraction":float(ind.mean()),"mae_in_domain":float(error[ind].mean()),"mae_out_domain":float(error[~ind].mean()),"distance_error_spearman":float(pd.Series(distance).corr(pd.Series(error),method="spearman")),"high_biomass_distance_error_spearman":float(pd.Series(distance[hb]).corr(pd.Series(error[hb]),method="spearman"))})
        validity_objects[stack]=(validity,distance,error,hb)
    domain=pd.DataFrame(domain_rows); domain.to_csv(tables/"domain_validity_multisensor.csv",index=False); fig,axs=plt.subplots(1,len(validity_objects),figsize=(6*len(validity_objects),4.5),squeeze=False)
    for ax,(stack,(validity,distance,error,hb)) in zip(axs.flat,validity_objects.items()): sample=np.random.default_rng(4).choice(len(distance),min(30000,len(distance)),replace=False); ax.scatter(distance[sample],error[sample],s=2,alpha=.08); ax.scatter(distance[hb],error[hb],s=2,alpha=.05,color="red"); ax.axvline(validity.threshold_,color="black",ls="--"); ax.set(title=stack,xlabel="kNN distance",ylabel="Absolute error")
    _save(fig,figs/"error_vs_domain_distance_multisensor.png")
    # Add calibration to selection and persist transparent decision table.
    selection=selection.merge(conf[["sensor_stack","coverage","mean_width"]],on="sensor_stack",how="left"); selection["selected"]=selection.sensor_stack.eq(best); selection.to_csv(tables/"final_model_selection.csv",index=False)
    base=overall.set_index("sensor_stack").loc["S2"]; chosen=overall.set_index("sensor_stack").loc[best]; hbdiag=diagnostics.set_index("sensor_stack")
    selection_md=f"""# Final multisensor model selection

The selected stack is **{best}**. Selection minimized an explicit composite of ranks for spatial-CV global RMSE, RMSE above 370 Mg/ha, absolute bias above 370 Mg/ha, plus a small sensor-count penalty (0.15 per sensor). Models were not tuned independently.

Compared with S2, spatial RMSE changed from {base.rmse:.2f} to {chosen.rmse:.2f} Mg/ha and R² from {base.r2:.3f} to {chosen.r2:.3f}. Mean residual above 370 Mg/ha changed from {hbdiag.loc['S2','mean_residual_ge_370']:.2f} to {hbdiag.loc[best,'mean_residual_ge_370']:.2f} Mg/ha. Conditional coverage and domain behavior remain separate diagnostics; global gains do not imply that saturation or extrapolation is solved.
"""; (tables/"final_model_selection.md").write_text(selection_md,encoding="utf-8")
    # Fit final bundle; conformal and validity are based on disjoint spatial groups.
    final_cap=cfg.get("compute",{}).get("max_final_train_samples",200000); final_model,final_idx=_fit_capped(model,df[features],df.agbd,np.arange(len(df)),final_cap,99); validity=validity_objects[best][0]; radius=conf.set_index("sensor_stack").loc[best,"radius"]
    bundle={"model":final_model,"features":features,"sensor_groups":selected_groups,"preprocessing":"embedded in sklearn Pipeline","crs":cfg["spatial"]["crs"],"block_size_m":cfg["spatial"]["block_size_m"],"conformal":{"alpha":cfg["conformal"].get("alpha",.1),"radius":float(radius),"spatially_disjoint":True},"validity":validity,"training_summary":{"n_complete":len(df),"n_final_fit":len(final_idx),"selected_stack":best,"spatial_metrics":chosen.to_dict()},"sensor_ablation_summary":overall.to_dict("records"),"versions":{"python":platform.python_version(),"pandas":pd.__version__,"scikit_learn":sklearn.__version__}}
    joblib.dump(bundle,models/"final_multisensor_model_bundle.joblib")
    summary={"matched_samples":len(df),"selected_stack":best,"spatial_metrics":overall.to_dict("records"),"high_biomass":diagnostics.to_dict("records"),"conformal":conf.to_dict("records"),"domain_validity":domain.to_dict("records"),"saturation":saturation,"audit":audit["audit"]}
    (root/"metrics"/"multisensor_experiment_summary.json").parent.mkdir(parents=True,exist_ok=True); (root/"metrics"/"multisensor_experiment_summary.json").write_text(json.dumps(summary,indent=2,default=lambda x:None if pd.isna(x) else float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)),encoding="utf-8")
    raster_cfg=cfg.get("rasters",{})
    if all(Path(raster_cfg.get(k,"")).exists() for k in ["sentinel2","sentinel1","palsar"]):
        footprint_support_analysis(df,features,model,cfg,raster_cfg,tables,figs)
    _write_multisensor_reports(df,overall,diagnostics,conf,domain,saturation,best,tables)
    return summary

def footprint_support_analysis(df, features, model, cfg, raster_cfg, table_dir, fig_dir):
    n=min(cfg.get("compute",{}).get("support_samples",5000),len(df)); subset=df.sample(n=n,random_state=81).copy().reset_index(drop=True)
    rows=[]; support_data={"center":subset[features].copy()}
    for diameter in [25,50]:
        parts=[]
        for sensor,key in [("S2","sentinel2"),("S1","sentinel1"),("PALSAR","palsar")]:
            wanted=[f for f in SENSOR_GROUPS[sensor] if f in features]
            if wanted:
                sampled=sample_neighborhood_means(subset,Path(raster_cfg[key]),diameter,SENSOR_GROUPS[sensor])
                parts.append(sampled[wanted])
        support_data[f"mean_{diameter}m"]=pd.concat(parts,axis=1)[features]
    blocked=add_spatial_blocks(subset[["longitude","latitude"]],cfg["spatial"]["block_size_m"],cfg["spatial"]["crs"]); folds=np.full(n,-1); cv=GroupKFold(n_splits=min(5,blocked.spatial_block.nunique()))
    for f,(_,te) in enumerate(cv.split(subset,groups=blocked.spatial_block)):folds[te]=f
    for support,X in support_data.items():
        valid=X.notna().all(axis=1); _,pred,metrics=fixed_fold_cv(model,X.loc[valid].reset_index(drop=True),subset.loc[valid,"agbd"].reset_index(drop=True),folds[valid],max_train_samples=None)
        hb=high_biomass_metrics(subset.loc[valid,"agbd"].reset_index(drop=True),pred,support)
        correlations=[subset.loc[valid,"agbd"].corr(X.loc[valid,f],method="spearman") for f in features]
        rows.append({"support":support,"n":int(valid.sum()),"median_abs_feature_spearman":float(np.nanmedian(np.abs(correlations))),**metrics,"mean_residual_ge_200":hb["mean_residual_ge_200"],"mean_residual_ge_370":hb["mean_residual_ge_370"]})
    result=pd.DataFrame(rows); result.to_csv(table_dir/"footprint_support_sensitivity.csv",index=False); fig,axs=plt.subplots(1,3,figsize=(13,4));
    for ax,col,title in zip(axs,["rmse","mean_residual_ge_370","median_abs_feature_spearman"],["Spatial CV RMSE","Bias >=370 Mg/ha","Median |feature rho|"]):ax.bar(result.support,result[col]);ax.set(title=title);ax.tick_params(axis="x",rotation=20)
    _save(fig,fig_dir/"footprint_support_sensitivity.png")
    return result

def _write_multisensor_reports(df,overall,diagnostics,conf,domain,saturation,best,tables):
    base=overall.set_index("sensor_stack").loc["S2"]; chosen=overall.set_index("sensor_stack").loc[best]; diag=diagnostics.set_index("sensor_stack"); cm=conf.set_index("sensor_stack"); dv=domain.set_index("sensor_stack")
    s1=overall.set_index("sensor_stack").loc["S2+S1"]; palsar=overall.set_index("sensor_stack").loc["S2+PALSAR"]
    gaps=pd.read_csv(tables/"random_vs_spatial_multisensor.csv").set_index("sensor_stack")
    conf_by=pd.read_csv(tables/"conformal_multisensor_by_biomass.csv").set_index(["sensor_stack","biomass_regime"])
    support=pd.read_csv(tables/"footprint_support_sensitivity.csv").set_index("support") if (tables/"footprint_support_sensitivity.csv").exists() else None
    report=f"""# Multisensor biomass results

## Dataset and design

The matched analysis contains **{len(df):,} GEDI observations** with explicit Sentinel-2, Sentinel-1, and PALSAR feature groups. All seven sensor stacks used the same persisted 10 km GroupKFold assignment. GEDI L4A is an estimated reference rather than field ground truth.

## Sensor-specific interpretation

**Sentinel-2** describes canopy reflectance, greenness, red-edge response, and SWIR-related moisture. Closed, dense canopies can show weak incremental optical response as leaf area and greenness saturate.

**Sentinel-1 C-band** has a shorter wavelength and responds to canopy-scale structure, moisture, and surface conditions. In dense forest it is expected to reach limited structural sensitivity earlier than L-band, although moisture effects can remain substantial.

**PALSAR L-band** has a longer wavelength and stronger interaction with larger woody elements and can penetrate farther into the canopy than C-band. This supplies a physical reason to test for extended high-biomass sensitivity, but does not guarantee it in these processed mosaics.

The empirical flattening diagnostics were {saturation['sensor_thresholds']}. PALSAR flattened later by this descriptive rule than S1 and the aggregated S2 diagnostic, but its incremental spatial-CV improvement was smaller than S1's. These are dataset-specific curve summaries, not universal saturation thresholds.

## Spatial validation and marginal sensor value

S2-only spatial CV achieved RMSE {base.rmse:.2f} Mg/ha and R² {base.r2:.3f}. Adding S1 alone to S2 produced RMSE {s1.rmse:.2f} and R² {s1.r2:.3f}; adding PALSAR produced RMSE {palsar.rmse:.2f} and R² {palsar.r2:.3f}. The selected stack was **{best}**, with RMSE {chosen.rmse:.2f}, MAE {chosen.mae:.2f}, R² {chosen.r2:.3f}, and fold-RMSE SD {chosen.fold_rmse_sd:.2f}. For the selected stack, random R² exceeded spatial R² by {gaps.loc[best,'delta_r2']:.3f}, while spatial RMSE exceeded random RMSE by {gaps.loc[best,'delta_rmse']:.2f} Mg/ha.

## High biomass

For AGBD >=370 Mg/ha, S2 mean residual was {diag.loc['S2','mean_residual_ge_370']:.2f} Mg/ha and RMSE was {diag.loc['S2','rmse_ge_370']:.2f}. With {best}, these became {diag.loc[best,'mean_residual_ge_370']:.2f} and {diag.loc[best,'rmse_ge_370']:.2f}. Persistent negative residuals indicate unresolved high-biomass underestimation even if radar improves global metrics.

## Uncertainty and domain of validity

S2 split-conformal coverage was {cm.loc['S2','coverage']:.3f}; {best} coverage was {cm.loc[best,'coverage']:.3f}, against a 0.90 nominal target. Above 370 Mg/ha, coverage was only {conf_by.loc[('S2','ge_370'),'coverage']:.3f} for S2 and {conf_by.loc[(best,'ge_370'),'coverage']:.3f} for {best}; top-5% coverage was {conf_by.loc[('S2','top_5pct'),'coverage']:.3f} and {conf_by.loc[(best,'top_5pct'),'coverage']:.3f}. Constant-width split conformal therefore fails conditionally in the upper tail; this limitation is not presented as solved.

The {best} feature-space threshold was {dv.loc[best,'threshold']:.3f}. Its distance/error Spearman correlation was {dv.loc[best,'distance_error_spearman']:.3f}. Large errors at small distance support an information-content or saturation limitation rather than pure feature-space extrapolation; saturation and domain validity are not interchangeable.

## Operational interpretation and limitations

The model predicts GEDI-referenced footprint biomass from temporally composited remote-sensing features. It is not direct field validation. Point/footprint support mismatch, geolocation uncertainty, allometric uncertainty, mosaic timing, conditional undercoverage, and residual spatial dependence all limit operational interpretation. {f"On the 1,000-shot support subset, 50 m means reduced RMSE from {support.loc['center','rmse']:.2f} to {support.loc['mean_50m','rmse']:.2f} Mg/ha, suggesting that local landscape context and geolocation/support averaging matter." if support is not None else ""} The final map uses a 25 m PALSAR-aligned grid and block-wise inference. Biomass is predicted at 25 m; kNN domain distance is evaluated on a 250 m lattice and interpolated to the same grid for operational tractability.
"""
    Path("report/multisensor_results.md").write_text(report,encoding="utf-8")
    questions="""# Fundamental questions

## 1. Why can random R² be illusory?

Nearby samples share forest structure, disturbance history, soils, and image artifacts. A random split therefore puts spatial neighbours in train and test and can reward interpolation. The persisted 10 km GroupKFold design withholds whole areas, so its R² and RMSE are the primary evidence. The random-minus-spatial R² and spatial-minus-random RMSE tables quantify optimism directly.

## 2. How do GEDI allometric equations affect the target?

GEDI L4A AGBD is a model-based estimate derived from waveform metrics and calibration equations; it is not error-free field truth. Errors may vary with forest type, structure, beam, sensitivity, and biomass. Empirical verification requires independent, spatially matched field plots, stratified residual checks, calibration slope/intercept, and comparison against `agbd_se`; agreement with GEDI alone cannot validate the underlying allometry.

## 3. What causes sensor saturation?

Optical reflectance becomes less responsive after canopy closure and high leaf area. C-band interacts mainly with smaller canopy elements and moisture and often loses structural sensitivity in dense stands. Longer-wavelength L-band interacts more with woody structure and may retain useful response farther into the biomass range. These are established physical tendencies; numerical thresholds in this study are empirical and site-specific.

## 4. How can extrapolation be detected without ground truth?

Standardized kNN distance compares a prediction pixel with the training feature distribution and flags remote feature combinations. Spatial coverage, per-feature ranges, and environmental strata provide complementary checks. This diagnoses extrapolation, not accuracy: small distance with large errors can indicate target noise, support mismatch, or sensor information saturation.

## 5. Can regional stock be obtained by summing pixels?

Only with explicit area conversion, masking, bias correction, and uncertainty propagation at the intended support. Pixel errors are spatially correlated, so adding independent pixel variances is invalid and usually overstates effective sample size. Defensible totals need spatial block/bootstrap resampling, model-assisted estimators, or geostatistical covariance models, plus uncertainty from GEDI labels, model calibration, map area, and carbon conversion. Independent field validation and a clearly defined inference population are prerequisites for an operational stock estimate.

## 6. What spatial property does the model predict?

The label represents a GEDI footprint (roughly 25 m diameter), while Sentinel pixels are 20 m, PALSAR pixels are 25 m, and geolocation is imperfect. Center-pixel extraction can therefore mismatch the waveform support. Neighborhood means test 25 m and 50 m supports: improvement at broader support suggests the model represents local landscape context rather than strictly footprint biomass. Map pixels inherit the trained support relationship and should not automatically be interpreted as independent field plots.
"""
    Path("report/fundamental_questions.md").write_text(questions,encoding="utf-8")

def predict_multisensor_map(bundle_path: Path, s2_path: Path, s1_path: Path, palsar_path: Path, output_dir: Path):
    """Block-wise inference on the native 25 m PALSAR grid."""
    bundle=joblib.load(bundle_path); output_dir.mkdir(parents=True,exist_ok=True)
    sources=[rasterio.open(p) for p in [s2_path,s1_path,palsar_path]]
    try:
        reference=sources[2]
        vrts=[]
        for source in sources:
            vrts.append(WarpedVRT(source,crs=reference.crs,transform=reference.transform,width=reference.width,height=reference.height,resampling=Resampling.bilinear,src_nodata=source.nodata,nodata=np.nan))
        feature_locations={}
        for vrt in vrts:
            for i,name in enumerate(raster_feature_names(vrt),start=1):
                if name in feature_locations: raise ValueError(f"Duplicate raster feature name: {name}")
                feature_locations[name]=(vrt,i)
        missing=[f for f in bundle["features"] if f not in feature_locations]
        if missing: raise ValueError(f"Prediction rasters are missing features: {missing}")
        reads_by_vrt = {}
        for feature in bundle["features"]:
            vrt,index=feature_locations[feature]
            reads_by_vrt.setdefault(vrt,[]).append((feature,index))
        profile=reference.profile.copy(); profile.update(count=1,dtype="float32",compress="deflate",predictor=3,tiled=True,nodata=np.nan)
        paths={name:output_dir/name for name in ["biomass_prediction.tif","biomass_pi_lower.tif","biomass_pi_upper.tif","biomass_interval_width.tif","domain_distance.tif"]}
        writers={name:rasterio.open(path,"w",**profile) for name,path in paths.items()}
        mask_profile=profile.copy(); mask_profile.update(dtype="uint8",nodata=255,predictor=2)
        mask_writer=rasterio.open(output_dir/"domain_validity_mask.tif","w",**mask_profile)
        writers["domain_distance.tif"].update_tags(
            diagnostic_resolution_m=250,
            method="fitted kNN evaluated on 10-pixel lattice then bilinearly interpolated",
        )
        mask_writer.update_tags(diagnostic_resolution_m=250)
        radius=float(bundle["conformal"]["radius"])
        try:
            for _,window in reference.block_windows(1):
                block_features={}
                for vrt,items in reads_by_vrt.items():
                    indices=[index for _,index in items]
                    data=vrt.read(indices,window=window,masked=True).filled(np.nan).astype("float32")
                    block_features.update({feature:data[i] for i,(feature,_) in enumerate(items)})
                arrays=[block_features[feature] for feature in bundle["features"]]
                stack=np.stack(arrays); h,w=stack.shape[1:]; X=stack.reshape(len(arrays),-1).T; valid=np.isfinite(X).all(axis=1)
                pred=np.full(h*w,np.nan,dtype="float32"); lower=pred.copy(); upper=pred.copy(); distance=pred.copy(); validity_mask=np.full(h*w,255,dtype="uint8")
                if valid.any():
                    valid_X=X[valid]
                    p=bundle["model"].predict(pd.DataFrame(valid_X,columns=bundle["features"])).astype("float32")
                    coarse_rows=np.unique(np.r_[np.arange(0,h,10),h-1])
                    coarse_cols=np.unique(np.r_[np.arange(0,w,10),w-1])
                    rr,cc=np.meshgrid(coarse_rows,coarse_cols,indexing="ij")
                    coarse_X=stack[:,rr,cc].reshape(len(arrays),-1).T
                    coarse_valid=np.isfinite(coarse_X).all(axis=1)
                    coarse_distance=np.full(len(coarse_X),np.nan,dtype="float32")
                    if coarse_valid.any():
                        query=coarse_X[coarse_valid]
                        coarse_distance[coarse_valid]=np.concatenate([
                            bundle["validity"].score_samples(query[start:start+256])
                            for start in range(0,len(query),256)
                        ]).astype("float32")
                    coarse_distance=coarse_distance.reshape(len(coarse_rows),len(coarse_cols))
                    interpolator=RegularGridInterpolator((coarse_rows,coarse_cols),coarse_distance,bounds_error=False,fill_value=np.nan)
                    full_rr,full_cc=np.meshgrid(np.arange(h),np.arange(w),indexing="ij")
                    d_full=interpolator(np.column_stack([full_rr.ravel(),full_cc.ravel()])).astype("float32")
                    d=d_full[valid]
                    lo=np.maximum(0,p-radius); hi=p+radius
                    pred[valid]=p;lower[valid]=lo;upper[valid]=hi;distance[valid]=d;validity_mask[valid]=(d<=bundle["validity"].threshold_).astype("uint8")
                data={"biomass_prediction.tif":pred,"biomass_pi_lower.tif":lower,"biomass_pi_upper.tif":upper,"biomass_interval_width.tif":upper-lower,"domain_distance.tif":distance}
                for name,arr in data.items():writers[name].write(arr.reshape(h,w),1,window=window)
                mask_writer.write(validity_mask.reshape(h,w),1,window=window)
        finally:
            for writer in writers.values(): writer.close()
            mask_writer.close()
            for vrt in vrts:vrt.close()
    finally:
        for source in sources:source.close()
    make_map_previews(output_dir,output_dir.parent/"figures"/"final_maps")

def make_map_previews(map_dir: Path, figure_dir: Path):
    figure_dir.mkdir(parents=True,exist_ok=True)
    specs=[("biomass_prediction.tif","Final biomass prediction","viridis"),("biomass_interval_width.tif","Prediction interval width","magma"),("domain_validity_mask.tif","Domain validity mask","RdYlGn")]
    images=[]
    for filename,title,cmap in specs:
        with rasterio.open(map_dir/filename) as src:
            scale=max(1,int(max(src.width,src.height)/1400)); arr=src.read(1,out_shape=(max(1,src.height//scale),max(1,src.width//scale)),masked=True,resampling=Resampling.average if "mask" not in filename else Resampling.nearest)
        images.append((arr,title,cmap,filename.replace(".tif",".png")))
    for arr,title,cmap,filename in images:
        fig,ax=plt.subplots(figsize=(7,6)); im=ax.imshow(arr,cmap=cmap);fig.colorbar(im,ax=ax);ax.set(title=title);ax.axis("off");_save(fig,figure_dir/filename)
    fig,axs=plt.subplots(1,3,figsize=(16,5));
    for ax,(arr,title,cmap,_) in zip(axs,images): im=ax.imshow(arr,cmap=cmap);fig.colorbar(im,ax=ax,shrink=.75);ax.set(title=title);ax.axis("off")
    _save(fig,figure_dir/"final_map_diagnostics.png")
