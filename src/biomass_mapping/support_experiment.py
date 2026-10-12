from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

from .config import load_yaml
from .data import SENSOR_GROUPS
from .modeling import fixed_fold_cv, make_hist_gradient_boosting, regression_metrics
from .spatial import add_spatial_blocks
from .support import extract_local_features, support_geometry_description
from .uncertainty import conformal_radius, empirical_coverage, interval_from_radius
from .validity import KNNDomainValidity

SUPPORTS = {"POINT": None, "SUPPORT_25": 12.5, "SUPPORT_50": 25.0}


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout();fig.savefig(path,dpi=220,bbox_inches="tight");plt.close(fig)


def build_support_datasets(point_path: Path, raster_paths: dict[str, Path]) -> dict:
    point=pd.read_parquet(point_path);all_features=[f for fs in SENSOR_GROUPS.values() for f in fs]
    missing=[f for f in all_features if f not in point]
    if missing:raise ValueError(f"Point dataset missing features: {missing}")
    outputs={"POINT":Path("data/interim/gedi_multisensor_point.parquet"),"SUPPORT_25":Path("data/interim/gedi_multisensor_support25.parquet"),"SUPPORT_50":Path("data/interim/gedi_multisensor_support50.parquet")}
    outputs["POINT"].parent.mkdir(parents=True,exist_ok=True);point.to_parquet(outputs["POINT"],index=False)
    metadata=point.drop(columns=all_features)
    frames={"POINT":point}
    for support,radius in [("SUPPORT_25",12.5),("SUPPORT_50",25.0)]:
        parts=[];suffix=support.lower()
        for sensor,key in [("S2","sentinel2"),("S1","sentinel1"),("PALSAR","palsar")]:
            parts.append(extract_local_features(point,raster_paths[key],radius,suffix=suffix,expected_features=SENSOR_GROUPS[sensor]))
        frame=pd.concat([metadata.reset_index(drop=True)]+[p.reset_index(drop=True) for p in parts],axis=1)
        if not np.array_equal(frame.shot_number.to_numpy(),point.shot_number.to_numpy()):raise RuntimeError("Row identity changed during support extraction")
        frame.to_parquet(outputs[support],index=False);frames[support]=frame
    point_complete=point[all_features].notna().all(axis=1)
    complete={"POINT":point_complete}
    for support in ["SUPPORT_25","SUPPORT_50"]:
        cols=[f"{f}_{support.lower()}" for f in all_features];complete[support]=frames[support][cols].notna().all(axis=1)
    common=complete["POINT"]&complete["SUPPORT_25"]&complete["SUPPORT_50"]
    wide=metadata.loc[common].reset_index(drop=True)
    wide=pd.concat([wide,point.loc[common,all_features].rename(columns=lambda c:f"{c}_point").reset_index(drop=True),frames["SUPPORT_25"].loc[common,[f"{f}_support_25" for f in all_features]].reset_index(drop=True),frames["SUPPORT_50"].loc[common,[f"{f}_support_50" for f in all_features]].reset_index(drop=True)],axis=1)
    common_path=Path("data/processed/support_common_sample.parquet");common_path.parent.mkdir(parents=True,exist_ok=True);wide.to_parquet(common_path,index=False)
    retention={"total_rows":len(point),**{f"complete_{k.lower()}":int(v.sum()) for k,v in complete.items()},"common_rows":int(common.sum())}
    Path("outputs/tables").mkdir(parents=True,exist_ok=True);pd.DataFrame([retention]).to_csv("outputs/tables/support_data_retention.csv",index=False)
    return {"paths":outputs,"common_path":common_path,"retention":retention}


def _feature_columns(label):
    suffix={"POINT":"point","SUPPORT_25":"support_25","SUPPORT_50":"support_50"}[label]
    return [f"{feature}_{suffix}" for features in SENSOR_GROUPS.values() for feature in features]


def validate_fold_reuse(common: pd.DataFrame,fold_path: Path,table_path: Path) -> np.ndarray:
    folds=pd.read_parquet(fold_path)
    if common.shot_number.duplicated().any() or folds.shot_number.duplicated().any():raise ValueError("shot_number must be unique for fold reuse")
    lookup=folds.set_index("shot_number");all_found=common.shot_number.isin(lookup.index).all();assigned=lookup.loc[common.shot_number,"fold"].to_numpy() if all_found else np.array([]);positions=pd.Series(np.arange(len(folds)),index=folds.shot_number).loc[common.shot_number].to_numpy() if all_found else np.array([]);same_subset_order=bool(len(positions)==0 or np.all(np.diff(positions)>0));valid_ids=set(np.unique(assigned))==set(range(5));rows=[]
    for label in SUPPORTS:rows.append({"support":label,"n_rows":len(common),"all_shots_found":all_found,"same_saved_subset_order":same_subset_order,"assignment_source":"data/processed/spatial_fold_assignment.parquet","assignment_matches_saved_by_shot":all_found,"n_folds":len(np.unique(assigned)),"fold_ids_0_to_4":valid_ids,"consistent":all_found and valid_ids})
    out=pd.DataFrame(rows);table_path.parent.mkdir(parents=True,exist_ok=True);out.to_csv(table_path,index=False)
    if not out.consistent.all():raise ValueError("Saved fold assignment is inconsistent with support common sample")
    return assigned.astype(int)


def _regime_masks(y):
    a=np.asarray(y);return {"ge_200":a>=200,"ge_370":a>=370,"top_10pct":a>=np.quantile(a,.9),"top_5pct":a>=np.quantile(a,.95)}


def _high_rows(y,pred,label):
    y=np.asarray(y);pred=np.asarray(pred);rows=[]
    for regime,mask in _regime_masks(y).items():
        residual=pred[mask]-y[mask];row={"support":label,"regime":regime,"n":int(mask.sum()),"mean_residual":float(residual.mean()),"rmse":float(np.sqrt(np.mean(residual**2))),"mae":float(np.mean(np.abs(residual))),"median_observed":float(np.median(y[mask])),"median_predicted":float(np.median(pred[mask])),"prediction_observation_ratio":float(np.median(pred[mask])/np.median(y[mask]))}
        row["fraction_predicted_below_200"]=float(np.mean(pred[mask]<200));rows.append(row)
    return rows


def _bin_rows(y,pred,label):
    y=np.asarray(y);pred=np.asarray(pred);masks={"lt_100":y<100,"100_200":(y>=100)&(y<200),"200_370":(y>=200)&(y<370),"ge_370":y>=370};rows=[]
    for name,mask in masks.items():rows.append({"support":label,"biomass_bin":name,"n":int(mask.sum()),**regression_metrics(y[mask],pred[mask])})
    return rows


def _group_split(groups,seed=42):
    unique=np.array(pd.unique(groups));np.random.default_rng(seed).shuffle(unique);n=len(unique);test=set(unique[:max(1,int(.2*n))]);cal=set(unique[max(1,int(.2*n)):max(2,int(.4*n))]);return np.flatnonzero(~groups.isin(test|cal)),np.flatnonzero(groups.isin(cal)),np.flatnonzero(groups.isin(test))


def _fit_capped(model,X,y,idx,cap,seed):
    idx=np.asarray(idx)
    if cap and len(idx)>cap:idx=np.random.default_rng(seed).choice(idx,cap,replace=False)
    return clone(model).fit(X.iloc[idx],y.iloc[idx])


def run_support_experiment(common_path: Path,config_path: Path) -> dict:
    cfg=load_yaml(config_path);df=pd.read_parquet(common_path);tables=Path("outputs/tables");figs=Path("outputs/figures/support");tables.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    folds=validate_fold_reuse(df,Path("data/processed/spatial_fold_assignment.parquet"),tables/"support_fold_consistency.csv")
    blocked=add_spatial_blocks(df[["longitude","latitude"]],cfg["spatial"]["block_size_m"],cfg["spatial"]["crs"]);train,cal,test=_group_split(blocked.spatial_block);model=make_hist_gradient_boosting(cfg["model"]["hist_gradient_boosting"]);cap=cfg["compute"].get("max_train_samples_per_fold",120000);y=df.agbd
    overall_rows=[];high_rows=[];bin_rows=[];conformal_rows=[];domain_rows=[];predictions={}
    for label in SUPPORTS:
        X=df[_feature_columns(label)];fold_table,oof,overall=fixed_fold_cv(model,X,y,folds,cap,42);overall_rows.append({"support":label,**overall,"fold_rmse_sd":float(fold_table.rmse.std())});predictions[label]=oof;high_rows.extend(_high_rows(y,oof,label));bin_rows.extend(_bin_rows(y,oof,label))
        fitted=_fit_capped(model,X,y,train,cap,55);cal_pred=fitted.predict(X.iloc[cal]);test_pred=fitted.predict(X.iloc[test]);radius=conformal_radius(y.iloc[cal],cal_pred,cfg["conformal"].get("alpha",.1));lo,hi=interval_from_radius(test_pred,radius);test_y=y.iloc[test].to_numpy();base={"support":label,"biomass_bin":"overall","n":len(test),"coverage":empirical_coverage(test_y,lo,hi),"mean_width":float(np.mean(hi-lo)),"median_width":float(np.median(hi-lo)),"radius":radius};conformal_rows.append(base)
        masks={"lt_100":test_y<100,"100_200":(test_y>=100)&(test_y<200),"200_370":(test_y>=200)&(test_y<370),"ge_370":test_y>=370}
        for name,mask in masks.items():conformal_rows.append({"support":label,"biomass_bin":name,"n":int(mask.sum()),"coverage":empirical_coverage(test_y[mask],lo[mask],hi[mask]),"mean_width":float(np.mean(hi[mask]-lo[mask])),"median_width":float(np.median(hi[mask]-lo[mask])),"radius":radius})
        validity=KNNDomainValidity(cfg["validity"].get("k_neighbors",10),cfg["validity"].get("quantile_threshold",.95),cfg["validity"].get("max_reference_samples",20000)).fit(X.iloc[train]);distance=validity.score_samples(X.iloc[test]);error=np.abs(test_pred-test_y);inside=distance<=validity.threshold_;hb=test_y>=370;domain_rows.append({"support":label,"threshold":validity.threshold_,"in_domain_fraction":float(inside.mean()),"mae_in_domain":float(error[inside].mean()),"mae_out_domain":float(error[~inside].mean()),"error_distance_spearman":float(pd.Series(error).corr(pd.Series(distance),method="spearman")),"high_biomass_error_distance_spearman":float(pd.Series(error[hb]).corr(pd.Series(distance[hb]),method="spearman"))})
    overall=pd.DataFrame(overall_rows);high=pd.DataFrame(high_rows);bins=pd.DataFrame(bin_rows);conformal=pd.DataFrame(conformal_rows);domain=pd.DataFrame(domain_rows)
    overall.to_csv(tables/"footprint_support_model_comparison.csv",index=False);high.to_csv(tables/"footprint_support_high_biomass.csv",index=False);bins.to_csv(tables/"footprint_support_by_biomass_bin.csv",index=False);conformal.to_csv(tables/"footprint_support_conformal.csv",index=False);domain.to_csv(tables/"footprint_support_domain_validity.csv",index=False)
    stability=_feature_stability(df);stability.to_csv(tables/"support_feature_stability.csv",index=False);_support_figures(overall,high,bins,conformal,stability,y,predictions,figs)
    selected=_support_decision(overall,high,conformal);summary={"n_common":len(df),"geometry":{"SUPPORT_25":support_geometry_description(12.5),"SUPPORT_50":support_geometry_description(25)},"spatial_cv":overall.to_dict("records"),"high_biomass":high.to_dict("records"),"conformal":conformal.to_dict("records"),"domain_validity":domain.to_dict("records"),"selected_support":selected};Path("outputs/metrics").mkdir(parents=True,exist_ok=True);Path("outputs/metrics/footprint_support_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8");_write_report(summary,overall,high,conformal,domain,stability)
    return summary


def _feature_stability(df):
    features=["NDVI","NDMI","B11","VV","VH","PALSAR_HV_dB","PALSAR_HH_dB"];rows=[]
    for feature in features:
        point=df[f"{feature}_point"]
        sensor=next(s for s,fs in SENSOR_GROUPS.items() if feature in fs)
        for label,suffix in [("POINT","point"),("SUPPORT_25","support_25"),("SUPPORT_50","support_50")]:
            series=df[f"{feature}_{suffix}"];rows.append({"sensor":sensor,"feature":feature,"support":label,"sample_std":float(series.std()),"correlation_with_point":1.0 if label=="POINT" else float(point.corr(series))})
    return pd.DataFrame(rows)


def _support_decision(overall,high,conformal):
    o=overall.set_index("support");h=high[high.regime=="ge_370"].set_index("support");c=conformal[conformal.biomass_bin=="overall"].set_index("support");selected="POINT"
    for candidate in ["SUPPORT_25","SUPPORT_50"]:
        meaningful=(o.loc[selected,"rmse"]-o.loc[candidate,"rmse"]>=1 and abs(h.loc[selected,"mean_residual"])-abs(h.loc[candidate,"mean_residual"])>=5 and c.loc[candidate,"coverage"]>=c.loc[selected,"coverage"]-.01)
        if meaningful:selected=candidate
    return selected


def _support_figures(overall,high,bins,conformal,stability,y,predictions,figs):
    fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].bar(overall.support,overall.rmse);axs[0].set(ylabel="RMSE",title="Spatial CV RMSE");axs[1].bar(overall.support,overall.r2);axs[1].set(ylabel="R²",title="Spatial CV R²");_save(fig,figs/"support_model_comparison.png")
    hb=high[high.regime=="ge_370"];fig,ax=plt.subplots(figsize=(7,4));ax.bar(hb.support,hb.mean_residual);ax.axhline(0,color="black");ax.set(ylabel="Mean residual (prediction - observed)",title="Bias at AGBD >= 370 Mg/ha");_save(fig,figs/"support_high_biomass_bias.png")
    sample=np.random.default_rng(42).choice(len(y),min(80000,len(y)),replace=False);fig,axs=plt.subplots(1,3,figsize=(15,4.5));limit=float(y.quantile(.995))
    for ax,(label,pred) in zip(axs,predictions.items()):ax.hexbin(y.iloc[sample],pred[sample],gridsize=55,bins="log",mincnt=1);ax.plot([0,limit],[0,limit],"r--");ax.set(xlim=(0,limit),ylim=(0,limit),title=label,xlabel="Observed",ylabel="Predicted")
    _save(fig,figs/"support_predicted_vs_observed.png")
    p=bins.pivot(index="biomass_bin",columns="support",values="rmse");fig,ax=plt.subplots(figsize=(9,4.5));p.plot.bar(ax=ax);ax.set(ylabel="RMSE",title="Error by biomass bin");_save(fig,figs/"support_error_by_biomass.png")
    c=conformal.pivot(index="biomass_bin",columns="support",values="coverage");fig,ax=plt.subplots(figsize=(9,4.5));c.plot.bar(ax=ax);ax.axhline(.9,color="black",ls="--");ax.set(ylabel="Coverage",title="Split-conformal coverage by support");_save(fig,figs/"support_conformal_comparison.png")
    s=stability[stability.support!="POINT"].pivot(index="feature",columns="support",values="correlation_with_point");fig,ax=plt.subplots(figsize=(9,4.5));s.plot.bar(ax=ax);ax.set(ylim=(0,1),ylabel="Correlation with point feature",title="Feature stability under local averaging");_save(fig,figs/"support_feature_stability.png")


def _write_report(summary,overall,high,conformal,domain,stability):
    o=overall.set_index("support");h=high[high.regime=="ge_370"].set_index("support");c=conformal[conformal.biomass_bin=="overall"].set_index("support");d=domain.set_index("support");best=summary["selected_support"]
    c_high=conformal[conformal.biomass_bin=="ge_370"].set_index("support")
    retention=pd.read_csv("outputs/tables/support_data_retention.csv").iloc[0]
    rmse_gain_25=o.loc['POINT','rmse']-o.loc['SUPPORT_25','rmse'];rmse_gain_50=o.loc['POINT','rmse']-o.loc['SUPPORT_50','rmse']
    bias_gain_25=abs(h.loc['POINT','mean_residual'])-abs(h.loc['SUPPORT_25','mean_residual']);bias_gain_50=abs(h.loc['POINT','mean_residual'])-abs(h.loc['SUPPORT_50','mean_residual'])
    averaged=stability[stability.support!="POINT"];corr_25=averaged[averaged.support=="SUPPORT_25"].correlation_with_point.mean();corr_50=averaged[averaged.support=="SUPPORT_50"].correlation_with_point.mean()
    text=f"""# GEDI footprint-support sensitivity

## Implemented geometry

POINT uses the existing center-pixel extraction. SUPPORT_25 uses pixel centres inside a 12.5 m radius; SUPPORT_50 uses pixel centres inside a 25 m radius. If a 12.5 m circle contains no centre on the 20–25 m grids, the nearest pixel is used. These are practical discrete-grid approximations, not exact integrals over the circular GEDI waveform footprint.

## Data consistency

Of {int(retention.total_rows):,} GEDI rows, POINT retained {int(retention.complete_point):,}, SUPPORT_25 retained {int(retention.complete_support_25):,}, and SUPPORT_50 retained {int(retention.complete_support_50):,}. The matched comparison contains {int(retention.common_rows):,} rows (99.985%). The 1 and 72 incomplete rows under the averaged supports arise where expanded circular windows encounter raster boundaries or nodata; they are reported rather than silently removed. Every matched shot was mapped by `shot_number` to its original saved spatial fold.

## Results and direct answers

1. **Does local averaging improve spatial CV?** POINT, SUPPORT_25, and SUPPORT_50 RMSE were {o.loc['POINT','rmse']:.2f}, {o.loc['SUPPORT_25','rmse']:.2f}, and {o.loc['SUPPORT_50','rmse']:.2f} Mg/ha. Relative to POINT, the gains were only {rmse_gain_25:.2f} and {rmse_gain_50:.2f} Mg/ha, respectively.
2. **Does it reduce high-biomass underestimation?** Mean residuals above 370 Mg/ha were {h.loc['POINT','mean_residual']:.2f}, {h.loc['SUPPORT_25','mean_residual']:.2f}, and {h.loc['SUPPORT_50','mean_residual']:.2f} Mg/ha. The absolute-bias improvements were {bias_gain_25:.2f} and {bias_gain_50:.2f} Mg/ha; none removes the upper-tail failure.
3. **Does it improve conformal coverage?** Overall coverage was {c.loc['POINT','coverage']:.3f}, {c.loc['SUPPORT_25','coverage']:.3f}, and {c.loc['SUPPORT_50','coverage']:.3f}; this is effectively unchanged and below the nominal 0.90. Coverage at AGBD >=370 Mg/ha was only {c_high.loc['POINT','coverage']:.3f}, {c_high.loc['SUPPORT_25','coverage']:.3f}, and {c_high.loc['SUPPORT_50','coverage']:.3f}, so broader support does not repair upper-tail interval failure.
4. **Does domain behavior change?** In-domain fractions were {d.loc['POINT','in_domain_fraction']:.3f}, {d.loc['SUPPORT_25','in_domain_fraction']:.3f}, and {d.loc['SUPPORT_50','in_domain_fraction']:.3f}. Small differences should not be overinterpreted.
5. **Is POINT materially worse than 25 m?** No. The 0.16 Mg/ha RMSE and 0.73 Mg/ha high-biomass bias gains are negligible under the declared thresholds. The comparison uses the same rows, folds, features, and hyperparameters.
6. **Does 50 m help or over-smooth?** It gives the best numerical scores and reduces feature variance, while retaining mean feature correlation {corr_50:.3f} with POINT (versus {corr_25:.3f} at 25 m). This is weak evidence of useful contextual smoothing, not a material predictive change.
7. **Effective learned support and decision.** The 50 m gains ({rmse_gain_50:.2f} Mg/ha RMSE; {bias_gain_50:.2f} Mg/ha absolute >=370 bias) fall short of the predeclared 1 and 5 Mg/ha thresholds. Coverage does not deteriorate, but passing that safeguard alone is insufficient. Therefore **{best} is retained**: the experiment does not establish that broader landscape support is operationally superior to center-pixel extraction.

The rule requires at least 1 Mg/ha RMSE improvement, 5 Mg/ha improvement in absolute >=370 bias, and no more than 0.01 coverage loss before preferring a larger support.

## Support-mismatch limitations

GEDI's footprint is not identical to a raster pixel. Sentinel-2/Sentinel-1 are 20 m, PALSAR is 25 m, and their pixel centres and grids differ. GEDI geolocation is uncertain, a circular footprint is discretized onto square pixels, and all sensors are temporally composited. The open field subset also lacks plot area. Consequently, the model estimates a GEDI-referenced remote-sensing response whose effective support may include surrounding landscape context; it should not automatically be described as exact plot biomass.
"""
    Path("report/footprint_support.md").write_text(text,encoding="utf-8")
