from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import SENSOR_GROUPS
from .raster import sample_raster_blockwise

REQUIRED_FIELD_COLUMNS = ["plot_id","longitude","latitude","field_agbd","measurement_date"]

def validate_field_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing=[c for c in REQUIRED_FIELD_COLUMNS if c not in df]
    if missing: raise ValueError(f"Field table is missing required columns: {missing}")
    out=df.copy(); out["measurement_date"]=pd.to_datetime(out.measurement_date,errors="coerce")
    if out.measurement_date.isna().any(): raise ValueError("measurement_date contains invalid values")
    if "plot_area_ha" in out and (out.plot_area_ha<=0).any(): raise ValueError("plot_area_ha must be positive")
    return out

def temporal_filter(df: pd.DataFrame, start: str|None=None, end: str|None=None) -> pd.DataFrame:
    out=validate_field_schema(df)
    if start: out=out[out.measurement_date>=pd.Timestamp(start)]
    if end: out=out[out.measurement_date<=pd.Timestamp(end)]
    return out.copy()

def sample_map_at_fields(df: pd.DataFrame, prediction_map: str|Path, lower_map: str|Path|None=None, upper_map: str|Path|None=None) -> pd.DataFrame:
    out=validate_field_schema(df); paths=[prediction_map,lower_map,upper_map]; names=["map_agbd","pi_lower","pi_upper"]
    for path,name in zip(paths,names):
        if path is None: continue
        with rasterio.open(path) as src:
            from rasterio.warp import transform
            x,y=transform("EPSG:4326",src.crs,out.longitude.tolist(),out.latitude.tolist()); values=np.array([v[0] for v in src.sample(zip(x,y),masked=True)],dtype=float); values[~np.isfinite(values)]=np.nan; out[name]=values
    return out

def field_validation_metrics(matched: pd.DataFrame) -> dict:
    valid=matched[["field_agbd","map_agbd"]].dropna(); y=valid.field_agbd.to_numpy(); pred=valid.map_agbd.to_numpy(); residual=pred-y
    calibration=LinearRegression().fit(pred.reshape(-1,1),y)
    result={"n":len(valid),"rmse":float(np.sqrt(np.mean(residual**2))),"mae":float(np.mean(np.abs(residual))),"bias":float(np.mean(residual)),"calibration_intercept":float(calibration.intercept_),"calibration_slope":float(calibration.coef_[0])}
    if {"pi_lower","pi_upper"}<=set(matched):
        v=matched[["field_agbd","pi_lower","pi_upper"]].dropna(); result["interval_coverage"]=float(((v.field_agbd>=v.pi_lower)&(v.field_agbd<=v.pi_upper)).mean())
    return result

def regression_summary(y_true, y_pred) -> dict:
    pair=pd.DataFrame({"true":y_true,"pred":y_pred}).dropna(); n=len(pair)
    result={"n":n,"mae":np.nan,"rmse":np.nan,"bias":np.nan,"median_absolute_error":np.nan,"spearman_r":np.nan,"pearson_r":np.nan,"r2_descriptive":np.nan,"regression_slope":np.nan,"regression_intercept":np.nan}
    if n==0:return result
    y=pair.true.to_numpy();p=pair.pred.to_numpy();res=p-y
    result.update(mae=float(mean_absolute_error(y,p)),rmse=float(np.sqrt(mean_squared_error(y,p))),bias=float(np.mean(res)),median_absolute_error=float(np.median(np.abs(res))))
    if n>=2 and np.std(y)>0 and np.std(p)>0:
        result.update(spearman_r=float(pair.true.corr(pair.pred,method="spearman")),pearson_r=float(pair.true.corr(pair.pred,method="pearson")),r2_descriptive=float(r2_score(y,p)))
        fit=LinearRegression().fit(y.reshape(-1,1),p);result.update(regression_slope=float(fit.coef_[0]),regression_intercept=float(fit.intercept_))
    return result

def interval_coverage_summary(field,lower,upper) -> dict:
    frame=pd.DataFrame({"field":field,"lower":lower,"upper":upper}).dropna();covered=(frame.field>=frame.lower)&(frame.field<=frame.upper);width=frame.upper-frame.lower
    return {"n":len(frame),"n_covered":int(covered.sum()),"empirical_coverage":float(covered.mean()) if len(frame) else np.nan,"mean_interval_width":float(width.mean()) if len(frame) else np.nan,"median_interval_width":float(width.median()) if len(frame) else np.nan}

def nearest_gedi_matches(field: pd.DataFrame, gedi: pd.DataFrame, crs="EPSG:32629") -> pd.DataFrame:
    field=validate_field_schema(field);required={"longitude","latitude","agbd","agbd_se","relative_se"}
    if not required<=set(gedi):raise ValueError(f"GEDI table missing columns: {sorted(required-set(gedi))}")
    transformer=Transformer.from_crs("EPSG:4326",crs,always_xy=True);gx,gy=transformer.transform(gedi.longitude.to_numpy(),gedi.latitude.to_numpy());fx,fy=transformer.transform(field.longitude.to_numpy(),field.latitude.to_numpy());distance,index=cKDTree(np.column_stack([gx,gy])).query(np.column_stack([fx,fy]),k=1)
    selected=gedi.iloc[index].reset_index(drop=True);out=field.reset_index(drop=True).copy();out["nearest_distance_m"]=distance;out["gedi_longitude"]=selected.longitude;out["gedi_latitude"]=selected.latitude;out["gedi_agbd"]=selected.agbd;out["gedi_agbd_se"]=selected.agbd_se;out["gedi_relative_se"]=selected.relative_se
    for optional in ["shot_number","delta_time"]:
        if optional in selected:out[f"gedi_{optional}"]=selected[optional]
    if "delta_time" in selected:
        out["gedi_acquisition_datetime"]=pd.Timestamp("2018-01-01")+pd.to_timedelta(selected.delta_time,unit="s");out["temporal_difference_days"]=(out.gedi_acquisition_datetime-out.measurement_date).abs().dt.total_seconds()/86400
    return out

def threshold_match_counts(nearest: pd.DataFrame, thresholds=(25,50,100,250,500)) -> pd.DataFrame:
    rows=[]
    for threshold in thresholds:
        matched=nearest[nearest.nearest_distance_m<=threshold]
        rows.append({"threshold_m":threshold,"n_matches":len(matched),"median_distance_m":float(matched.nearest_distance_m.median()) if len(matched) else np.nan,"median_temporal_difference_days":float(matched.temporal_difference_days.median()) if len(matched) and "temporal_difference_days" in matched else np.nan})
    return pd.DataFrame(rows)

def gedi_support_sensitivity(field: pd.DataFrame,gedi: pd.DataFrame,radii=(25,50,100),crs="EPSG:32629") -> pd.DataFrame:
    transformer=Transformer.from_crs("EPSG:4326",crs,always_xy=True);gx,gy=transformer.transform(gedi.longitude.to_numpy(),gedi.latitude.to_numpy());fx,fy=transformer.transform(field.longitude.to_numpy(),field.latitude.to_numpy());tree=cKDTree(np.column_stack([gx,gy]));nearest_distance,nearest_idx=tree.query(np.column_stack([fx,fy]),k=1);rows=[]
    for i,plot in field.reset_index(drop=True).iterrows():
        base={"plot_id":plot.plot_id,"field_agbd":plot.field_agbd,"nearest_distance_m":nearest_distance[i],"nearest_gedi_agbd":float(gedi.iloc[nearest_idx[i]].agbd)}
        for radius in radii:
            idx=tree.query_ball_point([fx[i],fy[i]],radius);vals=gedi.iloc[idx].agbd.dropna();base[f"n_gedi_within_{radius}m"]=len(vals);base[f"median_gedi_within_{radius}m"]=float(vals.median()) if len(vals)>=2 else np.nan;base[f"iqr_gedi_within_{radius}m"]=float(vals.quantile(.75)-vals.quantile(.25)) if len(vals)>=2 else np.nan
        rows.append(base)
    return pd.DataFrame(rows)

def _save(fig,path):
    path.parent.mkdir(parents=True,exist_ok=True);fig.tight_layout();fig.savefig(path,dpi=220,bbox_inches="tight");plt.close(fig)

def _sample_final_maps(field: pd.DataFrame,map_dir: Path) -> pd.DataFrame:
    out=field.copy();mapping={"biomass_prediction.tif":"predicted_agbd","biomass_pi_lower.tif":"pi_lower","biomass_pi_upper.tif":"pi_upper","biomass_interval_width.tif":"interval_width","domain_distance.tif":"map_domain_distance","domain_validity_mask.tif":"map_domain_validity_mask"}
    for filename,column in mapping.items():
        path=map_dir/filename
        if not path.exists():raise FileNotFoundError(path)
        with rasterio.open(path) as src:
            transformer=Transformer.from_crs("EPSG:4326",src.crs,always_xy=True);x,y=transformer.transform(out.longitude.to_numpy(),out.latitude.to_numpy());values=np.array([np.nan if np.ma.is_masked(v[0]) else float(v[0]) for v in src.sample(zip(x,y),masked=True)],dtype=float);values[~np.isfinite(values)]=np.nan;out[column]=values
    out["map_in_domain"]=out.map_domain_validity_mask.eq(1);return out.drop(columns="map_domain_validity_mask")

def _add_exact_bundle_domain(field_predictions: pd.DataFrame,bundle_path: Path,raster_paths: dict[str,Path]) -> pd.DataFrame:
    bundle=joblib.load(bundle_path);parts=[]
    for sensor,key in [("S2","sentinel2"),("S1","sentinel1"),("PALSAR","palsar")]:
        parts.append(sample_raster_blockwise(field_predictions,raster_paths[key],SENSOR_GROUPS[sensor]))
    features=pd.concat(parts,axis=1)[bundle["features"]]
    if features.isna().any(axis=None):raise ValueError("Native sensor features are missing for one or more field plots")
    out=field_predictions.copy();out["domain_distance"]=bundle["validity"].score_samples(features);out["in_domain"]=out.domain_distance<=bundle["validity"].threshold_;return out

def _json_safe(value):
    if isinstance(value,dict):return {str(k):_json_safe(v) for k,v in value.items()}
    if isinstance(value,list):return [_json_safe(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,)):value=float(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    return value

def run_field_validation(field_path: Path,gedi_path: Path,map_dir: Path,output_root=Path("outputs"),bundle_path=Path("outputs/models/final_multisensor_model_bundle.joblib"),raster_paths=None) -> dict:
    tables=output_root/"tables";figs=output_root/"figures"/"field";metrics_dir=output_root/"metrics";tables.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True);metrics_dir.mkdir(parents=True,exist_ok=True);Path("report").mkdir(exist_ok=True)
    raw=pd.read_csv(field_path);field=validate_field_schema(raw);audit_rows=[{"metric":"n_plots","value":len(field)},{"metric":"date_min","value":field.measurement_date.min().isoformat()},{"metric":"date_max","value":field.measurement_date.max().isoformat()},{"metric":"field_agbd_min","value":field.field_agbd.min()},{"metric":"field_agbd_max","value":field.field_agbd.max()},{"metric":"duplicate_plot_id","value":int(field.plot_id.duplicated().sum())},{"metric":"duplicate_coordinates","value":int(field.duplicated(["longitude","latitude"]).sum())},{"metric":"invalid_coordinates","value":int((~field.longitude.between(-180,180)|~field.latitude.between(-90,90)).sum())}]
    audit_rows.extend({"metric":f"missing_{c}","value":int(field[c].isna().sum())} for c in field.columns);audit_rows.extend([{"metric":"plot_area_available","value":int("plot_area_ha" in field)},{"metric":"allometric_method_available","value":int("allometric_method" in field)}]);pd.DataFrame(audit_rows).to_csv(tables/"field_validation_audit.csv",index=False)
    raster_paths=raster_paths or {"sentinel2":Path("data/raw/sentinel2_tai_features_2020_2024.tif"),"sentinel1":Path("data/raw/sentinel1_tai_features_2020_2024.tif"),"palsar":Path("data/raw/palsar_tai_features_2020_2024.tif")}
    predictions=_add_exact_bundle_domain(_sample_final_maps(field,map_dir),bundle_path,raster_paths);predictions.to_csv(tables/"field_model_predictions.csv",index=False);per_plot=predictions.copy();per_plot["residual"]=per_plot.predicted_agbd-per_plot.field_agbd;per_plot["absolute_error"]=per_plot.residual.abs();per_plot["relative_error"]=np.where(per_plot.field_agbd>0,per_plot.residual/per_plot.field_agbd,np.nan);per_plot["covered"]=per_plot.field_agbd.between(per_plot.pi_lower,per_plot.pi_upper);per_plot.to_csv(tables/"field_vs_model_per_plot.csv",index=False)
    model_metrics=regression_summary(per_plot.field_agbd,per_plot.predicted_agbd);model_metrics.update(note="R2 is descriptive only; n=10 is too small for a stable performance estimate");pd.DataFrame([model_metrics]).to_csv(tables/"field_vs_model_metrics.csv",index=False)
    coverage=interval_coverage_summary(per_plot.field_agbd,per_plot.pi_lower,per_plot.pi_upper);coverage["interpretation"]="GEDI-calibrated interval transfer check, not a formal field-truth coverage guarantee";pd.DataFrame([coverage]).to_csv(tables/"field_interval_coverage.csv",index=False)
    domain=per_plot[["plot_id","field_agbd","predicted_agbd","absolute_error","domain_distance","in_domain"]].copy();domain.to_csv(tables/"field_domain_validity.csv",index=False)
    gedi=pd.read_csv(gedi_path);nearest=nearest_gedi_matches(field,gedi);nearest.to_csv(tables/"field_gedi_nearest.csv",index=False);counts=threshold_match_counts(nearest);counts.to_csv(tables/"field_gedi_match_counts.csv",index=False);support=gedi_support_sensitivity(field,gedi);support.to_csv(tables/"field_gedi_support_sensitivity.csv",index=False)
    gedi_metrics=[]
    for threshold in [50,100,250]:
        matched=nearest[nearest.nearest_distance_m<=threshold];row={"threshold_m":threshold,**regression_summary(matched.field_agbd,matched.gedi_agbd)};row["median_gedi_minus_field"]=float((matched.gedi_agbd-matched.field_agbd).median()) if len(matched) else np.nan;gedi_metrics.append(row)
    gedi_metrics=pd.DataFrame(gedi_metrics);gedi_metrics.to_csv(tables/"field_vs_gedi_metrics.csv",index=False)
    _field_figures(per_plot,nearest,figs)
    summary={"n_field_plots_roi":len(field),"field_agbd_range":[float(field.field_agbd.min()),float(field.field_agbd.max())],"model_metrics":model_metrics,"model_interval_field_coverage":coverage,"nearest_gedi_distance_m":{"min":float(nearest.nearest_distance_m.min()),"median":float(nearest.nearest_distance_m.median()),"max":float(nearest.nearest_distance_m.max())},"field_gedi_match_counts":{str(int(r.threshold_m)):int(r.n_matches) for _,r in counts.iterrows()},"field_vs_gedi_metrics":gedi_metrics.to_dict("records"),"n_field_in_domain":int(per_plot.in_domain.sum()),"n_field_out_domain":int((~per_plot.in_domain).sum()),"gedi_time_reconstruction":"delta_time converted as seconds since 2018-01-01 00:00","caveats":["Only 10 ROI plots; metrics are unstable.","Plots were never used for training, tuning, calibration, feature selection, or model selection.","Plot area and detailed allometric method are absent from the downloaded table.","Only one field plot has a GEDI shot within 250 m; field-vs-GEDI performance is not estimable.","No local GEDI median is available because no plot has at least two shots within 100 m.","GEDI-calibrated intervals have no formal field-truth coverage guarantee.","Field, GEDI footprint, and map-pixel spatial supports are not guaranteed equivalent."]}
    (metrics_dir/"field_validation_summary.json").write_text(json.dumps(_json_safe(summary),indent=2,allow_nan=False),encoding="utf-8");_write_field_reports(summary,per_plot,nearest,counts,gedi_metrics,support)
    return {"field":field,"model_metrics":model_metrics,"nearest":nearest,"counts":counts,"gedi_metrics":gedi_metrics,"coverage":coverage,"summary":summary}

def _field_figures(per_plot,nearest,figs):
    fig,ax=plt.subplots(figsize=(6,5));ax.scatter(per_plot.field_agbd,per_plot.predicted_agbd);lo=min(per_plot.field_agbd.min(),per_plot.predicted_agbd.min());hi=max(per_plot.field_agbd.max(),per_plot.predicted_agbd.max());ax.plot([lo,hi],[lo,hi],"k--");
    for _,r in per_plot.iterrows():ax.annotate(str(r.plot_id).split("_")[1]+"_"+str(r.plot_id).split("_")[2],(r.field_agbd,r.predicted_agbd),fontsize=7)
    ax.set(xlabel="Field AGBD (Mg/ha)",ylabel="Predicted AGBD (Mg/ha)",title=f"Independent field comparison (n={len(per_plot)})");_save(fig,figs/"field_vs_model.png")
    fig,ax=plt.subplots(figsize=(7,4.5));ax.axhline(0,color="black",ls="--");ax.scatter(per_plot.predicted_agbd,per_plot.residual,c=per_plot.in_domain.map({True:"tab:blue",False:"tab:red"}));ax.set(xlabel="Predicted AGBD",ylabel="Residual (predicted - field)",title="Field residuals");_save(fig,figs/"field_model_residuals.png")
    matched=nearest[nearest.nearest_distance_m<=250];fig,ax=plt.subplots(figsize=(6,5));ax.scatter(matched.field_agbd,matched.gedi_agbd)
    if len(matched):
        lo=min(matched.field_agbd.min(),matched.gedi_agbd.min());hi=max(matched.field_agbd.max(),matched.gedi_agbd.max());ax.plot([lo,hi],[lo,hi],"k--")
    else: ax.text(.5,.5,"No GEDI match within 250 m",ha="center",va="center",transform=ax.transAxes)
    for _,r in matched.iterrows():ax.annotate(f"{r.nearest_distance_m:.0f}m",(r.field_agbd,r.gedi_agbd),fontsize=7)
    ax.set(xlabel="Field AGBD",ylabel="Nearest GEDI AGBD",title=f"Field vs GEDI within 250 m (n={len(matched)})");_save(fig,figs/"field_vs_gedi.png")
    order=per_plot.sort_values("field_agbd").reset_index(drop=True);x=np.arange(len(order));covered=order.covered.to_numpy();fig,ax=plt.subplots(figsize=(10,5));ax.vlines(x,order.pi_lower,order.pi_upper,color=np.where(covered,"tab:green","tab:red"),alpha=.7);ax.scatter(x,order.predicted_agbd,label="Prediction");ax.scatter(x,order.field_agbd,marker="x",color="black",label="Field");ax.set_xticks(x,order.plot_id,rotation=60,ha="right",fontsize=7);ax.set(ylabel="AGBD (Mg/ha)",title="Transfer of GEDI-calibrated intervals to field plots");ax.legend();_save(fig,figs/"field_prediction_intervals.png")
    fig,ax=plt.subplots(figsize=(6,4.5));ax.scatter(per_plot.domain_distance,per_plot.absolute_error,c=per_plot.in_domain.map({True:"tab:blue",False:"tab:red"}));
    for _,r in per_plot.iterrows():ax.annotate(str(r.plot_id)[-3:],(r.domain_distance,r.absolute_error),fontsize=7)
    ax.set(xlabel="Feature-space domain distance",ylabel="Absolute field error",title="Field error versus domain distance (n=10)");_save(fig,figs/"field_error_vs_domain_distance.png")

def _write_field_reports(summary,per_plot,nearest,counts,gedi_metrics,support):
    mm=summary["model_metrics"];cov=summary["model_interval_field_coverage"];count_map=summary["field_gedi_match_counts"];in_domain=summary["n_field_in_domain"]
    temporal=nearest.temporal_difference_days;largest=per_plot.nlargest(2,"absolute_error")
    text=f"""# Independent field validation

## Dataset and independence

Ten field plots strictly inside the ROI were used only after final model selection. They were not used for training, tuning, conformal calibration, feature selection, or model selection. Measurements date from {per_plot.measurement_date.min().date()} to {per_plot.measurement_date.max().date()} and range from {per_plot.field_agbd.min():.2f} to {per_plot.field_agbd.max():.2f} Mg/ha.

## Model versus field

For n=10, MAE was {mm['mae']:.2f} Mg/ha, RMSE {mm['rmse']:.2f}, bias {mm['bias']:.2f}, Spearman correlation {mm['spearman_r']:.3f}, and descriptive R² {mm['r2_descriptive']:.3f}. R² is not a stable generalization estimate at this sample size.

## GEDI matching and temporal support

GEDI acquisition dates were reconstructed from `delta_time` as seconds since 2018-01-01 00:00. Nearest distances ranged from {nearest.nearest_distance_m.min():.1f} to {nearest.nearest_distance_m.max():.1f} m (median {nearest.nearest_distance_m.median():.1f} m). Match counts were {count_map}. Absolute field–GEDI temporal differences ranged from {temporal.min():.0f} to {temporal.max():.0f} days (median {temporal.median():.0f}). Threshold-specific statistics are retained even where n is zero or too small for correlation.

## Interval transfer and domain validity

The GEDI-calibrated intervals covered {cov['n_covered']}/{cov['n']} field values ({cov['empirical_coverage']:.1%}), with mean width {cov['mean_interval_width']:.2f} Mg/ha; this is a transfer diagnostic, not a formal 90% field-truth guarantee. Exact fitted-bundle kNN distances from native sensor point samples classified {in_domain}/10 field plots as in-domain. The interpolated map-domain value is retained separately. The two largest errors ({largest.absolute_error.iloc[0]:.2f} and {largest.absolute_error.iloc[1]:.2f} Mg/ha) were out-of-domain, but other out-of-domain plots had small errors. With n=10, no error-distance relationship is fitted or claimed.

## Spatial-support sensitivity and limitations

The downloaded table has coordinates, date, and biomass but no plot area or detailed allometric-method metadata. Strict equivalence with a ~25 m GEDI footprint therefore cannot be assumed. No local GEDI median could be calculated because no plot had at least two shots within 100 m. Conclusions are an independent consistency check, not absolute pixel-level validation. The sample is too small for strong generalization claims, but is more independent of the GEDI-derived target than internal validation. Dedicated field campaigns or access to complete national-inventory metadata are needed.
""";Path("report/field_validation.md").write_text(text,encoding="utf-8")
    section=f"""### Independent field validation

#### 1. Dataset and independence from training

Ten ROI plots measured on 14–15 December 2022 were held out from every training, tuning, calibration, feature-selection, and model-selection step. Field AGBD ranged from {per_plot.field_agbd.min():.2f} to {per_plot.field_agbd.max():.2f} Mg/ha.

#### 2. Spatial and temporal matching protocol

Nearest GEDI shots were found in EPSG:32629. Distances ranged from {nearest.nearest_distance_m.min():.1f} to {nearest.nearest_distance_m.max():.1f} m, with only one match within 250 m. GEDI times reconstructed from `delta_time` differed from field dates by about {temporal.median():.0f} days.

#### 3. Model versus field results

For n=10, MAE was {mm['mae']:.2f} Mg/ha, RMSE {mm['rmse']:.2f}, bias {mm['bias']:.2f}, Spearman correlation {mm['spearman_r']:.3f}, and descriptive R² {mm['r2_descriptive']:.3f}. These estimates are unstable at n=10.

#### 4. GEDI versus field consistency

There were {count_map['50']}, {count_map['100']}, and {count_map['250']} nearest matches within 50, 100, and 250 m. Because all three thresholds contain the same single plot, field-versus-GEDI performance cannot be estimated. That shot differed from the field value by {gedi_metrics.loc[gedi_metrics.threshold_m==250,'bias'].iloc[0]:.2f} Mg/ha.

#### 5. Prediction interval transfer

GEDI-calibrated intervals covered {cov['n_covered']}/{cov['n']} field observations, but their mean width was {cov['mean_interval_width']:.2f} Mg/ha. This is not a formal field-truth coverage guarantee.

#### 6. Domain of validity

Exact bundle-based feature distances classified {in_domain}/10 plots as in-domain. The two largest errors were out-of-domain, but out-of-domain status did not invariably imply a large error.

#### 7. Limitations

The table lacks plot area and detailed allometric metadata; spatial support equivalence is unknown. No local GEDI median was possible within 100 m, the temporal gap was about three years, and n=10 precludes broad claims.

#### 8. Operational implication

These plots provide an independent consistency check, not definitive pixel-level validation. Dedicated field campaigns with plot geometry, allometry, contemporaneous acquisitions, and broader spatial coverage are required before operational accuracy or stock claims.
"""
    Path("report/field_validation_report_section.md").write_text(section,encoding="utf-8")
