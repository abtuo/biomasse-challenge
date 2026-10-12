import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from sklearn.linear_model import LinearRegression

from biomass_mapping.data import SENSOR_GROUPS, combination_features, validate_sensor_groups
from biomass_mapping.field_validation import field_validation_metrics, temporal_filter
from biomass_mapping.modeling import fixed_fold_cv, make_ridge
from biomass_mapping.multisensor import (
    grouped_permutation_importance,
    high_biomass_metrics,
    select_final_stack,
)
from biomass_mapping.raster import sample_raster_blockwise


def test_sensor_groups_are_explicit_and_complete():
    data = {c: [1.0] for cols in SENSOR_GROUPS.values() for c in cols}
    groups = validate_sensor_groups(pd.DataFrame(data))
    combos = combination_features(groups)
    expected = [feature for features in SENSOR_GROUPS.values() for feature in features]
    assert combos["S2+S1+PALSAR"] == expected


def test_blockwise_raster_feature_extraction(tmp_path):
    path = tmp_path / "tiny.tif"
    profile = {"driver":"GTiff","width":2,"height":2,"count":2,"dtype":"float32","crs":"EPSG:4326","transform":from_origin(0,2,1,1),"tiled":False}
    with rasterio.open(path,"w",**profile) as dst:
        dst.write(np.array([[[1,2],[3,4]],[[10,20],[30,40]]],dtype="float32")); dst.set_band_description(1,"A");dst.set_band_description(2,"B")
    points=pd.DataFrame({"longitude":[.5,1.5],"latitude":[1.5,.5]})
    out=sample_raster_blockwise(points,path,["A","B"])
    assert out.to_numpy().tolist()==[[1.0,10.0],[4.0,40.0]]


def test_same_folds_reused_across_feature_sets():
    X=pd.DataFrame({"a":np.arange(20.),"b":np.arange(20.)**2});y=pd.Series(np.arange(20.));folds=np.tile(np.arange(5),4)
    f1,_,_=fixed_fold_cv(make_ridge(),X[["a"]],y,folds)
    f2,_,_=fixed_fold_cv(make_ridge(),X[["a","b"]],y,folds)
    assert f1[["fold","n_test"]].equals(f2[["fold","n_test"]])


def test_high_biomass_metrics_and_severe_underestimate():
    y=pd.Series([50.,150.,250.,400.,500.]);pred=np.array([50.,140.,200.,150.,200.])
    result=high_biomass_metrics(y,pred,"test")
    assert result["mean_residual_ge_370"] < 0
    assert result["fraction_ge_370_predicted_below_200"] == .5


def test_grouped_permutation_importance_detects_signal():
    rng=np.random.default_rng(2);X=pd.DataFrame({"signal":rng.normal(size=200),"noise":rng.normal(size=200)});y=4*X.signal+rng.normal(scale=.1,size=200);model=LinearRegression().fit(X,y)
    result=grouped_permutation_importance(model,X,y,{"signal":["signal"],"noise":["noise"]},repeats=2)
    assert result.set_index("sensor").loc["signal","importance_mean"] > result.set_index("sensor").loc["noise","importance_mean"]


def test_final_selection_balances_global_and_high_biomass():
    overall=pd.DataFrame({"sensor_stack":["S2","S2+PALSAR"],"rmse":[100.,99.],"r2":[.2,.22],"fold_rmse_sd":[2.,2.]})
    diagnostics=pd.DataFrame({"sensor_stack":["S2","S2+PALSAR"],"rmse_ge_370":[250.,220.],"mean_residual_ge_370":[-200.,-160.]})
    assert select_final_stack(overall,diagnostics).iloc[0].sensor_stack=="S2+PALSAR"


def test_field_validation_schema_time_and_metrics():
    fields=pd.DataFrame({"plot_id":["a","b"],"longitude":[0.,1.],"latitude":[1.,2.],"field_agbd":[100.,200.],"plot_area_ha":[1.,1.],"measurement_date":["2021-01-01","2025-01-01"],"source":["a","a"],"map_agbd":[110.,180.],"pi_lower":[90.,150.],"pi_upper":[130.,210.]})
    assert len(temporal_filter(fields,end="2024-12-31"))==1
    metrics=field_validation_metrics(fields)
    assert metrics["n"]==2
    assert metrics["bias"]==-5.0
    assert metrics["interval_coverage"]==1.0
