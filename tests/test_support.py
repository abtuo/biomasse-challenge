import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from biomass_mapping.support import extract_local_features, support_geometry_description
from biomass_mapping.support_experiment import validate_fold_reuse


def _write_raster(path, values, nodata=None):
    profile={"driver":"GTiff","width":values.shape[1],"height":values.shape[0],"count":1,"dtype":"float32","crs":"EPSG:32631","transform":from_origin(499985,25,10,10),"nodata":nodata}
    with rasterio.open(path,"w",**profile) as dst:
        dst.write(values.astype("float32"),1);dst.set_band_description(1,"band")


def _point(x,y):
    lon,lat=Transformer.from_crs("EPSG:32631","EPSG:4326",always_xy=True).transform(x,y)
    return {"longitude":lon,"latitude":lat}


def test_circular_local_mean_and_same_row_preservation(tmp_path):
    path=tmp_path/"grid.tif";_write_raster(path,np.arange(1,10).reshape(3,3))
    points=pd.DataFrame([_point(500000,10)],index=[42])
    out=extract_local_features(points,path,15,suffix="mean")
    assert out.index.tolist()==[42]
    assert np.isclose(out.iloc[0,0],5.0)


def test_boundary_and_nodata_are_safe(tmp_path):
    path=tmp_path/"grid.tif";values=np.arange(1,10).reshape(3,3).astype(float);values[1,1]=-9999;_write_raster(path,values,nodata=-9999)
    boundary=pd.DataFrame([_point(499990,20)])
    out=extract_local_features(boundary,path,15,suffix="mean")
    assert np.isfinite(out.iloc[0,0])
    centre=pd.DataFrame([_point(500000,10)])
    tiny=extract_local_features(centre,path,4,suffix="tiny")
    assert tiny.isna().iloc[0,0]


def test_point_like_mean_on_uniform_raster(tmp_path):
    path=tmp_path/"uniform.tif";_write_raster(path,np.full((3,3),7.0))
    points=pd.DataFrame([_point(499990,0),_point(500000,10)])
    for radius in [12.5,25]:
        out=extract_local_features(points,path,radius,suffix="x")
        assert np.allclose(out.iloc[:,0],7.0)
    assert "nearest pixel fallback" in support_geometry_description(12.5)


def test_saved_folds_are_reused_by_shot_identity(tmp_path):
    folds=pd.DataFrame({"shot_number":np.arange(10),"spatial_block":[str(i) for i in range(10)],"fold":np.tile(np.arange(5),2)})
    path=tmp_path/"folds.parquet";folds.to_parquet(path,index=False)
    common=pd.DataFrame({"shot_number":[5,1,7,3,4,0,2,6,8,9]})
    assigned=validate_fold_reuse(common,path,tmp_path/"check.csv")
    assert assigned.tolist()==folds.set_index("shot_number").loc[common.shot_number,"fold"].tolist()
    assert pd.read_csv(tmp_path/"check.csv").consistent.all()
