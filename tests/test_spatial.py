import pandas as pd

from biomass_mapping.spatial import add_spatial_blocks


def test_blocks_created():
    df = pd.DataFrame({
        "longitude": [-7.0, -7.01, -6.5],
        "latitude": [5.5, 5.51, 5.7],
    })
    out = add_spatial_blocks(df, block_size_m=10000, crs="EPSG:32629")
    assert "spatial_block" in out.columns
    assert len(out) == 3
    assert out.loc[0, "spatial_block"] == out.loc[1, "spatial_block"]
