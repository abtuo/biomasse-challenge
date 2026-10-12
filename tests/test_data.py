import pandas as pd
import pytest

from biomass_mapping.data import validate_s2_schema


def test_schema_selects_only_recognized_sentinel_features():
    df = pd.DataFrame({"agbd":[1.], "agbd_se":[.1], "longitude":[-7.], "latitude":[5.], "B2":[.2], "NDVI":[.5], "sensitivity":[.9]})
    assert validate_s2_schema(df) == ["B2", "NDVI"]

def test_schema_rejects_missing_coordinates():
    with pytest.raises(ValueError, match="missing required"):
        validate_s2_schema(pd.DataFrame({"agbd":[1.], "agbd_se":[.1], "B2":[.2]}))
