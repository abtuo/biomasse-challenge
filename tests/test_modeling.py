import numpy as np

from biomass_mapping.modeling import regression_metrics


def test_regression_metrics_include_bias_and_nrmse():
    out = regression_metrics(np.array([1., 2., 3.]), np.array([2., 2., 2.]))
    assert {"rmse", "mae", "r2", "bias", "nrmse_mean"} <= set(out)
    assert out["bias"] == 0.0
