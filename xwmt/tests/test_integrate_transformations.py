import pytest
import numpy as np
import xarray as xr
import xbudget
import xgcm
import xwmt

ds = xr.open_dataset("xwmb_test_data_Baltic_3d.20230830.nc", decode_timedelta=False).isel(time=0)
coords = {
    'X': {'center': 'xh', 'outer': 'xq'},
    'Y': {'center': 'yh', 'outer': 'yq'},
    'Z': {'center': 'zl', 'outer': 'zi'},
}
metrics = {
    ('X','Y'): "areacello",
}
grid = xgcm.Grid(ds, coords=coords, metrics=metrics, periodic=None, autoparse_metadata=False)

budgets_dict = xbudget.load_preset_budget(model="MOM6")
# The test data set does not include sea ice melt diagnostics
del budgets_dict["mass"]["rhs"]["sum"]["surface_exchange_flux"]["sum"]["sea_ice_melt"]
xbudget.collect_budgets(grid, budgets_dict)
simple_budgets = xbudget.aggregate(budgets_dict)

## Default parameters except that we group all processes together
kwargs = {'group_processes': True}

# heat
def test_functional_3d_theta():
    answer_dict = {
        "xgcm": np.array([5.89988332e+08, 1.71073947e+08, 1.60436210e+09, 6.14312533e+08]),
        "xhistogram": np.array([6.11109007e+08, 6.16747397e+08, 1.16849868e+09, 4.80601100e+08])
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "heat",
            bins = np.linspace(0., 4., 5),
            **kwargs
            )['material_transformation']
        assert np.all(np.isclose(
            total_wmt.values,
            answer_dict[method]
        ))


# salt
def test_functional_3d_salt():
    answer_dict = {
        "xgcm": np.array([-6.92695360e+07, -2.92969936e+08, -5.65359683e+07,  1.31220262e+08]),
        "xhistogram": np.array([-7.09139741e+07, -2.94351853e+08, -2.40161246e+07,  6.65517384e+07])
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "salt",
            bins = np.linspace(5., 9., 5),
            **kwargs
            )['material_transformation']
        assert np.all(np.isclose(
            total_wmt.values,
            answer_dict[method]
        ))
# sigma2
def test_functional_3d_sigma2():
    answer_dict = {
        "xgcm": np.array([-3.89013506e+08,  1.11459836e+08,  3.97737451e+07,  7.12295765e+06]),
        "xhistogram": np.array([-3.76907156e+08,  9.13575231e+07,  4.22416664e+07,  6.72588395e+06])
    }
    for method in ["xgcm", "xhistogram"]:
        wmt = xwmt.WaterMassTransformations(grid, simple_budgets, method=method)
        total_wmt = wmt.integrate_transformations(
            "sigma2",
            bins=np.linspace(15., 19., 5),
            **kwargs
            )['material_transformation']
        assert np.all(np.isclose(
            total_wmt.values,
            answer_dict[method]
        ))
