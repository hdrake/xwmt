import warnings

import gsw
import numpy as np
import xarray as xr
import xgcm

def hlamdot_from_Jlam(grid, Jlam, dim):
    """
    Calculation of hlamdot (cell-depth integral of scalar tendency)
    from interfacial fluxes
    """
    # For convergence, need to reverse the sign
    dJlam = -grid.diff(Jlam, dim)
    if "Z_metrics" in list(vars(grid)):
        h = grid.Z_metrics["center"]
        h = h.where(h!=0.)
    else:
        h = grid.get_metric(dJlam, "Z")
    lamdot = dJlam/h
    hlamdot = h.fillna(0.)*lamdot.fillna(0.)
    return hlamdot

def calc_hlamdot_tendency(grid, datadict):
    """
    Wrapper functions to determine h times lambda_dot (vertically extensive tendency)
    """

    if "layer_integrated_tendency" in datadict:
        return datadict["layer_integrated_tendency"]
    
    elif "interfacial_flux" in datadict:
        return hlamdot_from_Jlam(
            grid,
            datadict["interfacial_flux"],
            "Z"
        )