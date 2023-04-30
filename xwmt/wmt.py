import numpy as np
import xarray as xr
from xhistogram.xarray import histogram
import gsw
import warnings

from xwmt.compute import (
    Jlammass_from_Qm_lm_l,
    calc_hlamdot_tendency,
    expand_surface_to_3d,
    get_xgcm_grid_vertical,
    hlamdot_from_Jlam,
    hlamdot_from_Ldot_hlamdotmass,
    bin_define,
    bin_percentile,
)

class WaterMassTransformations:
    """
    A class object with multiple methods to do full 3d watermass transformation analysis.
    """

    def __init__(self, ds, cp=3992.0, rho_ref=1035.0, alpha=None, beta=None, teos10=True):
        """
        Create a new watermass transformation object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant tendencies and/or surface fluxes along with grid information.
        cp : float, optional
            Specify value for the specific heat capacity (in J/kg/K). cp=3992.0 by default.
        rho_ref : float, optional
            Specify value for the reference seawater density (in kg/m^3). rho_ref=1035.0 by default.
        alpha : float, optional
            Specify value for the thermal expansion coefficient (in 1/K). alpha=None by default.
            If alpha is not given (i.e., alpha=None), it is derived from salinty and temperature fields using `gsw_alpha`.
        beta : float, optional
            Specify value for the haline contraction coefficient (in kg/g). beta=None by default.
            If beta is not given (i.e., beta=None), it is derived from salinty and temperature fields using `gsw_beta`.
        teos10 : boolean, optional
            Use Thermodynamic Equation Of Seawater - 2010 (TEOS-10). True by default.
        """

        self.ds = ds.copy()
        self.xgrid = get_xgcm_grid_vertical(self.ds, periodic=False)
        self.cp = cp
        self.rho_ref = rho_ref
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        self.teos10 = teos10

    # Set of terms for (1) heat and (2) salt fluxes
    # Use processes as default, fluxes when surface=True
    terms_dict = {"heat": "thetao", "salt": "so"}

    processes_heat_dict = {
        "Eulerian_tendency": "opottemptend",
        "horizontal_advection": "T_advection_xy",
        "vertical_advection": "Th_tendency_vert_remap",
        "boundary_forcing": "boundary_forcing_heat_tendency",
        "vertical_diffusion": "opottempdiff",
        "neutral_diffusion": "opottemppmdiff",
        "frazil_ice": "frazil_heat_tendency",
        "geothermal": "internal_heat_heat_tendency",
    }

    processes_salt_dict = {
        "Eulerian_tendency": "osalttend",
        "horizontal_advection": "S_advection_xy",
        "vertical_advection": "Sh_tendency_vert_remap",
        "boundary_forcing": "boundary_forcing_salt_tendency",
        "vertical_diffusion": "osaltdiff",
        "neutral_diffusion": "osaltpmdiff",
        "frazil_ice": None,
        "geothermal": None,
    }

    lambdas_dict = {
        "heat": ["theta"],
        "salt": ["salt"],
        "density": ["sigma0", "sigma1", "sigma2", "sigma3", "sigma4"],
    }

    def lambdas(self, lambda_name=None):
        if lambda_name is None:
            return sum(self.lambdas_dict.values(), [])
        else:
            return self.lambdas_dict.get(lambda_name, None)

    # Helper function to get variable name for given process term
    def process(self, tendency, term):
        # Organize by scalar and tendency
        if tendency == "heat":
            termcode = self.processes_heat_dict.get(term, None)
        elif tendency == "salt":
            termcode = self.processes_salt_dict.get(term, None)
        else:
            warnings.warn(f"Tendency {tendency} is not defined")
            return
        tendcode = self.terms_dict.get(tendency, None)
        return (tendcode, termcode)

    # Helper function to list available processes
    def processes(self, check=True):
        processes = self.processes_heat_dict.keys() | self.processes_salt_dict.keys()
        if check:
            _processes = []
            for process in processes:
                p1 = self.processes_salt_dict.get(process, None)
                p2 = self.processes_heat_dict.get(process, None)
                if ((p1 is None) or (p1 is not None and p1 in self.ds)) and (
                    (p2 is None) or (p2 is not None and p2 in self.ds)
                ):
                    _processes.append(process)
            return _processes
        else:
            return processes

    def datadict(self, tendency, term):
        (tendcode, termcode) = self.process(tendency, term)
        # tendcode: tendency form (heat or salt)
        # termcode: process term (e.g., boundary_forcing)
        if termcode is None or termcode not in self.ds:
            return

        if tendency == "salt":
            # Multiply salt tendency by 1000 to convert to g/m^2/s
            tend_arr = self.ds[termcode] * 1000
        else:
            tend_arr = self.ds[termcode]

        if term == "boundary_forcing":
            if termcode == "boundary_forcing_heat_tendency":
                # Need to multiply mass flux by cp to convert to energy flux (convert to W/m^2/degC)
                flux = (
                    expand_surface_to_3d(self.ds["wfo"], self.ds["lev_outer"]) * self.cp
                )
                scalar_in_mass = expand_surface_to_3d(
                    self.ds["tos"], self.ds["lev_outer"]
                )
            elif termcode == "boundary_forcing_salt_tendency":
                flux = expand_surface_to_3d(self.ds["wfo"], self.ds["lev_outer"])
                scalar_in_mass = expand_surface_to_3d(
                    xr.zeros_like(self.ds["sos"]), self.ds["lev_outer"]
                )
            else:
                raise ValueError(f"termcode {termcode} not yet supported.")
            return {
                "scalar": {"array": self.ds[tendcode]},
                "tendency": {"array": tend_arr, "extensive": True, "boundary": True},
                "boundary": {
                    "flux": flux,
                    "mass": True,
                    "scalar_in_mass": scalar_in_mass,
                },
            }
        else:
            return {
                "scalar": {"array": self.ds[tendcode]},
                "tendency": {"array": tend_arr, "extensive": True, "boundary": False},
            }

    def get_density(self, density_str=None):

        # Variables needed to calculate alpha, beta and density
        if (
            "alpha" not in vars(self) or "beta" not in vars(self) or self.teos10
        ) and "p" not in vars(self):
            self.p = xr.apply_ufunc(
                gsw.p_from_z, -self.ds["lev"], self.ds["lat"], 0, 0, dask="parallelized"
            )
        if self.teos10 and "sa" not in vars(self):
            self.sa = xr.apply_ufunc(
                gsw.SA_from_SP,
                self.ds["so"],
                self.p,
                self.ds["lon"],
                self.ds["lat"],
                dask="parallelized",
            )
        if self.teos10 and "ct" not in vars(self):
            self.ct = xr.apply_ufunc(
                gsw.CT_from_t, self.sa, self.ds["thetao"], self.p, dask="parallelized"
            )
        if not self.teos10 and ("sa" not in vars(self) or "ct" not in vars(self)):
            self.sa = self.ds.so
            self.ct = self.ds.thetao

        # Calculate thermal expansion coefficient alpha (1/K)
        if "alpha" not in vars(self):
            if "alpha" in self.ds:
                self.alpha = self.ds.alpha
            else:
                self.alpha = xr.apply_ufunc(
                    gsw.alpha, self.sa, self.ct, self.p, dask="parallelized"
                )

        # Calculate the haline contraction coefficient beta (kg/g)
        if "beta" not in vars(self):
            if "beta" in self.ds:
                self.beta = self.ds.beta
            else:
                self.beta = xr.apply_ufunc(
                    gsw.beta, self.sa, self.ct, self.p, dask="parallelized"
                )

        # Calculate potential density (kg/m^3)
        if density_str not in self.ds:
            if density_str == "sigma0":
                density = xr.apply_ufunc(
                    gsw.sigma0, self.sa, self.ct, dask="parallelized"
                )
            elif density_str == "sigma1":
                density = xr.apply_ufunc(
                    gsw.sigma1, self.sa, self.ct, dask="parallelized"
                )
            elif density_str == "sigma2":
                density = xr.apply_ufunc(
                    gsw.sigma2, self.sa, self.ct, dask="parallelized"
                )
            elif density_str == "sigma3":
                density = xr.apply_ufunc(
                    gsw.sigma3, self.sa, self.ct, dask="parallelized"
                )
            elif density_str == "sigma4":
                density = xr.apply_ufunc(
                    gsw.sigma4, self.sa, self.ct, dask="parallelized"
                )
            else:
                return self.alpha, self.beta, None
        else:
            return self.alpha, self.beta, self.ds[density_str]

        return self.alpha, self.beta, density.rename(density_str)

    def rho_tend(self, term):
        """
        Calculate the tendency of the locally-referenced potential density.
        """

        if "alpha" in vars(self) and "beta" in vars(self):
            alpha, beta = self.alpha, self.beta
        else:
            (alpha, beta, _) = self.get_density()

        # Either heat or salt tendency/flux may not be used
        rho_tend_heat, rho_tend_salt = None, None

        datadict = self.datadict("heat", term)
        if datadict is not None:
            heat_tend = calc_hlamdot_tendency(self.xgrid, self.datadict("heat", term))
            # Density tendency due to heat flux (kg/s/m^2)
            rho_tend_heat = -(alpha / self.cp) * heat_tend

        datadict = self.datadict("salt", term)
        if datadict is not None:
            salt_tend = calc_hlamdot_tendency(self.xgrid, self.datadict("salt", term))
            # Density tendency due to salt/salinity (kg/s/m^2)
            rho_tend_salt = beta * salt_tend

        return rho_tend_heat, rho_tend_salt

    def calc_hlamdot_and_lambda(self, lambda_name, term):
        """
        Get layer-integrated extensive tracer tendencies (* m/s) and corresponding scalar field of lambda
        lambda_name: str
            Specifies lambda
        term: str
            Specifies process term
        """

        # Get layer-integrated potential temperature tendency from tendency of heat (in W/m^2), lambda = theta
        if lambda_name == "theta":
            datadict = self.datadict("heat", term)
            if datadict is not None:
                hlamdot = calc_hlamdot_tendency(self.xgrid, datadict) / (self.rho_ref * self.cp)
                lam = datadict["scalar"]["array"]

        # Get layer-integrated salinity tendency tendency from tendency of salt (in g/s/m^2), lambda = salt
        elif lambda_name == "salt":
            datadict = self.datadict("salt", term)
            if datadict is not None:
                hlamdot = calc_hlamdot_tendency(self.xgrid, datadict) / self.rho_ref
                # TODO: Accurately define salinity field (What does this mean? - HFD)
                lam = datadict["scalar"]["array"]

        # Get layer-integrated potential density tendencies (in kg/s/m^2) from heat and salt, lambda = density
        # Here we want to output 2 transformation rates:
        # (1) transformation due to heat tend, (2) transformation due to salt tend
        elif lambda_name in self.lambdas("density"):
            rhos = self.rho_tend(term)
            hlamdot = {}
            for idx, tend in enumerate(self.terms_dict.keys()):
                hlamdot[tend] = rhos[idx]
            lam = self.get_density(lambda_name)[2]
            
        else:
            raise ValueError(f"{lambda_name} is not a supported lambda.")
            
        return hlamdot, lam

    def transform_hlamdot(self, lambda_name, term, bins=None):
        """
        Transform to lambda space
        """

        hlamdot, lam = self.calc_hlamdot_and_lambda(lambda_name, term)
        if hlamdot is None:
            return

        if bins is None:
            bins = bin_percentile(
                lam
            )  # automatically find the right range based on the distribution in l

        # Interpolate lambda to the cell interfaces
        lam_i = (
            self.xgrid.interp(lam, "Z", boundary="extend")
            .chunk({"lev_outer": -1})
            .rename(lam.name)
        )

        if lambda_name in self.lambdas("density"):
            hlamdot_transformed = []
            for tend in self.terms_dict.keys():
                (tendcode, termcode) = self.process(tend, term)
                if hlamdot[tend] is not None:
                    hlamdot_transformed.append(
                        (
                            self.xgrid.transform(
                                hlamdot[tend],
                                "Z",
                                target=bins,
                                target_data=lam_i,
                                method="conservative",
                            )
                            / np.diff(bins)
                        ).rename(termcode)
                    )
            hlamdot_transformed = xr.merge(hlamdot_transformed)
        else:
            (tendcode, termcode) = self.process(
                "salt" if lambda_name == "salt" else "heat", term
            )
            hlamdot_transformed = (
                self.xgrid.transform(
                    hlamdot, "Z", target=bins, target_data=lam_i, method="conservative"
                )
                / np.diff(bins)
            ).rename(termcode)
        return hlamdot_transformed

    def transform_hlamdot_and_integrate(self, lambda_name, term=None, bins=None):
        """
        Water mass transformation (G)
        """

        # If term is not given, use all available process terms
        if term is None:
            wmts = []
            for term in self.processes(False):
                wmt = self.transform_hlamdot_and_integrate(lambda_name, term, bins)
                if wmt is not None:
                    wmts.append(wmt)
            return xr.merge(wmts)

        hlamdot_transformed = self.transform_hlamdot(lambda_name, term, bins=bins)
        if hlamdot_transformed is not None and len(hlamdot_transformed): # What is the point of this?
            wmt = (hlamdot_transformed * self.ds["areacello"]).sum(["x", "y"])
            # rename dataarray only (not dataset)
            if isinstance(wmt, xr.DataArray):
                return wmt.rename(hlamdot_transformed.name)
            return wmt
        return hlamdot_transformed

    ### Helper function to groups terms based on density components (sum_components)
    ### and physical processes (group_processes)
    # Calculate the sum of grouped terms
    def _sum_terms(self, ds, newterm, terms):
        das = []
        for term in terms:
            if term in ds:
                das.append(ds[term])
        if len(das):
            ds[newterm] = sum(das)

    def _group_processes(self, hlamdot):
        if hlamdot is None:
            return
        for component in ["heat", "salt"]:
            process_dict = getattr(self, f"processes_{component}_dict")
            self._sum_terms(
                hlamdot,
                f"external_forcing_{component}",
                [
                    process_dict["boundary_forcing"],
                    process_dict["frazil_ice"],
                    process_dict["geothermal"],
                ],
            )
            self._sum_terms(
                hlamdot,
                f"diffusion_{component}",
                [
                    process_dict["vertical_diffusion"],
                    process_dict["neutral_diffusion"]
                ]
            )
            self._sum_terms(
                hlamdot,
                f"advection_{component}",
                [
                    process_dict["horizontal_advection"],
                    process_dict["vertical_advection"]
                ]
            )
            self._sum_terms(
                hlamdot,
                f"diabatic_forcing_{component}",
                [
                    f"external_forcing_{component}",
                    f"diffusion_{component}"
                ]
            )
            self._sum_terms(
                hlamdot,
                f"total_tendency_{component}",
                [
                    f"advection_{component}",
                    f"diabatic_forcing_{component}"
                ]
            )
        return hlamdot

    def _sum_components(self, hlamdot, group_processes = False):
        if hlamdot is None:
            return
        
        for proc in self.processes():
            self._sum_terms(
                hlamdot,
                proc,
                [
                    getattr(self, f"processes_{component}_dict")[proc]
                    for component in ["heat", "salt"]
                ]
            )
        if group_processes:
            for proc in ["external_forcing", "diffusion", "advection", "diabatic_forcing", "total_tendency"]:
                self._sum_terms(hlamdot, proc, [f"{proc}_{component}" for component in ["heat", "salt"]])
        return hlamdot

    def map_transformations(self, lambda_name, term=None, sum_components=True, group_processes=False, **kwargs):
        """
        Wrapper function for transform_hlamdot() to group terms based on tendency terms (heat, salt) and processes.
        """

        # If term is not given, use all available process terms
        if term is None:
            Fs = []
            for term in self.processes(False):
                _F = self.F(lambda_name, term, sum_components=False, group_processes=False, **kwargs)
                if _F is not None:
                    Fs.append(_F)
            F = xr.merge(Fs)
        else:
            # If term is given
            F = self.transform_hlamdot(lambda_name, term, **kwargs)
            if isinstance(F, xr.DataArray):
                F = F.to_dataset()

        if group_processes:
            F = self._group_processes(F)
        if sum_components:
            F = self._sum_components(F, group_processes=group_processes)

        if isinstance(F, xr.Dataset) and len(F) == 1:
            return F[list(F.data_vars)[0]]
        else:
            return F

    def integrate_transformations(self, lambda_name, *args, **kwargs):
        """
        Water mass transformation (G)

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'theta', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str, optional
            Specifies process term (e.g., 'boundary_forcing', 'vertical_diffusion', etc.). Use `processes()` to list all available terms.
        bins : array like, optional
            np.array with lambda values specifying the edges for each bin. If not specidied, array will be automatically derived from
            the scalar field of lambda (e.g., temperature).
        sum_components : boolean, optional
            Specify whether heat and salt tendencies are summed together (True) or kept separated (False). True by default.
        group_processes: boolean, optional
            Specify whether process terms are summed to categories forcing and diffusion. False by default.

        Returns
        -------
        G : {xarray.DataArray, xarray.Dataset}
            The water mass transformation along lamba for each time. G is xarray.DataArray when term is specified and sum_components=True.
            G is xarray.DataSet when multiple terms are included (term=None) or sum_components=False.
        """

        # Extract default function args
        group_processes = kwargs.pop("group_processes", False)
        sum_components = kwargs.pop("sum_components", True)
        # call the base function
        G = self.transform_hlamdot_and_integrate(lambda_name, *args, **kwargs)

        # process this function arguments
        if group_processes:
            G = self._group_processes(G)
        if sum_components:
            G = self._sum_components(G, group_processes=group_processes)

        if isinstance(G, xr.Dataset) and len(G) == 1:
            return G[list(G.data_vars)[0]]
        else:
            return G

    def isosurface_mean(self, *args, ti=None, tf=None, dl=0.1, **kwargs):
        """
        Mean transformation across lambda isosurface(s).

        Parameters
        ----------
        lambda_name : str
            Specifies lambda (e.g., 'theta', 'salt', 'sigma0', etc.). Use `lambdas()` for a list of available lambdas.
        term : str, optional
            Specifies process term (e.g., 'boundary_forcing', 'vertical_diffusion', etc.). Use `processes()` to list all available terms.
        val : float or ndarray
            Value(s) of lambda for which isosurface(s) is/are defined
        ti : str
            Starting date. ti=None by default.
        tf : str
            End date. tf=None by default.
        dl : float
            Width of lamba bin (delta) for which isosurface(s) is/are defined.
        sum_components : boolean, optional
            Specify whether heat and salt tendencies are summed together (True) or kept separated (False). True by default.
        group_processes : boolean, optional
            Specify whether process terms are summed to categories forcing and diffusion. False by default.

        Returns
        -------
        F_mean : {xarray.DataArray, xarray.Dataset}
            Spatial field of mean transformation at a given (set of) lambda value(s). F_mean is xarray.DataArray when term is specified and sum_components=True.
            F_mean is xarray.DataSet when multiple terms are included (term=None) or sum_components=False.
        """

        if len(args) == 3:
            (lambda_name, term, val) = args
        elif len(args) == 2:
            (lambda_name, val) = args
            term = None
        else:
            warnings.warn(
                "isosurface_mean() requires arguments (lambda_name, term, val,...) or (lambda_name, val,...)"
            )
            return

        if lambda_name not in self.lambdas("density"):
            tendency = [k for k, v in self.lambdas_dict.items() if v[0] == lambda_name]
            if len(tendency) == 1:
                tendcode = self.terms_dict.get(tendency[0], None)
            else:
                warnings.warn("Tendency is not defined")
                return
        else:
            tendcode = lambda_name

        # Define bins based on val
        kwargs["bins"] = bin_define(np.min(val) - dl, np.max(val) + dl, dl)

        # Calculate spatiotemporal field of transformation
        F = self.F(lambda_name, term, **kwargs)
        # TODO: Preferred method should be ndays_standard if calendar type is 'noleap'. Thus, avoiding to load the full time array
        if (
            "calendar_type" in self.ds.time.attrs
            and self.ds.time.attrs["calendar_type"].lower() == "noleap"
        ):
            # Number of days in each month
            n_years = len(np.unique(F.time.dt.year))
            # Monthly data
            dm = np.diff(F.indexes["time"].month)
            udm = np.unique([m + 12 if m == -11 else m for m in dm])
            if np.array_equal(udm, [1]):
                ndays_standard = np.array(
                    [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                )
                assert np.sum(ndays_standard) == 365
                dt = xr.DataArray(
                    ndays_standard[F.time.dt.month.values - 1],
                    coords=[self.ds.time],
                    dims=["time"],
                    name="days per month",
                )
            # Annual data
            dy = np.diff(F.indexes["time"].year)
            udy = np.unique(dy)
            if np.array_equal(udy, [1]):
                dt = xr.DataArray(
                    np.tile(365, n_years),
                    coords=[self.ds.time],
                    dims=["time"],
                    name="days per year",
                )
        elif "time_bounds" in self.ds:
            # Extract intervals (units are in ns)
            deltat = self.ds.time_bounds[:, 1].values - self.ds.time_bounds[:, 0].values
            # Convert intervals to days
            dt = xr.DataArray(
                deltat, coords=[self.ds.time], dims=["time"], name="days per month"
            ) / np.timedelta64(1, "D")
        elif "time_bnds" in self.ds:
            # Extract intervals (units are in ns)
            deltat = self.ds.time_bnds[:, 1].values - self.ds.time_bnds[:, 0].values
            # Convert intervals to days
            dt = xr.DataArray(
                deltat, coords=[self.ds.time], dims=["time"], name="days per month"
            ) / np.timedelta64(1, "D")
        else:
            # TODO: Create dt with ndays_standard but output warning that calendar_type is not specified.
            # warnings.warn('Unsupported calendar type')
            print("Unsupported calendar type", self.ds.time.attrs)
            return

        # Convert to dask array for lazy calculations
        dt = dt.chunk(1)
        F_mean = (
            F.sel({tendcode: val}, method="nearest").sel(time=slice(ti, tf))
            * dt.sel(time=slice(ti, tf))
        ).sum("time") / dt.sel(time=slice(ti, tf)).sum("time")
        return F_mean
