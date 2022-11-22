from dataclasses import dataclass
import numpy as np
import xarray as xr


import ggpymanager.utils


def concatenate_indices(ind_a, ind_b):
    """
    Concatenate two indices from sensor configuration.

    Examples
    --------

    >>> index = concatenate_indices(sensors_co2.get_index(), sensors_co.get_index())

    Parameters
    ----------
    ind_a : tuple of arrays
        Three arrays with the x, y, and z coordinates for sensor positions.
    ind_b : tuple of arrays
        Three arrays with the x, y, and z coordinates for sensor positions.

    Returns
    -------
    new_ind : tuple of arrays
        New concatenated index of positions.
    """
    new_ind = tuple()
    for a, b in zip(ind_a, ind_b):
        new_ind += (np.concatenate([a,b]),)
    return new_ind


def var_of_sum(covariance):
    """
    Compute the variance of a sum of random variables given the covariance matrix of the
    variables.

    Parameters
    ----------
    covariance : 2-d array
        Covariance matrix of the random variables.

    Returns
    -------
    var : float
        Variance of the summation over the random variable.
    """
    var = 0
    for i in range(covariance.shape[0]):
        var += covariance[i, i]
        for j in range(i+1, covariance.shape[0]):
            var += 2 * covariance[i, j]
    return var


stacked_dims = ["measurement", "state", "measurement_2", "state_2"]
dim_pairs = [
    ["sensor", "time_measurement"],
    ["source_group", "time_state"],
    ["sensor_2", "time_measurement_2"],
    ["source_group_2", "time_state_2"],
]


def convert_to_ppm(obj):
    # Convert from mu g/m^3 to ppm
    # At 273.15 K and 1 bar
    Vm = 22.71108  # standard molar volume of ideal gas [l/mol]
    m = 44.01  # molecular weight mass [g/mol]
    cubic_meter_to_liter = 1000
    mu_g_to_g = 1e-6
    to_ppm = 1e6
    factor = Vm / m / cubic_meter_to_liter
    return obj * factor


def stack_xr(xr_data_array):
    for stacked_dim, dim_pair in zip(stacked_dims, dim_pairs):
        if dim_pair[0] in xr_data_array.dims and dim_pair[1] in xr_data_array.dims:
            xr_data_array = xr_data_array.stack({stacked_dim: dim_pair})
    return xr_data_array


def unstack_xr(xr_data_array):
    for stacked_dim in stacked_dims:
        if stacked_dim in xr_data_array.dims:
            xr_data_array = xr_data_array.unstack(stacked_dim)
    return xr_data_array


# Create covariance matrices

# Temporal correlation with correlation length tau
def compute_corr(delta_t, tau_h, tau_d, delta_l=0, tau_l=1):
    # delta_h = np.min([delta_t % 24, 24 - delta_t % 24])
    delta_h = delta_t
    # Leave out daily correlation
    # delta_d = np.rint(delta_t / 24)
    delta_d = 0.
    corr_factor = delta_h / tau_h + delta_d / tau_d + delta_l / tau_l
    if (corr_factor) <= 6:
        return np.exp(-corr_factor)
    else:
        return 0.0


def compute_prior_covariance(xr_prior_var, tau_h, tau_d, tau_l=None, point_list=None):
    """
    Compute the prior covariance matrix from the variance of the prior and correlation
    lenghts.

    Parameters
    ----------
    xr_prior_var : xarray.DataArray
        Variance of the prior with dims=["source_group", "time_state"].
    tau_h : float
        Correlation length for hours.
    tau_d : float
        Correlation length for days.
    tau_l : float
        Correlation length for the mean distance between sources.
    point_list : list of shapely.Points
        List of the position of all sources.

    Returns
    -------
    prior_covariance : array
        Array with 2-dim.
    """
    n_source_group = len(xr_prior_var.coords["source_group"])
    n_time_state = len(xr_prior_var.coords["time_state"])

    correlation_time = np.eye(n_time_state)
    for i in range(1, n_time_state):
        # print("{}\r".format(i/n_meteo), end="")
        corr = compute_corr(
            delta_t=i,
            tau_h=tau_h,
            tau_d=tau_d,
            delta_l=0.0,
            tau_l=1.0,
        )
        if corr > 0:
            correlation_time += corr * (
                np.diag(np.ones(n_time_state - i), i)
                + np.diag(np.ones(n_time_state - i), -i)
            )

    correlation_space = np.eye(n_source_group)
    if tau_l is not None:
        for i in range(n_source_group):
            for j in range(n_source_group):
                if i != j:
                    corr = compute_corr(
                        delta_t=i,
                        tau_h=tau_h,
                        tau_d=tau_d,
                        delta_l=point_list[i].distance(point_list[j]),
                        tau_l=tau_l,
                    )
                    correlation_space[i, j] = corr
                    correlation_space[j, i] = corr

    correlation_time = xr.DataArray(
        data=correlation_time, dims=["time_state", "time_state_2"]
    )
    correlation_space = xr.DataArray(
        data=correlation_space, dims=["source_group", "source_group_2"]
    )
    prior_correlation = correlation_time * correlation_space
    prior_std = np.sqrt(xr_prior_var)
    prior_std_2 = prior_std.rename(
        {"source_group": "source_group_2", "time_state": "time_state_2"}
    )
    prior_covariance = prior_std * prior_correlation * prior_std_2
    # prior_covariance = prior_covariance.stack(
    #     state=["source_group", "time_state"], state_2=["source_group_2", "time_state_2"]
    # )

    return prior_covariance


def get_K_kernel(con_path, sensors_index, n_sensors, emissions_mask, n_emissions):
    con_dict = ggpymanager.utils.read_gral_concentration(con_path)
    height_list = np.unique(sensors_index[2])
    xr_K = xr.DataArray(
        data=np.zeros((n_sensors, n_emissions)),
        dims=["sensor", "source_group"],
        coords={
            "source_group": emissions_mask[emissions_mask == True].coords[
                "source_group"
            ]
        },
    )
    for key, val in con_dict.items():
        # Height needed for sensors
        height = int(key[0]) - 1
        if height in height_list:
            source_group = int(key[1:])
            # Source group id in emissions
            if source_group < len(emissions_mask):
                if emissions_mask.sel(source_group=source_group):
                    # Array not empty
                    if val.size > 1:
                        index = sensors_index[2] == height
                        xr_K.loc[index, source_group] = val[
                            (
                                sensors_index[0][index],
                                sensors_index[1][index],
                            )
                        ]
    return xr_K
