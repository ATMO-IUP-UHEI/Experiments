from dataclasses import dataclass
import numpy as np
import xarray as xr


@dataclass
class CONSTANTS:
    TEST_INT = 3

stacked_dims = ["measurement", "state", "measurement_2", "state_2"]
dim_pairs = [
    ["sensor", "time_measurement"],
    ["source_group", "time_state"],
    ["sensor_2", "time_measurement_2"],
    ["source_group_2", "time_state_2"],
    ]
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
    corr_factor = (delta_t % 24) / tau_h + (delta_t // 24) / tau_d + delta_l / tau_l
    if (corr_factor) <= 3:
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
