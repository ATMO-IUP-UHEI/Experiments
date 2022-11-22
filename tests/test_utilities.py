import numpy as np
import xarray as xr

from context import utilities as utils

tol = 1e-6


def test_utilities():
    pseudo_data = np.ones((2, 2)) * 2
    xr_state = xr.DataArray(data=pseudo_data, dims=["source_group", "time_state"])
    xr_state_2 = xr.DataArray(data=pseudo_data, dims=["source_group_2", "time_state_2"])
    xr_measurement = xr.DataArray(data=pseudo_data, dims=["sensor", "time_measurement"])
    xr_measurement_2 = xr.DataArray(
        data=pseudo_data, dims=["sensor_2", "time_measurement_2"]
    )

    # Test for only one stacked dim
    assert len(utils.stack_xr(xr_state).dims) == 1
    assert len(utils.stack_xr(xr_state_2).dims) == 1
    assert len(utils.stack_xr(xr_measurement).dims) == 1
    assert len(utils.stack_xr(xr_measurement_2).dims) == 1
    # Test for right name of stacked dim
    assert utils.stack_xr(xr_state).dims[0] == "state"
    assert utils.stack_xr(xr_state_2).dims[0] == "state_2"
    assert utils.stack_xr(xr_measurement).dims[0] == "measurement"
    assert utils.stack_xr(xr_measurement_2).dims[0] == "measurement_2"

    assert utils.unstack_xr(utils.stack_xr(xr_state)).dims == xr_state.dims
    assert utils.unstack_xr(utils.stack_xr(xr_state_2)).dims == xr_state_2.dims
    assert utils.unstack_xr(utils.stack_xr(xr_measurement)).dims == xr_measurement.dims
    assert (
        utils.unstack_xr(utils.stack_xr(xr_measurement_2)).dims == xr_measurement_2.dims
    )

    assert utils.compute_corr(
        delta_t=1.0,
        tau_h=1.0,
        tau_d=1.0,
    ) == np.exp(-1)
    assert utils.compute_corr(
        delta_t=24.0,
        tau_h=1.0,
        tau_d=1.0,
    ) == 0.
    assert (
        utils.compute_corr(
            delta_t=12.0,
            tau_h=1.0,
            tau_d=1.0,
        )
        == 0.0
    )

    cov = utils.compute_prior_covariance(
        xr_prior_var=xr_state,
        tau_h=0.001,
        tau_d=0.001,
    )
    assert utils.stack_xr(cov).dims == utils.stack_xr(xr_state * xr_state_2).dims
    for id_source_group in range(len(cov["source_group"])):
        for id_source_group_2 in range(len(cov["source_group_2"])):
            for id_time_state in range(len(cov["time_state"])):
                for id_time_state_2 in range(len(cov["time_state_2"])):
                    if (
                        id_source_group == id_source_group_2
                        and id_time_state == id_time_state_2
                    ):
                        print(
                            cov.isel(
                                source_group=id_source_group,
                                source_group_2=id_source_group_2,
                                time_state=id_time_state,
                                time_state_2=id_time_state_2,
                            )
                        )
                        print(
                            xr_state.isel(
                                source_group=id_source_group,
                                time_state=id_time_state,
                            )
                        )
                        assert (
                            np.abs(
                                cov.isel(
                                    source_group=id_source_group,
                                    source_group_2=id_source_group_2,
                                    time_state=id_time_state,
                                    time_state_2=id_time_state_2,
                                )
                                - xr_state.isel(
                                    source_group=id_source_group,
                                    time_state=id_time_state,
                                )
                            )
                            < tol
                        )
                    else:
                        assert (
                            np.abs(
                                cov.isel(
                                    source_group=id_source_group,
                                    source_group_2=id_source_group_2,
                                    time_state=id_time_state,
                                    time_state_2=id_time_state_2,
                                )
                            )
                            < tol
                        )
