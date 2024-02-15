import random

import numpy as np

from ship_model_lib.propulsor import PropulsorDataScalar, PropulsorDataBseries
import pytest

from ship_model_lib.ship_model import HullOperatingPoint


@pytest.fixture
def propulsor_b_series() -> PropulsorDataBseries:
    return PropulsorDataBseries(
        pd_pitch_diameter_ratio=1.2,
        ear_blade_area_ratio=0.8,
        dp_diameter_propeller_m=4.5,
        z_blade_number=5,
    )


@pytest.fixture
def propulsor_scalar() -> PropulsorDataScalar:
    return PropulsorDataScalar(efficiency=0.7)


def test_negative_thrust(propulsor_b_series: PropulsorDataBseries):
    speed_list = np.array([0, 14, 0])
    thrust_list = np.array([random.random(), -random.random(), -random.random()]) * 1000
    # Test with scalar value
    for vessel_speed_kn, thrust_resistance_newton in zip(speed_list, thrust_list):
        performance_data = (
            propulsor_b_series.get_propulsor_data_from_vessel_speed_thrust(
                vessel_speed_kn=vessel_speed_kn,
                thrust_resistance_newton=thrust_resistance_newton,
            )
        )
        assert np.allclose(performance_data.n_rpm, 0)
        assert np.allclose(performance_data.shaft_power_kw, 0)
        assert np.allclose(performance_data.propeller_thrust_newton, 0)

    # Test with vectors
    performance_data = propulsor_b_series.get_propulsor_data_from_vessel_speed_thrust(
        vessel_speed_kn=speed_list, thrust_resistance_newton=thrust_list
    )
    assert np.allclose(performance_data.n_rpm, 0)
    assert np.allclose(performance_data.j, 0)
    assert np.allclose(performance_data.shaft_power_kw, 0)
    assert np.allclose(performance_data.propeller_thrust_newton, 0)


def test_propulsor_data_scalar():
    """Test the propulsor data scalar class"""
    efficiency = random.random()
    vessel_speed_kn = 14
    thrust = 1000
    hull_operating_point = HullOperatingPoint(
        vessel_speed_kn=vessel_speed_kn,
        calm_water_resistance_newton=thrust,
        added_resistance_wind_newton=0,
        added_resistance_wave_newton=0,
    )
    propulsor = PropulsorDataScalar(efficiency=efficiency)
    propeller_operating_point = propulsor.get_propulsor_data_from_vessel_speed_thrust(
        vessel_speed_kn=vessel_speed_kn,
        thrust_resistance_newton=thrust,
    )
    assert np.allclose(
        propeller_operating_point.shaft_power_kw,
        hull_operating_point.total_towing_power_kw / efficiency,
    )
