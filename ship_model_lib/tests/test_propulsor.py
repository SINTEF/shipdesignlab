import pytest
import random
import numpy as np
import os

from ship_model_lib.propulsor import PropulsorDataScalar, PropulsorDataBseries
from ship_model_lib.ship_model import HullOperatingPoint

from ship_model_lib.propulsor import (
    WakeFractionThrustDeductionFactorPoint,
    OpenWaterPropellerCurvePoint,
)

from ship_model_lib.propulsor import PropulsorDataOpenWater, kn_to_m_per_s
from dataclasses import fields
from ship_model_lib.propulsor import PropulsorDataBseries, ReCorrection
from scipy import interpolate


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

def test_get_propulsor_data_open_water_from_vessel_speed_rps():

    vessel_speed_kn = np.array([13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18])
    wake_fraction = np.array(
        [0.338, 0.337, 0.336, 0.335, 0.334, 0.332, 0.329, 0.328, 0.326, 0.325, 0.322]
    )
    thrust_deduction = np.array(
        [0.201, 0.205, 0.209, 0.214, 0.218, 0.22, 0.223, 0.224, 0.227, 0.229, 0.233]
    )
    wake_factor_thrust_deduction_points = [
        WakeFractionThrustDeductionFactorPoint(
            wake_fraction_factor=wake_frac,
            thrust_deduction_factor=thrust_ded,
            vessel_speed_kn=speed,
        )
        for wake_frac, thrust_ded, speed in zip(
            wake_fraction, thrust_deduction, vessel_speed_kn
        )
    ]

    j_array = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    kt_array = (
            np.array(
                [5.97, 5.54, 5.08, 4.58, 4.07, 3.57, 3.08, 2.57, 2.07, 1.58, 1.08, 0.53, 0.0]
            )
            * 1e-1
    )
    kq_array = (
            np.array(
                [8.86, 8.26, 7.72, 7.09, 6.49, 5.90, 5.31, 4.66, 4.00, 3.32, 2.62, 1.83, 0.87]
            )
            * 1e-2
    )

    propeller_curve_points = [
        OpenWaterPropellerCurvePoint(j=j, kt=kt, kq=kq)
        for j, kt, kq in zip(j_array, kt_array, kq_array)
    ]

    propulsor = PropulsorDataOpenWater(
        propeller_curve_points=propeller_curve_points,
        dp_diameter_propeller_m=4.5,
        pitch_diameter_ratio=1.2,
        wake_thrust_reduction=wake_factor_thrust_deduction_points,
    )
  #  if not os.getenv("CI"):  # Skip plotting on CI or in test context
   #     propulsor.plot_open_water_curves()


    j_value = (
            random.random() * (propulsor._j.max() - propulsor._j.min()) + propulsor._j.min()
    )
    speed_kn = random.random() * 20
    n_rps = kn_to_m_per_s(speed_kn) / (j_value * propulsor._d)
    propulsion_point = propulsor.get_propulsor_data_from_vessel_speed_rps(
        vessel_speed_kn=speed_kn, n_rps=n_rps
    )
    eff_hull = (propulsion_point.resistance_newton * speed_kn) / (
            propulsion_point.propeller_thrust_newton * propulsion_point.wake_velocity_kn
    )
    eff_open_water = (
                             propulsion_point.propeller_thrust_newton
                             * kn_to_m_per_s(propulsion_point.wake_velocity_kn)
                     ) / (propulsion_point.shaft_power_kw * 1000)
    print(propulsion_point)
    assert propulsion_point.efficiency_hull == pytest.approx(eff_hull, rel=1e-4)
    assert propulsion_point.efficiency_open_water == pytest.approx(eff_open_water, rel=1e-4)

    # Test the inverse
    propulsion_point_inv = propulsor.get_propulsor_data_from_vessel_speed_thrust(
        vessel_speed_kn=speed_kn,
        thrust_resistance_newton=propulsion_point.resistance_newton,
    )
    for field in fields(propulsion_point_inv):
        assert getattr(propulsion_point_inv, field.name)[0] == pytest.approx(
            getattr(propulsion_point, field.name), rel=1e-4
        )

def test_get_propulsor_data_Bseries_from_vessel_speed_rps():

    pitch_diameter_ratio = 0.721
    blade_area_ratio = 0.431
    dp_diameter_propeller_m = 9.86
    blade_number = 4
    # This data should be written in pytest.fixture style
    vessel_speed_kn = np.array([13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18])
    wake_fraction = np.array(
        [0.338, 0.337, 0.336, 0.335, 0.334, 0.332, 0.329, 0.328, 0.326, 0.325, 0.322]
    )
    thrust_deduction = np.array(
        [0.201, 0.205, 0.209, 0.214, 0.218, 0.22, 0.223, 0.224, 0.227, 0.229, 0.233]
    )

    j_array = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])
    kt_array = (
            np.array(
                [5.97, 5.54, 5.08, 4.58, 4.07, 3.57, 3.08, 2.57, 2.07, 1.58, 1.08, 0.53, 0.0]
            )
            * 1e-1
    )
    kq_array = (
            np.array(
                [8.86, 8.26, 7.72, 7.09, 6.49, 5.90, 5.31, 4.66, 4.00, 3.32, 2.62, 1.83, 0.87]
            )
            * 1e-2
    )

    propeller_curve_points = [
        OpenWaterPropellerCurvePoint(j=j, kt=kt, kq=kq)
        for j, kt, kq in zip(j_array, kt_array, kq_array)
    ]

    wake_factor_thrust_deduction_points = [
        WakeFractionThrustDeductionFactorPoint(
            wake_fraction_factor=wake_frac,
            thrust_deduction_factor=thrust_ded,
            vessel_speed_kn=speed,
        )
        for wake_frac, thrust_ded, speed in zip(
            wake_fraction, thrust_deduction, vessel_speed_kn
        )
    ]

    propulsor_bseries = PropulsorDataBseries(
        pd_pitch_diameter_ratio=pitch_diameter_ratio,
        ear_blade_area_ratio=blade_area_ratio,
        dp_diameter_propeller_m=dp_diameter_propeller_m,
        z_blade_number=blade_number,
        re=2e6,
        kp=30e-6,
        wake_thrust_reduction=wake_factor_thrust_deduction_points,
        re_correction=ReCorrection.ITTC78,
    )
    #propulsor_bseries.plot_open_water_curves()

    propulsor = PropulsorDataOpenWater(
        propeller_curve_points=propeller_curve_points,
        dp_diameter_propeller_m=4.5,
        pitch_diameter_ratio=1.2,
        wake_thrust_reduction=wake_factor_thrust_deduction_points,
    )
    #if not os.getenv("CI"):  # Skip plotting on CI or in test context
     #   propulsor.plot_open_water_curves()




    j_value = (
            random.random() * (propulsor._j.max() - propulsor._j.min()) + propulsor._j.min()
    )
    speed_kn = random.random() * 20
    n_rps = kn_to_m_per_s(speed_kn) / (j_value * propulsor._d)
    propulsion_point = propulsor.get_propulsor_data_from_vessel_speed_rps(
        vessel_speed_kn=speed_kn, n_rps=n_rps
    )
    eff_hull = (propulsion_point.resistance_newton * speed_kn) / (
            propulsion_point.propeller_thrust_newton * propulsion_point.wake_velocity_kn
    )
    eff_open_water = (
                             propulsion_point.propeller_thrust_newton
                             * kn_to_m_per_s(propulsion_point.wake_velocity_kn)
                     ) / (propulsion_point.shaft_power_kw * 1000)
    print(propulsion_point)
    assert propulsion_point.efficiency_hull == pytest.approx(eff_hull, rel=1e-4)
    assert propulsion_point.efficiency_open_water == pytest.approx(eff_open_water, rel=1e-4)

    # Test the inverse
    propulsion_point_inv = propulsor.get_propulsor_data_from_vessel_speed_thrust(
        vessel_speed_kn=speed_kn,
        thrust_resistance_newton=propulsion_point.resistance_newton,
    )
    for field in fields(propulsion_point_inv):
        assert getattr(propulsion_point_inv, field.name)[0] == pytest.approx(
            getattr(propulsion_point, field.name), rel=1e-4
        )


def test_input_data_are_the_same_length():
    # | hide
    data_set_vessel_speed_kn = np.array(
        [13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18]
    )
    data_set_wake_factor = np.array(
        [0.338, 0.337, 0.336, 0.335, 0.334, 0.332, 0.329, 0.328, 0.326, 0.325, 0.322]
    )
    data_set_thrust_reduction = np.array(
        [0.201, 0.205, 0.209, 0.214, 0.218, 0.22, 0.223, 0.224, 0.227, 0.229, 0.233]
    )
    data_set_total_resistance_kilo_newton = np.array(
        [546.3, 586.7, 626.9, 666.1, 707.3, 751.4, 799.4, 854.4, 921.3, 1001.2, 1098]
    )
    # | hide
    data_set_propeller_diameter_pd_m = 7
    data_set_j = np.array([x / 20 for x in range(0, 19)])
    data_set_kt = (
            np.array(
                [
                    4.109,
                    3.905,
                    3.699,
                    3.491,
                    3.279,
                    3.064,
                    2.847,
                    2.629,
                    2.412,
                    2.194,
                    1.977,
                    1.758,
                    1.536,
                    1.311,
                    1.078,
                    0.838,
                    0.587,
                    0.33,
                    0.068,
                ]
            )
            * 1e-1
    )
    data_set_kq = (
            np.array(
                [
                    4.828,
                    4.607,
                    4.387,
                    4.168,
                    3.949,
                    3.729,
                    3.509,
                    3.29,
                    3.071,
                    2.851,
                    2.629,
                    2.403,
                    2.17,
                    1.926,
                    1.668,
                    1.394,
                    1.103,
                    0.797,
                    0.0484,
                ]
            )
            * 1e-2
    )
    # | hide
    data_set_n_rpm = [74.1, 77, 79.9, 82.7, 85.6, 88.5, 91.6, 94.7, 98.2, 101.9, 106.3]
    data_set_power_kw = [
        4770,
        5327,
        5923,
        6552,
        7232,
        7966,
        8797,
        9719,
        10883,
        12250,
        13982,
    ]
    # | hide
    propeller_curve_points = [
        OpenWaterPropellerCurvePoint(j=j, kt=kt, kq=kq)
        for j, kt, kq in zip(data_set_j, data_set_kt, data_set_kq)
    ]
    wake_factor_thrust_deduction_points = [
        WakeFractionThrustDeductionFactorPoint(
            vessel_speed_kn=vessel_speed_kn,
            wake_fraction_factor=wake_factor,
            thrust_deduction_factor=thrust_reduction,
        )
        for vessel_speed_kn, wake_factor, thrust_reduction in zip(
            data_set_vessel_speed_kn, data_set_wake_factor, data_set_thrust_reduction
        )
    ]

    data_set_propeller = PropulsorDataOpenWater(
        propeller_curve_points=propeller_curve_points,
        dp_diameter_propeller_m=data_set_propeller_diameter_pd_m,
        wake_thrust_reduction=wake_factor_thrust_deduction_points,
    )
    rtol = 0.075
    interpolate_wake_factor = interpolate.PchipInterpolator(
        data_set_vessel_speed_kn, data_set_wake_factor
    )
    interpolate_thrust_reduction = interpolate.PchipInterpolator(
        data_set_vessel_speed_kn, data_set_thrust_reduction
    )
    data_set_propeller_performance = (
        data_set_propeller.get_propulsor_data_from_vessel_speed_thrust(
            vessel_speed_kn=data_set_vessel_speed_kn,
            thrust_resistance_newton=data_set_total_resistance_kilo_newton * 1000,
        )
    )

    assert np.allclose(
        data_set_propeller_performance.n_rpm / data_set_n_rpm, 1, rtol=rtol
    ), (
        f"Propeller data {data_set_propeller_performance.n_rpm}, "
        f"Calculated propeller rpm {data_set_propeller_performance.n_rpm} "
        f"is not equal to the expected {data_set_n_rpm} rpm"
    )
    assert np.allclose(
        data_set_propeller_performance.shaft_power_kw / data_set_power_kw, 1, rtol=rtol
    ), (
        f"Propeller data {data_set_propeller_performance}, "
        f"Calculated propeller shaft power kW {data_set_propeller_performance.shaft_power_kw} "
        f"is not equal to the expected {data_set_power_kw} kW"
    )
    assert np.allclose(
        data_set_propeller_performance.resistance_newton
        / (data_set_total_resistance_kilo_newton * 1000),
        1,
        rtol=rtol,
    ), (
        f"Propeller data {data_set_propeller_performance}, "
        f"Calculated propeller thrust newton {data_set_propeller_performance.resistance_newton} "
        f"is not equal to the expected {data_set_total_resistance_kilo_newton * 1000} newton"
    )
    # | hide
    # Check the input data are of the same length
    assert len(data_set_vessel_speed_kn) == len(
        data_set_wake_factor
    ), f"Length {len(data_set_wake_factor)} is not equal to what is expected {len(data_set_vessel_speed_kn)}"
    assert len(data_set_vessel_speed_kn) == len(
        data_set_thrust_reduction
    ), f"Length {len(data_set_thrust_reduction)} is not equal to what is expected {len(data_set_vessel_speed_kn)}"
    assert len(data_set_vessel_speed_kn) == len(
        data_set_total_resistance_kilo_newton
    ), f"Length {len(data_set_total_resistance_kilo_newton)} is not equal to what is expected {len(data_set_vessel_speed_kn)}"
    assert len(data_set_j) == len(
        data_set_kt
    ), f"Length {len(data_set_kt)} is not equal to what is expected {len(data_set_j)}"
    assert len(data_set_j) == len(
        data_set_kq
    ), f"Length {len(data_set_kq)} is not equal what is expected {len(data_set_j)}"
    assert len(data_set_vessel_speed_kn) == len(
        data_set_n_rpm
    ), f"Length {len(data_set_n_rpm)} is not equal to what is expected {len(data_set_vessel_speed_kn)}"
    assert len(data_set_vessel_speed_kn) == len(
        data_set_power_kw
    ), f"Length {len(data_set_power_kw)} is not equal what is expected {len(data_set_vessel_speed_kn)}"
