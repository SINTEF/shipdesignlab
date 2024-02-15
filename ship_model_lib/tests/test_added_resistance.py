import os
import random
import time
from typing import List

import numpy as np
import pandas as pd
import pytest
from operation_profile_lib.operation_profile_structure import Weather
from pandas.errors import EmptyDataError
from tqdm import tqdm

from ship_model_lib.added_resistance import (
    ShipDimensionsAddedResistance,
    AddedResistanceByStaWave2,
    AddedResistanceBySNNM,
    WaveSpectrumType,
    GRAVITY,
    get_wave_frequency,
    AddedResistanceWindITTC,
)
from ship_model_lib.propulsor import (
    OpenWaterPropellerCurvePoint,
    PropulsorDataOpenWater,
    WakeFractionThrustDeductionFactorPoint,
)
from ship_model_lib.types import ShipType
from ship_model_lib.utility import (
    m_per_s_to_kn,
    get_speed_kn_from_froude_number,
    kn_to_m_per_s,
)
from plotly.subplots import make_subplots

pd.options.plotting.backend = "plotly"


@pytest.fixture
def ship_dimension() -> ShipDimensionsAddedResistance:
    return ShipDimensionsAddedResistance(
        b_beam_m=32.26,
        lpp_length_between_perpendiculars_m=190,
        cb_block_coefficient=0.6,
        ta_draft_aft_m=9.5,
        tf_draft_forward_m=9.5,
        kyy_radius_gyration_in_lateral_direction_non_dim=0.26,
    )


@pytest.fixture
def dtc_ship_dimension() -> ShipDimensionsAddedResistance:
    return ShipDimensionsAddedResistance(
        b_beam_m=51.0,
        lpp_length_between_perpendiculars_m=355.0,
        cb_block_coefficient=0.661,
        ta_draft_aft_m=14.5,
        tf_draft_forward_m=14.5,
        kyy_radius_gyration_in_lateral_direction_non_dim=88.19 / 355.0,
    )


@pytest.fixture
def s_cb84_ship_dimension() -> ShipDimensionsAddedResistance:
    return ShipDimensionsAddedResistance(
        b_beam_m=32.26,
        lpp_length_between_perpendiculars_m=178,
        cb_block_coefficient=0.84,
        ta_draft_aft_m=11.57,
        tf_draft_forward_m=11.57,
    )


@pytest.fixture
def reference_data_snnm_fn_0_1() -> pd.DataFrame:
    path_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_ref_data = os.path.join(
        path_dir, "reference_data_for_added_resistance_snnm.csv"
    )
    try:
        df = pd.read_csv(path_to_ref_data, index_col=0)
    except EmptyDataError:
        df = pd.DataFrame()
    return df


@pytest.fixture
def dtc_propeller_curves() -> PropulsorDataOpenWater:
    data = [
        OpenWaterPropellerCurvePoint(j=j, kt=kt, kq=kq)
        for j, kt, kq in zip(
            np.linspace(0.0, 1.0, 21),
            np.array(
                [
                    0.509,
                    0.492,
                    0.472,
                    0.450,
                    0.427,
                    0.403,
                    0.378,
                    0.353,
                    0.327,
                    0.302,
                    0.276,
                    0.250,
                    0.225,
                    0.199,
                    0.172,
                    0.145,
                    0.118,
                    0.089,
                    0.058,
                    0.026,
                    0.0,
                ]
            ),
            np.array(
                [
                    0.713,
                    0.691,
                    0.667,
                    0.640,
                    0.613,
                    0.584,
                    0.554,
                    0.524,
                    0.493,
                    0.462,
                    0.430,
                    0.398,
                    0.366,
                    0.333,
                    0.299,
                    0.264,
                    0.228,
                    0.191,
                    0.151,
                    0.109,
                    0.065,
                ]
            ),
        )
    ]
    wake_fraction_thrust_deduction_factor_points = [
        WakeFractionThrustDeductionFactorPoint(
            vessel_speed_kn=speed, wake_fraction_factor=wf, thrust_deduction_factor=t
        )
        for speed, wf, t in zip(
            np.array([0.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]),
            np.array([0.25, 0.264, 0.276, 0.277, 0.281, 0.277, 0.275]),
            np.array([0.1, 0.081, 0.101, 0.093, 0.089, 0.101, 0.090]),
        )
    ]
    return PropulsorDataOpenWater(
        propeller_curve_points=data,
        dp_diameter_propeller_m=8.911,
        wake_thrust_reduction=wake_fraction_thrust_deduction_factor_points,
        pitch_diameter_ratio=0.959,
    )


def test_added_resistance_for_stawave2(ship_dimension: ShipDimensionsAddedResistance):
    froude_number = 0.2
    speed_m_per_s = froude_number * np.sqrt(
        ship_dimension.lpp_length_between_perpendiculars_m * GRAVITY
    )
    r_aw_list = []
    wave_length_list = []
    speed_kn = m_per_s_to_kn(speed_m_per_s)
    wave_height_array = np.linspace(0.1, 10, 100)

    # Wave spectrum is JONSWAP for the first case
    added_resistance = AddedResistanceByStaWave2(
        ship_dimension=ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
    )

    # Test with heading / without / > 45 deg
    resistance_newton = added_resistance.get_added_resistance_newton(
        vessel_speed_kn=speed_kn,
        weather=Weather(
            significant_wave_height_m=3, wave_direction_deg=0, mean_wave_period_s=10
        ),
    )[0]

    resistance_newton_heading = added_resistance.get_added_resistance_newton(
        vessel_speed_kn=speed_kn,
        weather=Weather(
            significant_wave_height_m=3, mean_wave_period_s=10, wave_direction_deg=0
        ),
        heading_deg=0,
    )
    resistance_newton_heading_gt_45 = added_resistance.get_added_resistance_newton(
        vessel_speed_kn=speed_kn,
        weather=Weather(
            significant_wave_height_m=3, mean_wave_period_s=10, wave_direction_deg=0
        ),
        heading_deg=50,
    )
    assert resistance_newton == pytest.approx(resistance_newton_heading)
    assert resistance_newton_heading_gt_45 == pytest.approx(0.0)

    # test added resistance with when heading is given but the wave direction is not given
    resistance_newton_zero = added_resistance.get_added_resistance_newton(
        vessel_speed_kn=speed_kn,
        weather=Weather(
            significant_wave_height_m=0, mean_wave_period_s=0, wave_direction_deg=0
        ),
        heading_deg=0,
    )
    assert resistance_newton_zero == pytest.approx(0.0)

    # Test with an array that contains 0 wave height
    vessel_speed_kn = np.array([0, 0])
    weather = Weather(
        significant_wave_height_m=np.array([0, 1]),
        mean_wave_period_s=np.array([0, 2]),
        wave_direction_deg=np.array([0, 0]),
    )
    heading = np.array([0, 0])
    resistance_newton_contains_zero = added_resistance.get_added_resistance_newton(
        vessel_speed_kn=vessel_speed_kn, weather=weather, heading_deg=heading
    )
    assert resistance_newton_contains_zero[0] == pytest.approx(0)
    assert resistance_newton_contains_zero[1] > 0


def test_wave_encounter_angle(ship_dimension: ShipDimensionsAddedResistance):
    """Test the wave encounter angle calculation"""
    added_resistance = AddedResistanceByStaWave2(
        ship_dimension=ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
    )
    heading = random.randint(0, 10) * 360 + random.random() * 22.5
    wave_direction = random.randint(0, 10) * 360 + random.random() * 22.5
    weather = Weather(
        significant_wave_height_m=3,
        mean_wave_period_s=10,
        wave_direction_deg=wave_direction,
    )
    added_resistance_value = added_resistance.get_added_resistance_newton(
        vessel_speed_kn=10, weather=weather, heading_deg=heading
    )
    assert added_resistance_value > 0, "The added resistance should be not zero"
    heading = random.randint(0, 10) * 360 + wave_direction + 45 + random.random() * 22.5
    added_resistance_value = added_resistance.get_added_resistance_newton(
        vessel_speed_kn=10, weather=weather, heading_deg=heading
    )
    assert added_resistance_value == pytest.approx(
        0
    ), "The added resistance should be zero"


def test_snnm_method_for_resistance_component(
    dtc_ship_dimension, s_cb84_ship_dimension, reference_data_snnm_fn_0_1
):
    """Test the SNNM method"""
    added_resistance = AddedResistanceBySNNM(
        ship_dimension=s_cb84_ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
        ship_type=ShipType.bulk_handysize,
    )
    fig = reference_data_snnm_fn_0_1.plot()
    lpp = s_cb84_ship_dimension.lpp_length_between_perpendiculars_m
    wave_length_normalized = np.linspace(0.25, 2.0, 36)
    wave_length_m = wave_length_normalized * lpp
    omega = get_wave_frequency(wave_length_m)
    wave_incident_angle_list_rad = np.array(
        [0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi * 5 / 6, np.pi]
    )
    b = s_cb84_ship_dimension.b_beam_m
    fr = 0.1
    vessel_speed_kn = get_speed_kn_from_froude_number(froude_number=fr, lpp_m=lpp)
    for wave_incident_angle_rad in wave_incident_angle_list_rad:
        wave_resistance = added_resistance._get_non_dimensional_wave_resistance(
            wave_frequency_rad_per_s=omega,
            wave_incident_angle_rad=wave_incident_angle_rad,
            vessel_speed_kn=vessel_speed_kn,
        )
        wave_resistance_motion = (
            added_resistance._get_non_dimensional_wave_resistance_due_to_motion(
                wave_frequency_rad_per_s=omega,
                wave_incident_angle_rad=wave_incident_angle_rad,
                vessel_speed_kn=vessel_speed_kn,
            )
        )
        (
            wave_resistance_reflection,
            _,
        ) = added_resistance._get_non_dimensional_wave_resistance_due_to_reflection(
            wave_frequency_rad_per_s=omega,
            wave_incident_angle_rad=wave_incident_angle_rad,
            vessel_speed_kn=vessel_speed_kn,
        )
        index = 0
        for wave_resistance_component in [
            wave_resistance,
            wave_resistance_motion,
            wave_resistance_reflection,
        ]:
            index += 1
            fig.add_scatter(
                x=wave_length_normalized,
                y=wave_resistance_component,
                name=f"wave_incident_angle_rad={np.rad2deg(wave_incident_angle_rad):0.1f}-{index}",
            )
    fig.show()

    added_resistance = AddedResistanceBySNNM(
        ship_dimension=dtc_ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
        ship_type=ShipType.container,
    )
    lpp = dtc_ship_dimension.lpp_length_between_perpendiculars_m
    wave_length_normalized = np.linspace(0.1, 1.6, 16)
    wave_incident_angle_list_deg = np.linspace(0, 180, 37)
    wave_incident_angle_list_rad = np.deg2rad(wave_incident_angle_list_deg)
    wave_length_m = wave_length_normalized * lpp
    omega = get_wave_frequency(wave_length_m)
    b = dtc_ship_dimension.b_beam_m
    fr = 0.0
    vessel_speed_kn = get_speed_kn_from_froude_number(froude_number=fr, lpp_m=lpp)
    result = added_resistance._get_non_dimensional_wave_resistance(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=wave_incident_angle_list_rad[0],
        vessel_speed_kn=vessel_speed_kn,
    )
    for wave_incident_angle_rad in wave_incident_angle_list_rad[1:]:
        result_temp = added_resistance._get_non_dimensional_wave_resistance(
            wave_frequency_rad_per_s=omega,
            wave_incident_angle_rad=wave_incident_angle_rad,
            vessel_speed_kn=vessel_speed_kn,
        )
        result = np.vstack((result, result_temp))
    fig = make_subplots()
    fig.add_surface(
        x=np.flip(wave_length_normalized),
        y=wave_incident_angle_list_deg,
        z=result,
    )
    fig.show()


def test_snnm_method_for_wave_resistance(s_cb84_ship_dimension):
    """Test the SNNM method"""
    froude_number = 0.2
    added_resistance_stawave2 = AddedResistanceByStaWave2(
        ship_dimension=s_cb84_ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
    )
    added_resistance = AddedResistanceBySNNM(
        ship_dimension=s_cb84_ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
        ship_type=ShipType.ro_pax,
    )
    speed_kn = get_speed_kn_from_froude_number(
        froude_number=froude_number,
        lpp_m=s_cb84_ship_dimension.lpp_length_between_perpendiculars_m,
    )
    wave_height_list = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.8])
    wave_period_list = np.array([4.5, 6.5, 7.3, 9.0, 9.8, 10.6])
    wave_heading = np.array([i * 30 for i in range(12)])
    fig = make_subplots()
    resistance_stawave2_list = []
    for wave_height, wave_period in zip(wave_height_list, wave_period_list):
        start_time = time.time()
        resistance_stawave2_newton = (
            added_resistance_stawave2.get_added_resistance_newton(
                vessel_speed_kn=speed_kn,
                weather=Weather(
                    significant_wave_height_m=wave_height,
                    mean_wave_period_s=wave_period,
                    wave_direction_deg=0,
                ),
                heading_deg=0,
            )[0]
        )
        # print(f"SW2: wave_height={wave_height}, wave_period={wave_period}, time={(time.time() - start_time) / 7}")
        start_time = time.time()
        resistance_newton = np.array(
            [
                added_resistance.get_added_resistance_newton(
                    vessel_speed_kn=speed_kn,
                    weather=Weather(
                        significant_wave_height_m=wave_height,
                        mean_wave_period_s=wave_period,
                        wave_direction_deg=wave_direction_deg,
                    ),
                    heading_deg=0,
                )[0]
                for wave_direction_deg in wave_heading
            ]
        )
        # print(f"SNNM: wave_height={wave_height}, wave_period={wave_period}, time={(time.time() - start_time) / 7}")

        fig.add_scatter(
            x=wave_heading,
            y=resistance_newton / resistance_stawave2_newton,
            name=f"wave_height={wave_height}, wave_period={wave_period}",
        )
        resistance_stawave2_list.append(resistance_stawave2_newton)

    fig.show()
    fig = make_subplots()
    fig.add_scatter(x=wave_height_list, y=resistance_stawave2_list)
    fig.show()


def test_snnm_method_for_integration_interval(s_cb84_ship_dimension):
    froude_number = 0.2
    speed_kn = get_speed_kn_from_froude_number(
        froude_number=froude_number,
        lpp_m=s_cb84_ship_dimension.lpp_length_between_perpendiculars_m,
    )
    added_resistance = AddedResistanceBySNNM(
        ship_dimension=s_cb84_ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
        ship_type=ShipType.ro_pax,
    )
    wave_height_list = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 5.8])
    wave_period_list = np.array([4.5, 6.5, 7.3, 9.0, 9.8, 10.6])
    wave_heading = np.array([i * 30 for i in range(12)])
    fig = make_subplots()
    # for interval in [100, 1000, 10000, 100000]:
    df = pd.DataFrame(index=wave_heading)
    for wave_height, wave_period in zip(wave_height_list, wave_period_list):
        result = []
        for wave_direction_deg in tqdm(wave_heading):
            start_time = time.time()
            result.append(
                added_resistance.get_added_resistance_newton(
                    vessel_speed_kn=speed_kn,
                    weather=Weather(
                        significant_wave_height_m=wave_height,
                        mean_wave_period_s=wave_period,
                        wave_direction_deg=wave_direction_deg,
                    ),
                    heading_deg=0,
                )[0]
            )
            # print(f"wave_height={wave_height}, wave_period={wave_period}, time={(time.time() - start_time) / 7}")
        df[f"hs-{wave_height}-tp-{wave_period}"] = np.array(result)
    # df.to_csv(f"snnm_integration_interval_100.csv", index=True)


def test_added_resistance_wind():
    """Test the added resistance by wind"""
    number_of_samples = 10
    transverse_area = 30 * 26.7
    wind_speed_m_per_s = np.array(
        [np.random.uniform(0, 10) for _ in range(number_of_samples)]
    )
    vessel_speed_kn = np.array(
        [np.random.uniform(5, 15) for _ in range(number_of_samples)]
    )
    wind_direction = np.array(
        [np.random.uniform(0, 360) for _ in range(number_of_samples)]
    )
    heading = np.array([np.random.uniform(0, 360) for _ in range(number_of_samples)])
    added_resistance = AddedResistanceWindITTC(
        transverse_area_m2=transverse_area,
        is_laden=True,
        ship_type=ShipType.ro_pax,
    )
    vs = kn_to_m_per_s(vessel_speed_kn)
    angle_vs = np.pi / 2 - np.deg2rad(heading)
    angle_vw = 3 * np.pi / 2 - np.deg2rad(wind_direction)
    vector_vs = (vs * np.array([np.cos(angle_vs), np.sin(angle_vs)])).transpose()
    vector_vw = (
        wind_speed_m_per_s * np.array([np.cos(angle_vw), np.sin(angle_vw)])
    ).transpose()
    vector_rel_vw = vector_vw - vector_vs
    rel_wind_speed_ref = np.linalg.norm(vector_rel_vw, axis=1)
    rel_wind_speed_angle_ref = np.rad2deg(
        np.arccos(
            np.sum(vector_vs * -vector_rel_vw, axis=1) / (vs * rel_wind_speed_ref)
        )
    )
    fig = make_subplots()
    for index in range(number_of_samples):
        fig.add_scatter(
            x=[0, vector_vs[index][0]],
            y=[0, vector_vs[index][1]],
            name=f"Vessel-{index}",
        )
        fig.add_scatter(
            x=[0, vector_vw[index][0]], y=[0, vector_vw[index][1]], name="Wind"
        )
        fig.add_scatter(
            x=[vector_vs[index][0], vector_vs[index][0] + vector_rel_vw[index][0]],
            y=[vector_vs[index][1], vector_vs[index][1] + vector_rel_vw[index][1]],
            name="Relative wind",
        )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    fig.show()
    # print(f"Given: wind speed = {wind_speed_m_per_s} m/s, "
    #       f"vessel speed = {vessel_speed_kn} kn, "
    #       f"wind direction = {wind_direction} deg, "
    #       f"heading = {heading} deg")
    (
        rel_wind_speed,
        rel_wind_speed_angle,
    ) = added_resistance._get_relative_wind_angle_and_speed(
        vessel_speed_m_per_s=vs,
        weather=Weather(
            wind_speed_m_per_s=wind_speed_m_per_s, wind_direction_deg=wind_direction
        ),
        heading_deg=heading,
    )
    for index in range(number_of_samples):
        assert rel_wind_speed_ref[index] == pytest.approx(rel_wind_speed[index])
        assert rel_wind_speed_angle_ref[index] == pytest.approx(
            rel_wind_speed_angle[index] * 180 / np.pi
        )
    resistance = added_resistance.get_added_resistance_newton(
        vessel_speed_kn=vs,
        weather=Weather(
            wind_speed_m_per_s=wind_speed_m_per_s, wind_direction_deg=wind_direction
        ),
        heading_deg=heading,
    )
    # print(f"Resistance = {resistance} N")


def test_added_resistance_wind_with_zero_speed():
    """Test the added resistance by wind with zero speed"""
    transverse_area = 30 * 26.7
    wind_speed_m_per_s = 5 * np.random.uniform(0.1, 1)
    vessel_speed_kn = np.zeros(1)
    wind_direction = 360 * np.random.uniform(0, 1)
    heading = np.zeros(1)
    added_resistance = AddedResistanceWindITTC(
        transverse_area_m2=transverse_area,
        is_laden=True,
        ship_type=ShipType.ro_pax,
    )
    rel_speed, rel_angle = added_resistance._get_relative_wind_angle_and_speed(
        vessel_speed_m_per_s=vessel_speed_kn,
        weather=Weather(
            wind_speed_m_per_s=wind_speed_m_per_s, wind_direction_deg=wind_direction
        ),
        heading_deg=heading,
    )
    assert np.allclose(rel_speed, wind_speed_m_per_s)
    assert np.allclose(np.rad2deg(rel_angle), wind_direction)

    # Test the case where the relative wind speed is zero
    vessel_speed_kn = wind_speed_m_per_s
    heading = wind_direction + 180
    rel_speed, rel_angle = added_resistance._get_relative_wind_angle_and_speed(
        vessel_speed_m_per_s=vessel_speed_kn,
        weather=Weather(
            wind_speed_m_per_s=wind_speed_m_per_s, wind_direction_deg=wind_direction
        ),
        heading_deg=heading,
    )
    assert np.allclose(rel_speed, 0)

    # Test the case where all speed is zero
    wind_speed_m_per_s = np.zeros(1)
    vessel_speed_kn = np.zeros(1)
    heading = np.zeros(1)
    rel_speed, rel_angle = added_resistance._get_relative_wind_angle_and_speed(
        vessel_speed_m_per_s=vessel_speed_kn,
        weather=Weather(
            wind_speed_m_per_s=wind_speed_m_per_s, wind_direction_deg=wind_direction
        ),
        heading_deg=heading,
    )
    assert np.allclose(rel_speed, 0)

    # Test the case with mixture - first all zero speed, then zero relative wind speed,
    # then vessel zero speed, finally normal case
    wind_speed_m_per_s = np.array([0, 5, 0, 6.5])
    wind_direction = np.array([0, 0, 90, 60])
    vessel_speed_kn = np.array([0, 5, 0, 4.5])
    heading = np.array([0, 180, 90, 30])
    rel_speed, rel_angle = added_resistance._get_relative_wind_angle_and_speed(
        vessel_speed_m_per_s=vessel_speed_kn,
        weather=Weather(
            wind_speed_m_per_s=wind_speed_m_per_s, wind_direction_deg=wind_direction
        ),
        heading_deg=heading,
    )
    assert np.allclose(rel_speed[:2], 0)
    assert np.allclose(rel_speed[2], wind_speed_m_per_s[2])
    assert np.allclose(rel_angle[2], np.deg2rad(wind_direction[2]))
    assert np.alltrue(rel_speed[3] > 0)


@pytest.fixture
def added_resistance_ntnu_general_cargo() -> AddedResistanceBySNNM:
    """Fixture for added resistance by NTNU general cargo"""
    ship_dimension = ShipDimensionsAddedResistance(
        lpp_length_between_perpendiculars_m=194,
        b_beam_m=32.266,
        ta_draft_aft_m=12.64,
        tf_draft_forward_m=12.64,
        cb_block_coefficient=0.79,
    )
    return AddedResistanceBySNNM(
        ship_dimension=ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        ship_type=ShipType.ro_pax,
        length_of_entrance=38.51,
        length_of_run=32.31,
    )


def test_added_resistance_wave_with_ntnu_head_sea(added_resistance_ntnu_general_cargo):
    """Test the added resistance by wave with NTNU data"""
    path_to_ref_data = os.path.join(
        os.path.dirname(__file__), "Liu_method_alldir_output_head_sea.csv"
    )
    ref_data = pd.read_csv(path_to_ref_data, index_col=0)
    lpp = (
        added_resistance_ntnu_general_cargo.ship_dimension.lpp_length_between_perpendiculars_m
    )
    b = added_resistance_ntnu_general_cargo.ship_dimension.b_beam_m
    cb = added_resistance_ntnu_general_cargo.ship_dimension.cb_block_coefficient
    kyy = (
        added_resistance_ntnu_general_cargo.ship_dimension.kyy_radius_gyration_in_lateral_direction_non_dim
    )
    wave_length_normalized = np.linspace(0.15, 2.04, 190)
    wave_length_m = wave_length_normalized * lpp
    omega = get_wave_frequency(wave_length_m)
    fr = 0.182
    vs_m_per_s = fr * np.sqrt(GRAVITY * lpp)
    vs_kn = m_per_s_to_kn(vs_m_per_s)
    theta = 0
    a1 = (
        added_resistance_ntnu_general_cargo._get_a1(
            wave_frequency_rad_per_s=omega,
            wave_incident_angle_rad=np.deg2rad(theta),
            vessel_speed_kn=vs_kn,
        )
        * 60.3
        * cb**1.34
        * (4 * kyy) ** 2
    )
    a2 = added_resistance_ntnu_general_cargo._get_a2(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=np.deg2rad(theta),
        vessel_speed_kn=vs_kn,
    )
    a2 *= np.ones_like(a1)
    a3 = added_resistance_ntnu_general_cargo._a3
    a3 *= np.ones_like(a1)
    w_normalized = added_resistance_ntnu_general_cargo._get_omega_normalized(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=np.deg2rad(theta),
        vessel_speed_kn=vs_kn,
    )
    b1 = np.ones_like(a1)
    b1[w_normalized < 1.0] = 11.0
    b1[w_normalized >= 1.0] = -8.5
    d1 = np.ones_like(w_normalized)
    d1[w_normalized < 1.0] = 566 * np.power(lpp * cb / b, -2.66)
    d1[w_normalized >= 1.0] = (
        -566
        * np.power(lpp / b, -2.66)
        * (4 - 125 * added_resistance_ntnu_general_cargo._trim_angle_rad)
    )
    (
        r_awr,
        r_awr_comp,
    ) = added_resistance_ntnu_general_cargo._get_non_dimensional_wave_resistance_due_to_reflection(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=np.deg2rad(theta),
        vessel_speed_kn=vs_kn,
    )
    r_awm = added_resistance_ntnu_general_cargo._get_non_dimensional_wave_resistance_due_to_motion(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=np.deg2rad(theta),
        vessel_speed_kn=vs_kn,
    )
    r_aw = r_awr + r_awm
    assert np.allclose(ref_data["a1"], a1, atol=1e-3)
    assert np.allclose(ref_data["a2"], a2, atol=1e-3)
    assert np.allclose(ref_data["a3"], a3, atol=1e-3)
    assert np.allclose(ref_data["W"], w_normalized, atol=1e-3)
    assert np.allclose(ref_data["b1"], b1, atol=1e-3)
    assert np.allclose(ref_data["d1"], d1, atol=1e-3)
    assert np.allclose(ref_data["R1_AWRL"], r_awr_comp[0], atol=1e-3)
    assert np.allclose(ref_data["R2_AWRL"], r_awr_comp[1], atol=1e-3)
    assert np.allclose(ref_data["R3_AWRL"], r_awr_comp[2], atol=1e-3)
    assert np.allclose(ref_data["R4_AWRL"], r_awr_comp[3], atol=1e-3)
    assert np.allclose(ref_data["R_AWRL"], r_awr, atol=1e-3)
    assert np.allclose(ref_data["R_AWML"], r_awm, atol=1e-3)
    assert np.allclose(ref_data["Raw"], r_aw, atol=1e-3)

    # fig = ref_data.plot()
    # fig.add_scatter(x=wave_length_normalized, y=a1, name=f"a1-ref")
    # fig.add_scatter(x=wave_length_normalized, y=a2, name=f"a2-ref")
    # fig.add_scatter(x=wave_length_normalized, y=a3, name=f"a3-ref")
    # fig.add_scatter(x=wave_length_normalized, y=w_normalized, name=f"w-ref")
    # fig.add_scatter(x=wave_length_normalized, y=b1, name=f"b1-ref")
    # fig.add_scatter(x=wave_length_normalized, y=d1, name=f"d1-ref")
    # fig.add_scatter(x=wave_length_normalized, y=r_awr_comp[0], name=f"r1_awr-ref")

    # r_awr = added_resistance._get_non_dimensional_wave_resistance_due_to_reflection(
    #     wave_frequency_rad_per_s=omega,
    #     wave_incident_angle_rad=np.deg2rad(theta),
    #     vessel_speed_kn=vs_kn,
    # )[0]
    # resistance_component = r_awm + r_awr
    # fig.add_scatter(x=wave_length_normalized, y=resistance_component, name=f"theta={theta}")
    # fig.add_scatter(x=wave_length_normalized, y=r_awm, name=f"theta={theta} (M)")
    # fig.add_scatter(x=wave_length_normalized, y=r_awr, name=f"theta={theta} (R)")
    # fig.show()


def test_added_resistance_wave_with_ntnu_following_sea_120(
    added_resistance_ntnu_general_cargo,
):
    """Test the added resistance by wave with NTNU data"""
    show_plot = False
    theta = 120  # Wave encounter angle 0 - head sea, 180 - following sea
    path_to_ref_data = os.path.join(
        os.path.dirname(__file__), "Liu_method_alldir_output_120.csv"
    )
    ref_data = pd.read_csv(path_to_ref_data, index_col=0)

    lpp = (
        added_resistance_ntnu_general_cargo.ship_dimension.lpp_length_between_perpendiculars_m
    )
    b = added_resistance_ntnu_general_cargo.ship_dimension.b_beam_m
    cb = added_resistance_ntnu_general_cargo.ship_dimension.cb_block_coefficient
    kyy = (
        added_resistance_ntnu_general_cargo.ship_dimension.kyy_radius_gyration_in_lateral_direction_non_dim
    )

    wave_length_normalized = np.linspace(0.15, 2.04, 190)
    wave_length_m = wave_length_normalized * lpp
    omega = get_wave_frequency(wave_length_m)
    fr = 0.182
    vs_m_per_s = fr * np.sqrt(GRAVITY * lpp)
    vs_kn = m_per_s_to_kn(vs_m_per_s)
    a1 = (
        added_resistance_ntnu_general_cargo._get_a1(
            wave_frequency_rad_per_s=omega,
            wave_incident_angle_rad=np.deg2rad(theta),
            vessel_speed_kn=vs_kn,
        )
        * 60.3
        * cb**1.34
        * (4 * kyy) ** 2
    )
    a2 = added_resistance_ntnu_general_cargo._get_a2(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=np.deg2rad(theta),
        vessel_speed_kn=vs_kn,
    )
    a2 *= np.ones_like(a1)
    a3 = added_resistance_ntnu_general_cargo._a3
    a3 *= np.ones_like(a1)
    w_normalized = added_resistance_ntnu_general_cargo._get_omega_normalized(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=np.deg2rad(theta),
        vessel_speed_kn=vs_kn,
    )
    b1 = np.ones_like(a1)
    b1[w_normalized < 1.0] = 11.0
    b1[w_normalized >= 1.0] = -8.5
    d1 = np.ones_like(w_normalized)
    d1[w_normalized < 1.0] = 566 * np.power(lpp * cb / b, -2.66)
    d1[w_normalized >= 1.0] = (
        -566
        * np.power(lpp / b, -2.66)
        * (4 - 125 * added_resistance_ntnu_general_cargo._trim_angle_rad)
    )
    (
        r_awr,
        r_awr_comp,
    ) = added_resistance_ntnu_general_cargo._get_non_dimensional_wave_resistance_due_to_reflection(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=np.deg2rad(theta),
        vessel_speed_kn=vs_kn,
    )
    r_awm = added_resistance_ntnu_general_cargo._get_non_dimensional_wave_resistance_due_to_motion(
        wave_frequency_rad_per_s=omega,
        wave_incident_angle_rad=np.deg2rad(theta),
        vessel_speed_kn=vs_kn,
    )
    r_aw = r_awr + r_awm
    assert np.allclose(ref_data["a1"], a1, atol=1e-3)
    assert np.allclose(ref_data["a2"], a2, atol=1e-3)
    assert np.allclose(ref_data["a3"], a3, atol=1e-3)
    assert np.allclose(ref_data["W"], w_normalized, atol=1e-3)
    assert np.allclose(ref_data["b1"], b1, atol=1e-3)
    assert np.allclose(ref_data["d1"], d1, atol=1e-3)
    assert np.allclose(ref_data["R1_AWRL"], r_awr_comp[0], atol=1e-3)
    assert np.allclose(ref_data["R2_AWRL"], r_awr_comp[1], atol=1e-3)
    assert np.allclose(ref_data["R3_AWRL"], r_awr_comp[2], atol=1e-3)
    assert np.allclose(ref_data["R4_AWRL"], r_awr_comp[3], atol=1e-3)
    assert np.allclose(ref_data["R_AWRL"], r_awr, atol=1e-3)
    assert np.allclose(ref_data["R_AWML"], r_awm, atol=1e-3)
    assert np.allclose(ref_data["Raw"], r_aw, atol=1e-3)

    if show_plot:
        fig = ref_data.plot()
        fig.add_scatter(x=wave_length_normalized, y=a1, name=f"a1-ref")
        fig.add_scatter(x=wave_length_normalized, y=a2, name=f"a2-ref")
        fig.add_scatter(x=wave_length_normalized, y=a3, name=f"a3-ref")
        fig.add_scatter(x=wave_length_normalized, y=w_normalized, name=f"w-ref")
        fig.add_scatter(x=wave_length_normalized, y=b1, name=f"b1-ref")
        fig.add_scatter(x=wave_length_normalized, y=d1, name=f"d1-ref")
        fig.add_scatter(x=wave_length_normalized, y=r_awr_comp[0], name=f"r1_awr-ref")

        r_awr = added_resistance_ntnu_general_cargo._get_non_dimensional_wave_resistance_due_to_reflection(
            wave_frequency_rad_per_s=omega,
            wave_incident_angle_rad=np.deg2rad(theta),
            vessel_speed_kn=vs_kn,
        )[
            0
        ]
        resistance_component = r_awm + r_awr
        fig.add_scatter(
            x=wave_length_normalized, y=resistance_component, name=f"theta={theta}"
        )
        fig.add_scatter(x=wave_length_normalized, y=r_awm, name=f"theta={theta} (M)")
        fig.add_scatter(x=wave_length_normalized, y=r_awr, name=f"theta={theta} (R)")
        fig.show()


def test_added_resistance_ntnu_general_cargo_arbitrary_heading(
    added_resistance_ntnu_general_cargo,
):
    """Test added resistance for general cargo ship."""
    show_plot = False
    path_to_ref_data = os.path.join(
        os.path.dirname(__file__), "Liu_method_alldir_output_arbitrary_wave_dir.csv"
    )
    ref_data = pd.read_csv(path_to_ref_data, index_col=0)
    lpp = (
        added_resistance_ntnu_general_cargo.ship_dimension.lpp_length_between_perpendiculars_m
    )
    wave_length_normalized = np.linspace(0.15, 2.04, 190)
    wave_length_m = wave_length_normalized * lpp
    omega = get_wave_frequency(wave_length_m)
    fr = 0.182
    vs_kn = get_speed_kn_from_froude_number(fr, lpp)
    fig = ref_data.plot()
    for theta in [0, 30, 60, 90, 120, 150, 180]:
        (
            r_awr,
            r_awr_comp,
        ) = added_resistance_ntnu_general_cargo._get_non_dimensional_wave_resistance_due_to_reflection(
            wave_frequency_rad_per_s=omega,
            wave_incident_angle_rad=np.deg2rad(theta),
            vessel_speed_kn=vs_kn,
        )
        r_awm = added_resistance_ntnu_general_cargo._get_non_dimensional_wave_resistance_due_to_motion(
            wave_frequency_rad_per_s=omega,
            wave_incident_angle_rad=np.deg2rad(theta),
            vessel_speed_kn=vs_kn,
        )
        r_aw = r_awr + r_awm
        assert np.allclose(ref_data[f"r_aw-{180 - theta}"], r_aw, atol=1e-3)
        assert np.allclose(ref_data[f"r_awm-{180 - theta}"], r_awm, atol=1e-3)
        assert np.allclose(ref_data[f"r_awr-{180 - theta}"], r_awr, atol=1e-3)
        if show_plot:
            fig.add_scatter(
                x=wave_length_normalized, y=r_aw, name=f"r_aw_theta={theta}"
            )
            fig.add_scatter(
                x=wave_length_normalized, y=r_awr, name=f"r_awr_theta={theta}"
            )
            fig.add_scatter(
                x=wave_length_normalized, y=r_awm, name=f"r_awm_theta={theta}"
            )
    if show_plot:
        fig.show()
