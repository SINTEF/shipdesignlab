import random
import numpy as np
import pandas as pd
import pytest
import os
from datetime import datetime
from collections import namedtuple
from operation_profile_lib.operation_profile_structure import Weather, OperationPoint, Location

from ship_model_lib.added_resistance import (
    AddedResistanceByStaWave2,
    AddedResistanceWindITTC,
    AddedResistanceBySNNM,
    AddedResistanceBySeaMarginCurve,
    WaveSpectrumType,
)
from ship_model_lib.machinery import (
    PowerLoad,
    Curve,
    Point,
    PropulsionType,
    MachinerySystem,
    FuelByMassFraction,
    EmissionType,
    EmissionFactor,
    PowerSourceWithEfficiency,
    MachinerySubsystemSimple,
)
from ship_model_lib.ship_dimensions import (
    ShipDimensionsHollenbachTwinScrew,
    ShipDimensionsAddedResistance,
)
from ship_model_lib.ship_model import ShipModel, ShipDescription, ShipType
from ship_model_lib.propulsor import PropulsorDataScalar, PropulsorDataBseries
from ship_model_lib.calm_water_resistance import (
    CalmWaterResistanceBySpeedPowerCurve,
    CalmWaterResistanceHollenbachTwinScrewDesignDraft,
    CalmWaterResistanceBySpeedResistanceCurve,
)
from test_machinery import get_machinery_system_nodel
from test_added_resistance import ship_dimension

from ship_model_lib.ship_model import (
    WakeFractionThrustDeductionFactorPoint,
    PropulsorDataBseries,
    PropulsorDataOpenWater,
    PropulsorDataScalar,
)

from ship_model_lib.ship_model import (
    CalmWaterResistanceHollenbachSingleScrewDesignDraft,
    ShipDimensionsHollenbachSingleScrew,
    CalmWaterResistanceHollenbachSingleScrewBallastDraft,
)

from ship_model_lib.ship_model import ShipModel
from ship_model_lib.types import ShipDescription, HullData, PropulsorData



def make_ship_model(calm_water_resistance, propulsor_efficiency, machinery_system):
    description = ShipDescription(name="Test Ship")
    hull_data = HullData(calm_water_resistance=calm_water_resistance)
    propulsor_data = PropulsorData(efficiency=propulsor_efficiency)
    return ShipModel(
        description=description,
        hull_data=hull_data,
        propulsor_data=propulsor_data,
        machinery_system=machinery_system,
    )





def verify_ship_model_performance(
    ship_model: ShipModel, speed_array_kn: np.ndarray = None
):
    """Verify the method to get performance of the ship model."""
    auxiliary_power_kw = 500
    performance_data = ship_model.get_ship_performance_data_from_speed(
        vessel_speed_kn=speed_array_kn, auxiliary_power_kw=auxiliary_power_kw
    )
    performance_data_backward = ship_model.get_ship_performance_data_from_power(
        power_out_source_kw=performance_data.power_source_data.total.power_on_source_kw,
        auxiliary_power_kw=auxiliary_power_kw,
    )

    assert np.all(performance_data.propeller_data.shaft_power_kw[1:] > 0)
    if ship_model.machinery_system.propulsion_type == PropulsionType.ELECTRIC:
        assert (
            performance_data.power_source_data.mechanical_system.power_on_source_kw == 0
        ), "fail 1"
        assert np.all(
            performance_data.power_source_data.electric_system.power_on_source_kw
            > performance_data.propeller_data.shaft_power_kw + auxiliary_power_kw
        ), "fail 2"
    else:
        assert np.all(
            performance_data.power_source_data.mechanical_system.power_on_source_kw
            > performance_data.propeller_data.shaft_power_kw
        )
        assert np.all(
            performance_data.power_source_data.electric_system.power_on_source_kw
            >= auxiliary_power_kw
        )
    assert np.allclose(
        performance_data_backward.power_source_data.total.fuel_consumption.total_fuel_consumption,
        performance_data.power_source_data.total.fuel_consumption.total_fuel_consumption,
    )
    assert np.allclose(
        performance_data_backward.hull_data.vessel_speed_kn, speed_array_kn
    )
    assert np.allclose(
        performance_data_backward.propeller_data.shaft_power_kw,
        performance_data.propeller_data.shaft_power_kw,
    )
    assert np.allclose(
        performance_data_backward.propeller_data.n_rpm,
        performance_data.propeller_data.n_rpm,
    )


@pytest.fixture
def calm_water_resistance() -> CalmWaterResistanceBySpeedPowerCurve:
    design_speed_kn = 20
    speed_ref_array = np.linspace(0, design_speed_kn, 21)
    power_ref_array = 10000 * 0.85 / design_speed_kn**3 * speed_ref_array**3
    return CalmWaterResistanceBySpeedPowerCurve(
        speed_ref_kn=speed_ref_array,
        power_ref_kw=power_ref_array,
    )


@pytest.fixture
def calm_water_resistance_hollenbach(
    ship_dimension,
) -> CalmWaterResistanceHollenbachTwinScrewDesignDraft:
    """Return the calm water resistance model for the Hollenbach twin screw design draft."""
    return CalmWaterResistanceHollenbachTwinScrewDesignDraft(
        ship_dimensions=ship_dimension
    )


@pytest.fixture
def machinery_system() -> MachinerySystem:
    return get_machinery_system_nodel(
        propulsion_type=PropulsionType.ELECTRIC,
        efficiency_propulsion_drive=0.85,
        efficiency_power_source=0.45,
        efficiency_auxiliary_load=1.0,
        rated_power_source_kw=10000,
    )


@pytest.fixture
def propulsor_b_series() -> PropulsorDataBseries:
    return PropulsorDataBseries(
        pd_pitch_diameter_ratio=1.2,
        ear_blade_area_ratio=0.8,
        dp_diameter_propeller_m=4.5,
        z_blade_number=5,
    )


def test_ship_model_get_performance_data_from_speed_and_vice_versa(
    calm_water_resistance, machinery_system
):
    """Test ShipModel.get_performance_data_from_speed and ShipMode.get_performance_from_power."""
    # Test with the electric propulsion system
    design_speed_kn = 20

    ship_model = make_ship_model(calm_water_resistance, 0.7, machinery_system)
    # ship_model = ShipModel(
    #     calm_water_resistance=calm_water_resistance,
    #     propulsor=PropulsorDataScalar(efficiency=0.7),
    #     machinery_system=machinery_system,
    # )
    speed_array_kn = np.linspace(1, design_speed_kn, 21)
    verify_ship_model_performance(ship_model=ship_model, speed_array_kn=speed_array_kn)

    # Test with the mechanical propulsion system
    machinery_system_mechanical_propulsion = get_machinery_system_nodel(
        propulsion_type=PropulsionType.MECHANICAL,
        efficiency_propulsion_drive=0.85,
        efficiency_power_source=0.45,
        efficiency_auxiliary_load=1.0,
        rated_power_source_kw=10000,
        rated_power_auxiliary_kw=1000,
    )
    ship_model.machinery_system = machinery_system_mechanical_propulsion
    verify_ship_model_performance(ship_model=ship_model, speed_array_kn=speed_array_kn)


@pytest.fixture
def added_resistance_stawave2(ship_dimension):
    """Test fixture for the added resistance."""
    return AddedResistanceByStaWave2(
        ship_dimension=ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
    )


@pytest.fixture
def added_resistance_snnm(ship_dimension):
    """Test fixture for the added resistance."""
    return AddedResistanceBySNNM(
        ship_type=ShipType.ro_pax,
        ship_dimension=ship_dimension,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
        gamma=3.3,
    )


@pytest.fixture
def added_resistance_wind(ship_dimension):
    """Test fixture for the added resistance due to wind."""
    return AddedResistanceWindITTC(
        ship_type=ShipType.ro_pax,
        transverse_area_m2=ship_dimension.av_transverse_area_above_water_line_m2,
        is_laden=True,
    )


class ShipDimension(ShipDimensionsHollenbachTwinScrew, ShipDimensionsAddedResistance):
    pass


@pytest.fixture
def ship_dimension() -> ShipDimension:
    """Test fixture for the ship dimension."""
    return ShipDimension(
        lpp_length_between_perpendiculars_m=201.9,
        b_beam_m=26.7,
        los_length_over_surface_m=212,
        lwl_length_water_line_m=212,
        cb_block_coefficient=0.58,
        ta_draft_aft_m=6,
        tf_draft_forward_m=6,
        dp_diameter_propeller_m=4.5,
        has_bulb=False,
        number_rudders=2,
        av_transverse_area_above_water_line_m2=700,
    )


def test_propulsion_point_calculation(calm_water_resistance_hollenbach):
    """Test some failure cases of the propulsion point calculation."""
    propulsor = PropulsorDataBseries(
        pd_pitch_diameter_ratio=1.2,
        ear_blade_area_ratio=0.8,
        dp_diameter_propeller_m=4.5,
        z_blade_number=5,
    )

    ship_model = ShipModel(
        ship_description=ShipDescription(name="Test ship", type=ShipType.ro_pax),
        calm_water_resistance=calm_water_resistance_hollenbach,
        propulsor=propulsor,
    )

    speed_array_kn = np.linspace(1, 24, 24)
    ship_performance_data = ship_model.get_ship_performance_data_from_speed(
        vessel_speed_kn=speed_array_kn
    )
    resistance_kn = ship_performance_data.hull_data.total_resistance_newton
    propeller_thrust_kn = ship_performance_data.propeller_data.propeller_thrust_newton
    propeller_speed_rpm = ship_performance_data.propeller_data.n_rpm
    shaft_power_kw = ship_performance_data.propeller_data.shaft_power_kw
    assert np.all(resistance_kn > 0)
    assert np.all(propeller_speed_rpm > 0)
    assert np.all(propeller_thrust_kn > 0)
    assert np.all(shaft_power_kw > 0)


def test_zero_speed(
    calm_water_resistance,
    calm_water_resistance_hollenbach,
    machinery_system,
    ship_dimension,
):
    for vessel_speed_kn in [0, np.zeros(3)]:
        ship_model = ShipModel(
            calm_water_resistance=calm_water_resistance_hollenbach,
            propulsor=PropulsorDataScalar(efficiency=0.7),
            machinery_system=machinery_system,
        )
        ship_performance_data = ship_model.get_ship_performance_data_from_speed(
            vessel_speed_kn
        )
        assert np.all(
            np.atleast_1d(ship_performance_data.hull_data.total_resistance_newton) == 0
        )
        assert np.all(
            np.atleast_1d(ship_performance_data.propeller_data.propeller_thrust_newton)
            == 0
        )
        assert np.all(np.atleast_1d(ship_performance_data.propeller_data.n_rpm) == 0)
        assert np.all(
            np.atleast_1d(ship_performance_data.propeller_data.shaft_power_kw) == 0
        )

        ship_model.calm_water_resistance = calm_water_resistance
        ship_performance_data = ship_model.get_ship_performance_data_from_speed(
            vessel_speed_kn
        )
        assert np.all(
            np.atleast_1d(ship_performance_data.hull_data.total_resistance_newton) == 0
        )
        assert np.all(
            np.atleast_1d(ship_performance_data.propeller_data.propeller_thrust_newton)
            == 0
        )
        assert np.all(np.atleast_1d(ship_performance_data.propeller_data.n_rpm) == 0)
        assert np.all(
            np.atleast_1d(ship_performance_data.propeller_data.shaft_power_kw) == 0
        )


def test_added_resistance_all(
    calm_water_resistance_hollenbach,
    added_resistance_snnm,
    added_resistance_wind,
    propulsor_b_series,
    machinery_system,
):
    ship_model = ShipModel(
        calm_water_resistance=calm_water_resistance_hollenbach,
        added_resistance_wave=added_resistance_snnm,
        added_resistance_wind=added_resistance_wind,
        propulsor=propulsor_b_series,
    )
    ship_performance_data = ship_model.get_ship_performance_data_from_speed(
        vessel_speed_kn=14,
        weather=Weather(
            significant_wave_height_m=4,
            mean_wave_period_s=9,
            wave_direction_deg=30,
            wind_speed_m_per_s=10,
            wind_direction_deg=35,
        ),
        heading_deg=0,
        auxiliary_power_kw=1000,
    )
    assert ship_performance_data.hull_data.total_resistance_newton > 0
    assert ship_performance_data.hull_data.calm_water_resistance_newton > 0
    assert ship_performance_data.hull_data.added_resistance_wave_newton > 0
    assert ship_performance_data.hull_data.added_resistance_wind_newton > 0

    # Test with incomplete weather data
    ship_performance_data = ship_model.get_ship_performance_data_from_speed(
        vessel_speed_kn=14,
        weather=Weather(
            significant_wave_height_m=4,
            mean_wave_period_s=9,
        ),
        heading_deg=0,
        auxiliary_power_kw=1000,
    )
    assert ship_performance_data.hull_data.total_resistance_newton > 0
    assert ship_performance_data.hull_data.calm_water_resistance_newton > 0
    assert ship_performance_data.hull_data.added_resistance_wave_newton > 0

    # Test with zero values for weather
    ship_performance_data = ship_model.get_ship_performance_data_from_speed(
        vessel_speed_kn=14,
        weather=Weather(
            significant_wave_height_m=0,
            mean_wave_period_s=0,
            wave_direction_deg=0,
            wind_speed_m_per_s=0,
            wind_direction_deg=0,
        ),
        heading_deg=0,
        auxiliary_power_kw=1000,
    )
    assert ship_performance_data.hull_data.total_resistance_newton > 0
    assert ship_performance_data.hull_data.calm_water_resistance_newton > 0
    assert ship_performance_data.hull_data.added_resistance_wave_newton == 0


@pytest.fixture
def added_resistance_by_sea_margin_constant():
    return AddedResistanceBySeaMarginCurve(
        sea_margin_perc=np.array([15, 15]),
        significant_wave_height_m=np.array([0, 1000]),
    )


@pytest.fixture
def added_resistance_by_sea_margin_curve():
    significant_wave_heights = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    sea_margin_perc = significant_wave_heights**2 * 4
    return AddedResistanceBySeaMarginCurve(
        sea_margin_perc=sea_margin_perc,
        significant_wave_height_m=significant_wave_heights,
    )


def test_added_resistance_by_sea_margin(
    calm_water_resistance_hollenbach,
    added_resistance_by_sea_margin_constant,
    added_resistance_by_sea_margin_curve,
):
    """Test added resistance by sea margin"""
    ship_model = ShipModel(
        calm_water_resistance=calm_water_resistance_hollenbach,
        added_resistance_wave=added_resistance_by_sea_margin_constant,
    )
    ship_performance_data = ship_model.get_ship_performance_data_from_speed(
        vessel_speed_kn=14,
        weather=Weather(significant_wave_height_m=10 * random.random()),
    )
    assert np.allclose(
        ship_performance_data.hull_data.added_resistance_wave_newton,
        ship_performance_data.hull_data.calm_water_resistance_newton * 0.15,
    )

    wave_height = 10 * random.random()
    sea_margin_percent = added_resistance_by_sea_margin_curve.get_sea_margin_percent(
        significant_wave_height_m=wave_height
    )
    print(wave_height, sea_margin_percent)
    ship_model = ShipModel(
        calm_water_resistance=calm_water_resistance_hollenbach,
        added_resistance_wave=added_resistance_by_sea_margin_curve,
    )
    ship_performance_data = ship_model.get_ship_performance_data_from_speed(
        vessel_speed_kn=14, weather=Weather(significant_wave_height_m=wave_height)
    )
    assert np.allclose(
        ship_performance_data.hull_data.added_resistance_wave_newton,
        ship_performance_data.hull_data.calm_water_resistance_newton
        * sea_margin_percent
        / 100,
    )


# This part has been trasnferred from nbdev jupyters

@pytest.fixture
def vessel_speed_kn():
    return np.array([13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18])

@pytest.fixture
def wake_factor():
    return np.array([0.338, 0.337, 0.336, 0.335, 0.334, 0.332, 0.329, 0.328, 0.326, 0.325, 0.322])

@pytest.fixture
def thrust_deduction():
    return np.array([0.201, 0.205, 0.209, 0.214, 0.218, 0.22, 0.223, 0.224, 0.227, 0.229, 0.233])

@pytest.fixture
def propeller_diameter():
    return  9.81

@pytest.fixture
def pitch_diameter_ratio():
    return 0.781

@pytest.fixture
def wake_fraction_thrust_deduction_values(vessel_speed_kn, wake_factor, thrust_deduction):

    return [
         WakeFractionThrustDeductionFactorPoint(
             wake_fraction_factor=wake_factor_each,
             thrust_deduction_factor=thrust_deduction_each,
             vessel_speed_kn=vessel_speed_each,
         )
    for wake_factor_each, thrust_deduction_each, vessel_speed_each in zip(
        wake_factor, thrust_deduction, vessel_speed_kn
     )
    ]

@pytest.fixture
def propulsor_data_bseries(wake_fraction_thrust_deduction_values,
                           propeller_diameter,
                           pitch_diameter_ratio,
                           ):


    return PropulsorDataBseries(
        dp_diameter_propeller_m=propeller_diameter,
        pd_pitch_diameter_ratio=pitch_diameter_ratio,
        ear_blade_area_ratio=0.431,
        z_blade_number=4,
        wake_thrust_reduction=wake_fraction_thrust_deduction_values,
    )

@pytest.fixture
def propulsor_data_scalar():
    return PropulsorDataScalar(efficiency=0.7)

@pytest.fixture
def speed():
    return np.linspace(13, 18.0, 10)




ShipDimensionValues = namedtuple(
    "ShipDimensionValues",
    [
        "b_beam_m",
        "lpp_length_between_perpendiculars_m",
        "los_length_over_surface_m",
        "lwl_length_water_line_m",
        "cb_block_coefficient",
        "ta_draft_aft_m",
        "tf_draft_forward_m",
        "wetted_surface_m2",
        "av_transverse_area_above_water_line_m2",
        "area_bilge_keel_m2",
        "kyy_radius_gyration_in_lateral_direction_non_dim",
        "propeller_diameter"
    ]
)

@pytest.fixture(scope="session")
def ship_dimensions():
    return ShipDimensionValues(
        b_beam_m=24,
        lpp_length_between_perpendiculars_m=145,
        los_length_over_surface_m=150,
        lwl_length_water_line_m=146.7,
        cb_block_coefficient=0.75,
        ta_draft_aft_m=8.2,
        tf_draft_forward_m=8.2,
        wetted_surface_m2=4400,
        av_transverse_area_above_water_line_m2=2,
        area_bilge_keel_m2=1,
        kyy_radius_gyration_in_lateral_direction_non_dim=0.26,
        propeller_diameter=9.81,  # Add other parameters as needed
    )


@pytest.fixture
def ship_dimensions_single_screw(ship_dimensions):
    return ShipDimensionsHollenbachSingleScrew(
        b_beam_m=ship_dimensions.b_beam_m,
        lpp_length_between_perpendiculars_m=ship_dimensions.lpp_length_between_perpendiculars_m,
        los_length_over_surface_m=ship_dimensions.los_length_over_surface_m,
        lwl_length_water_line_m=ship_dimensions.lwl_length_water_line_m,
        cb_block_coefficient=ship_dimensions.cb_block_coefficient,
        ta_draft_aft_m=ship_dimensions.ta_draft_aft_m,
        tf_draft_forward_m=ship_dimensions.tf_draft_forward_m,
        dp_diameter_propeller_m=ship_dimensions.propeller_diameter,
        wetted_surface_m2=ship_dimensions.wetted_surface_m2,
        av_transverse_area_above_water_line_m2=ship_dimensions.av_transverse_area_above_water_line_m2,
        area_bilge_keel_m2=ship_dimensions.area_bilge_keel_m2,
    )

@pytest.fixture
def  ship_dimensions_stawave2 (ship_dimensions):
    return   ShipDimensionsAddedResistance(
        b_beam_m=ship_dimensions.b_beam_m,
        lpp_length_between_perpendiculars_m=ship_dimensions.lpp_length_between_perpendiculars_m,
        cb_block_coefficient=ship_dimensions.cb_block_coefficient,
        ta_draft_aft_m=ship_dimensions.ta_draft_aft_m,
        tf_draft_forward_m=ship_dimensions.tf_draft_forward_m,
        kyy_radius_gyration_in_lateral_direction_non_dim=0.26,
    )

@pytest.fixture
def added_resistance_stawave2_newton(ship_dimensions_stawave2):
    return AddedResistanceByStaWave2(
    ship_dimension=ship_dimensions_stawave2,
    wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
)

@pytest.fixture
def weather():
    return  Weather(
        significant_wave_height_m=3,
        mean_wave_period_s=10,
        wave_direction_deg=180,
        wind_speed_m_per_s=10,
        wind_direction_deg=160,
        ocean_current_speed_m_per_s=3,
        ocean_current_direction_deg=10,
    )


@pytest.fixture
def fuel ():
    return FuelByMassFraction(hfo=1)
@pytest.fixture
def vessel_speed():
    return 16.0
@pytest.fixture
def rated_power_kw():
    return 10000.0
@pytest.fixture
def rated_power_aux_kw():
    return 2000.0
@pytest.fixture
def scalar_efficiency():
    return 0.5
@pytest.fixture
def scalar_efficiency_electric():
    return 0.5
@pytest.fixture
def mechanical_system_efficiency():
    return 0.95
@pytest.fixture
def electric_system_efficiency():
    return 0.95
@pytest.fixture
def efficiency_curve():
    return Curve(
        points=[
            Point(x=0.25, y=0.3),
            Point(x=0.5, y=0.4),
            Point(x=0.75, y=0.5),
            Point(x=1.0, y=0.35),
        ]
    )

@pytest.fixture
def emission_factors_scalar(rated_power_kw):
    factor_no_x_scalar = EmissionFactor(
        emission_type=EmissionType("nox"), factor=1.0, rated_power_kw=rated_power_kw
    )

    factor_co_2_scalar = EmissionFactor(
        emission_type=EmissionType("co2"), factor=2.0, rated_power_kw=rated_power_kw
    )

    factor_so_x_scalar = EmissionFactor(
        emission_type=EmissionType("sox"), factor=3.0, rated_power_kw=rated_power_kw
    )
    factor_pm_scalar = EmissionFactor(
        emission_type=EmissionType("pm"), factor=4.0, rated_power_kw=rated_power_kw
    )

    emission_factors_scalar = [
        factor_co_2_scalar,
        factor_no_x_scalar,
        factor_so_x_scalar,
        factor_pm_scalar,
    ]
    return emission_factors_scalar


@pytest.fixture
def emission_factors_curve(fuel,
                           efficiency_curve,
                           rated_power_kw):
    nox_curve = Curve(
        [Point(x=0.25, y=6), Point(x=0.5, y=10), Point(x=0.75, y=12), Point(x=1, y=20)]
    )
    factor_no_x_curve = EmissionFactor(
        emission_type=EmissionType("nox"), factor=nox_curve, rated_power_kw=rated_power_kw
    )

    co2_curve = fuel.get_co_2_curve(efficiency=efficiency_curve)
    factor_co_2_curve = EmissionFactor(
        emission_type=EmissionType("co2"), factor=co2_curve, rated_power_kw=rated_power_kw
    )

    so_x_curve = Curve(
        [Point(x=0.25, y=10), Point(x=0.5, y=7), Point(x=0.75, y=5), Point(x=1, y=8)]
    )
    factor_so_x_curve = EmissionFactor(
        emission_type=EmissionType("sox"), factor=so_x_curve, rated_power_kw=rated_power_kw
    )

    pm_curve = Curve(
        [
            Point(x=0.25, y=0.2),
            Point(x=0.5, y=0.15),
            Point(x=0.75, y=0.1),
            Point(x=1, y=0.1),
        ]
    )
    factor_pm_curve = EmissionFactor(
        emission_type=EmissionType("pm"), factor=pm_curve, rated_power_kw=rated_power_kw
    )

    emission_factors_curve = [
        factor_co_2_curve,
        factor_no_x_curve,
        factor_so_x_curve,
        factor_pm_curve,
    ]
    return emission_factors_curve




@pytest.fixture
def machinery_system_cases(fuel,
                     rated_power_kw,
                     efficiency_curve,
                     scalar_efficiency,
                     rated_power_aux_kw,
                     scalar_efficiency_electric,
                     emission_factors_scalar,
                     emission_factors_curve,
                     electric_system_efficiency,
                     mechanical_system_efficiency):
    power_source_mechanical_system_scalar_no_emissions = PowerSourceWithEfficiency(
        fuel=fuel, efficiency=scalar_efficiency, rated_power_kw=rated_power_kw
    )
    power_source_mechanical_system_curve_no_emissions = PowerSourceWithEfficiency(
        fuel=fuel, efficiency=efficiency_curve, rated_power_kw=rated_power_kw
    )
    aux_power_source_electric_system_scalar_no_emissions = PowerSourceWithEfficiency(
        fuel=fuel, efficiency=scalar_efficiency_electric, rated_power_kw=rated_power_aux_kw
    )
    aux_power_source_electric_system_curve_no_emissions = PowerSourceWithEfficiency(
        fuel=fuel, efficiency=efficiency_curve, rated_power_kw=rated_power_aux_kw
    )
    power_source_mechanical_system_scalar_emissions_scalar = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=scalar_efficiency,
        rated_power_kw=rated_power_kw,
        emission_factors=emission_factors_scalar,
    )
    power_source_mechanical_system_curve_emissions_scalar = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency_curve,
        rated_power_kw=rated_power_kw,
        emission_factors=emission_factors_scalar,
    )
    aux_power_source_electric_system_scalar_emissions_scalar = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=scalar_efficiency_electric,
        rated_power_kw=rated_power_aux_kw,
        emission_factors=emission_factors_scalar,
    )
    aux_power_source_electric_system_curve_emissions_scalar = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency_curve,
        rated_power_kw=rated_power_aux_kw,
        emission_factors=emission_factors_scalar,
    )
    power_source_mechanical_system_scalar_emissions_curve = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=scalar_efficiency,
        rated_power_kw=rated_power_kw,
        emission_factors=emission_factors_curve,
    )
    power_source_mechanical_system_curve_emissions_curve = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency_curve,
        rated_power_kw=rated_power_kw,
        emission_factors=emission_factors_curve,
    )
    aux_power_source_electric_system_scalar_emissions_curve = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=scalar_efficiency_electric,
        rated_power_kw=rated_power_aux_kw,
        emission_factors=emission_factors_curve,
    )
    aux_power_source_electric_system_curve_emissions_curve = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency_curve,
        rated_power_kw=rated_power_aux_kw,
        emission_factors=emission_factors_curve,
    )
    main_power_source_electric_system_scalar_no_emissions = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=scalar_efficiency_electric,
        rated_power_kw=rated_power_kw + rated_power_aux_kw,
    )
    main_power_source_electric_system_curve_no_emissions = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency_curve,
        rated_power_kw=rated_power_kw + rated_power_aux_kw,
    )
    main_power_source_electric_system_scalar_emissions_scalar = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=scalar_efficiency_electric,
        rated_power_kw=rated_power_kw + rated_power_aux_kw,
        emission_factors=emission_factors_scalar,
    )
    main_power_source_electric_system_curve_emissions_scalar = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency_curve,
        rated_power_kw=rated_power_kw + rated_power_aux_kw,
        emission_factors=emission_factors_scalar,
    )
    main_power_source_electric_system_scalar_emissions_curve = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=scalar_efficiency_electric,
        rated_power_kw=rated_power_kw + rated_power_aux_kw,
        emission_factors=emission_factors_curve,
    )
    main_power_source_electric_system_curve_emissions_curve = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency_curve,
        rated_power_kw=rated_power_kw + rated_power_aux_kw,
        emission_factors=emission_factors_curve,
    )

    mechanical_load = PowerLoad(efficiency=mechanical_system_efficiency)
    electric_load = PowerLoad(efficiency=electric_system_efficiency)

    mechanical_system_scalar_no_emissions = MachinerySubsystemSimple(
        power_source=power_source_mechanical_system_scalar_no_emissions,
        propulsion_load=mechanical_load,
    )
    aux_electric_system_scalar_no_emissions = MachinerySubsystemSimple(
        power_source=aux_power_source_electric_system_scalar_no_emissions,
        auxiliary_load=electric_load,
    )
    mechanical_system_curve_no_emissions = MachinerySubsystemSimple(
        power_source=power_source_mechanical_system_curve_no_emissions,
        propulsion_load=mechanical_load,
    )
    aux_electric_system_curve_no_emissions = MachinerySubsystemSimple(
        power_source=aux_power_source_electric_system_curve_no_emissions,
        auxiliary_load=electric_load,
    )
    mechanical_system_scalar_emissions_scalar = MachinerySubsystemSimple(
        power_source=power_source_mechanical_system_scalar_emissions_scalar,
        propulsion_load=mechanical_load,
    )
    aux_electric_system_scalar_emissions_scalar = MachinerySubsystemSimple(
        power_source=aux_power_source_electric_system_scalar_emissions_scalar,
        auxiliary_load=electric_load,
    )
    mechanical_system_scalar_emissions_curve = MachinerySubsystemSimple(
        power_source=power_source_mechanical_system_scalar_emissions_curve,
        propulsion_load=mechanical_load,
    )
    aux_electric_system_scalar_emissions_curve = MachinerySubsystemSimple(
        power_source=aux_power_source_electric_system_scalar_emissions_curve,
        auxiliary_load=electric_load,
    )
    mechanical_system_curve_emissions_scalar = MachinerySubsystemSimple(
        power_source=power_source_mechanical_system_curve_emissions_scalar,
        propulsion_load=mechanical_load,
    )
    aux_electric_system_curve_emissions_scalar = MachinerySubsystemSimple(
        power_source=aux_power_source_electric_system_curve_emissions_scalar,
        auxiliary_load=electric_load,
    )
    mechanical_system_curve_emissions_curve = MachinerySubsystemSimple(
        power_source=power_source_mechanical_system_curve_emissions_curve,
        propulsion_load=mechanical_load,
    )
    aux_electric_system_curve_emissions_curve = MachinerySubsystemSimple(
        power_source=aux_power_source_electric_system_curve_emissions_curve,
        auxiliary_load=electric_load,
    )

    main_electric_system_scalar_no_emissions = MachinerySubsystemSimple(
        power_source=main_power_source_electric_system_scalar_no_emissions,
        propulsion_load=electric_load,
        auxiliary_load=electric_load,
    )

    main_electric_system_curve_no_emissions = MachinerySubsystemSimple(
        power_source=main_power_source_electric_system_curve_no_emissions,
        propulsion_load=electric_load,
        auxiliary_load=electric_load,
    )

    main_electric_system_scalar_emissions_scalar = MachinerySubsystemSimple(
        power_source=main_power_source_electric_system_scalar_emissions_scalar,
        propulsion_load=electric_load,
        auxiliary_load=electric_load,
    )

    main_electric_system_curve_emissions_scalar = MachinerySubsystemSimple(
        power_source=main_power_source_electric_system_curve_emissions_scalar,
        propulsion_load=electric_load,
        auxiliary_load=electric_load,
    )

    main_electric_system_scalar_emissions_curve = MachinerySubsystemSimple(
        power_source=main_power_source_electric_system_scalar_emissions_curve,
        propulsion_load=electric_load,
        auxiliary_load=electric_load,
    )

    main_electric_system_curve_emissions_curve = MachinerySubsystemSimple(
        power_source=main_power_source_electric_system_curve_emissions_curve,
        propulsion_load=electric_load,
        auxiliary_load=electric_load,
    )

    machinery_system_scalar_no_emissions = MachinerySystem(
        propulsion_type=PropulsionType.MECHANICAL,
        mechanical_system=mechanical_system_scalar_no_emissions,
        electric_system=aux_electric_system_scalar_no_emissions,
    )
    machinery_system_curve_no_emissions = MachinerySystem(
        propulsion_type=PropulsionType.MECHANICAL,
        mechanical_system=mechanical_system_curve_no_emissions,
        electric_system=aux_electric_system_scalar_no_emissions,
    )
    machinery_system_scalar_emissions_scalar = MachinerySystem(
        propulsion_type=PropulsionType.MECHANICAL,
        mechanical_system=mechanical_system_scalar_emissions_scalar,
        electric_system=aux_electric_system_scalar_emissions_scalar,
    )
    machinery_system_curve_emissions_scalar = MachinerySystem(
        propulsion_type=PropulsionType.MECHANICAL,
        mechanical_system=mechanical_system_curve_emissions_scalar,
        electric_system=aux_electric_system_curve_emissions_scalar,
    )
    machinery_system_scalar_emissions_curve = MachinerySystem(
        propulsion_type=PropulsionType.MECHANICAL,
        mechanical_system=mechanical_system_scalar_emissions_curve,
        electric_system=aux_electric_system_scalar_emissions_curve,
    )
    machinery_system_curve_emissions_curve = MachinerySystem(
        propulsion_type=PropulsionType.MECHANICAL,
        mechanical_system=mechanical_system_curve_emissions_curve,
        electric_system=aux_electric_system_curve_emissions_curve,
    )

    machinery_system_electric_scalar_no_emissions = MachinerySystem(
        propulsion_type=PropulsionType.ELECTRIC,
        electric_system=main_electric_system_scalar_no_emissions,
    )

    machinery_system_electric_curve_no_emissions = MachinerySystem(
        propulsion_type=PropulsionType.ELECTRIC,
        electric_system=main_electric_system_curve_no_emissions,
    )

    machinery_system_electric_scalar_emissions_scalar = MachinerySystem(
        propulsion_type=PropulsionType.ELECTRIC,
        electric_system=main_electric_system_scalar_emissions_scalar,
    )

    machinery_system_electric_curve_emissions_scalar = MachinerySystem(
        propulsion_type=PropulsionType.ELECTRIC,
        electric_system=main_electric_system_curve_emissions_scalar,
    )

    machinery_system_electric_scalar_emissions_curve = MachinerySystem(
        propulsion_type=PropulsionType.ELECTRIC,
        electric_system=main_electric_system_scalar_emissions_curve,
    )

    machinery_system_electric_curve_emissions_curve = MachinerySystem(
        propulsion_type=PropulsionType.ELECTRIC,
        electric_system=main_electric_system_curve_emissions_curve,
    )

    machinery_system = [
        machinery_system_scalar_no_emissions,
        machinery_system_curve_no_emissions,
        machinery_system_scalar_emissions_scalar,
        machinery_system_curve_emissions_scalar,
        machinery_system_scalar_emissions_curve,
        machinery_system_curve_emissions_curve,
        machinery_system_electric_scalar_no_emissions,
        machinery_system_electric_curve_no_emissions,
        machinery_system_electric_scalar_emissions_scalar,
        machinery_system_electric_curve_emissions_scalar,
        machinery_system_electric_scalar_emissions_curve,
        machinery_system_electric_curve_emissions_curve,
    ]
    return machinery_system

@pytest.fixture
def ship_description():
    return ShipDescription(name="Test vessel",
                           type=ShipType.bulk_handysize)

@pytest.fixture
def calm_water_resistance_hddss_kilo_newton(ship_dimensions_single_screw,
                                            speed):

    return CalmWaterResistanceHollenbachSingleScrewDesignDraft(
            ship_dimensions=ship_dimensions_single_screw)

@pytest.fixture
def test_speeds_kn():
    return np.linspace(13.0, 17.4, 10)



def test_calm_water_resistance_Hollenbach_single_screw_design_draft(ship_dimensions_single_screw):
    CalmWaterResistanceHollenbachSingleScrewDesignDraft (
        ship_dimensions=ship_dimensions_single_screw
    )

def test_calm_water_resistance_Hollenbach_single_screw_ballast_draft(ship_dimensions_single_screw):
    CalmWaterResistanceHollenbachSingleScrewBallastDraft(
        ship_dimensions=ship_dimensions_single_screw
    )

def test_calm_water_resistance_Hollenbach_twin_screw_design_draft():
    ship_dimensions_twin_screw = ShipDimensionsHollenbachTwinScrew(
        b_beam_m=24,
        lpp_length_between_perpendiculars_m=145,
        los_length_over_surface_m=150,
        lwl_length_water_line_m=146.7,
        cb_block_coefficient=0.75,
        ta_draft_aft_m=8.2,
        tf_draft_forward_m=8.2,
        wetted_surface_m2=4400,
        av_transverse_area_above_water_line_m2=2,
        area_bilge_keel_m2=1,
        dp_diameter_propeller_m=propeller_diameter,
    )

    CalmWaterResistanceHollenbachTwinScrewDesignDraft(
            ship_dimensions=ship_dimensions_twin_screw
    )

def test_calm_water_resistance_by_speed_resistance_curve(ship_dimensions_single_screw,speed):


    calm_water_resistance_hddss_kilo_newton = CalmWaterResistanceHollenbachSingleScrewDesignDraft(
        ship_dimensions=ship_dimensions_single_screw)
    resistance_array = calm_water_resistance_hddss_kilo_newton.get_resistance_from_speed(speed)
    CalmWaterResistanceBySpeedResistanceCurve(
        speed_ref_kn=speed,
        resistance_ref_k_n=resistance_array
    )


def test_calm_water_resistance_by_speed_power_curve(
    ship_dimensions_single_screw,
    speed,
    propeller_diameter,
    pitch_diameter_ratio,
    wake_fraction_thrust_deduction_values
):
    propulsor_data_bseries = PropulsorDataBseries(
        dp_diameter_propeller_m=propeller_diameter,
        pd_pitch_diameter_ratio=pitch_diameter_ratio,
        ear_blade_area_ratio=0.431,
        z_blade_number=4,
        wake_thrust_reduction=wake_fraction_thrust_deduction_values,
    )
    calm_water_resistance = CalmWaterResistanceHollenbachSingleScrewDesignDraft(
        ship_dimensions=ship_dimensions_single_screw
    )

    resistance = calm_water_resistance.get_resistance_from_speed(speed) * 1000
    propulsor_output = propulsor_data_bseries.get_propulsor_data_from_vessel_speed_thrust(
        vessel_speed_kn=speed,
        thrust_resistance_newton=resistance
    )
    power = propulsor_output.shaft_power_kw

    CalmWaterResistanceBySpeedPowerCurve(
        speed_ref_kn=speed,
        power_ref_kw=power
    )

def test_added_resistance_by_StaWave2 (ship_dimensions_stawave2):
    AddedResistanceByStaWave2(
        ship_dimension=ship_dimensions_stawave2,
        wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC_1984,
    )

def test_ship_model_get_power_from_speed (
        machinery_system_cases,
        added_resistance_stawave2_newton,
        vessel_speed,
        propulsor_data_bseries,
        weather,
        ship_description,
        calm_water_resistance_hddss_kilo_newton
    ):


    for index, machinery in enumerate(machinery_system_cases):
        ship_model_machinery_system = ShipModel(
            ship_description=ship_description,
            calm_water_resistance=calm_water_resistance_hddss_kilo_newton,
            added_resistance_wave=added_resistance_stawave2_newton,
            propulsor=propulsor_data_bseries,
            machinery_system=machinery,
        )
    ship_model_machinery_system.get_ship_performance_data_from_speed(
            vessel_speed_kn=vessel_speed, weather=weather, auxiliary_power_kw=500
    )


def test_comparing_get_ship_data_from_speed_with_get_ship_performance_data_from_power(
        machinery_system_cases,
        test_speeds_kn,
        calm_water_resistance_hddss_kilo_newton,
        added_resistance_stawave2_newton,
        propulsor_data_bseries,
        weather
):
    ship_description = ShipDescription(name="Test vessel", type=ShipType.bulk_handysize)

    for index, machinery in enumerate(machinery_system_cases):
        for speed_kn in test_speeds_kn:
            ship_model_machinery_simple = ShipModel(
                ship_description=ship_description,
                calm_water_resistance=calm_water_resistance_hddss_kilo_newton,
                added_resistance_wave=added_resistance_stawave2_newton,
                propulsor=propulsor_data_bseries,
                machinery_system=machinery,
            )
            result_speed = ship_model_machinery_simple.get_ship_performance_data_from_speed(
                vessel_speed_kn=speed_kn, weather=weather, auxiliary_power_kw=0
            )
            result_power = ship_model_machinery_simple.get_ship_performance_data_from_power(
                power_out_source_kw=result_speed.power_source_data.total.power_on_source_kw,
                weather=weather,
                auxiliary_power_kw=0,
            )
            assert np.isclose(
                result_speed.propeller_data.n_rpm,
                result_power.propeller_data.n_rpm,
                rtol=1e-3,
            ), (
                f"Speed calculated propeller rpm {result_speed.propeller_data.n_rpm}"
                f" is not equal to power calculated propeller rpm {result_power.propeller_data.n_rpm} "
                f"for speed {speed_kn} with machinery {machinery.propulsion_type.name},"
                f"propeller data speed result speed {result_speed.propeller_data.vessel_speed_kn}, "
                f"power result speed {result_power.propeller_data.vessel_speed_kn},"
                f"propeller data speed result speed {result_speed.propeller_data.shaft_power_kw},"
                f" power result speed {result_power.propeller_data.shaft_power_kw},"
                f"Speed result: mechanical power {result_speed.power_source_data.mechanical_system.power_on_source_kw} "
                f"electric power {result_speed.power_source_data.electric_system.power_on_source_kw} "
                f"total power {result_speed.power_source_data.total.power_on_source_kw},"
            )



def test_ship_model_only_calm_water_input(calm_water_resistance_hddss_kilo_newton):
    ShipModel(calm_water_resistance=calm_water_resistance_hddss_kilo_newton)

def test_ship_model_calm_water_added_resistance_input (
    calm_water_resistance_hddss_kilo_newton,
    added_resistance_stawave2_newton
):
    ShipModel(
        calm_water_resistance=calm_water_resistance_hddss_kilo_newton,
        added_resistance_wave=added_resistance_stawave2_newton,
    )


def test_to_evaluate_array_inputs_to_shipmodel (
        weather,
        test_speeds_kn,
        machinery_system_cases,
        calm_water_resistance_hddss_kilo_newton,
        propulsor_data_bseries
):
    ones_array = np.ones_like(test_speeds_kn)
    ship_description = ShipDescription(name="Test vessel", type=ShipType.bulk_handysize)
    weather_array = Weather(
        significant_wave_height_m=ones_array * weather.significant_wave_height_m,
        mean_wave_period_s=ones_array * weather.mean_wave_period_s,
        wave_direction_deg=ones_array * weather.wave_direction_deg,
        wind_speed_m_per_s=ones_array * weather.wind_speed_m_per_s,
        wind_direction_deg=ones_array * weather.wave_direction_deg,
        ocean_current_speed_m_per_s=ones_array * weather.ocean_current_speed_m_per_s,
        ocean_current_direction_deg=ones_array * weather.ocean_current_direction_deg,
    )
    electric_power_list = np.array([500] * len(test_speeds_kn))
    for index, machinery in enumerate(machinery_system_cases):
        ship_model_machinery_simple = ShipModel(
            ship_description=ship_description,
            calm_water_resistance=calm_water_resistance_hddss_kilo_newton,
            propulsor=propulsor_data_bseries,
            machinery_system=machinery,
        )
        result_speed = ship_model_machinery_simple.get_ship_performance_data_from_speed(
            vessel_speed_kn=test_speeds_kn,
            weather=weather_array,
            auxiliary_power_kw=electric_power_list,
        )
        result_power = ship_model_machinery_simple.get_ship_performance_data_from_power(
            power_out_source_kw=result_speed.power_source_data.total.power_on_source_kw,
            weather=weather_array,
            auxiliary_power_kw=electric_power_list,
        )
        assert np.allclose(
            result_speed.propeller_data.vessel_speed_kn,
            result_power.propeller_data.vessel_speed_kn,
            rtol=1e-25,
        ), (f"The results are not the same for propeller speed: {result_speed.propeller_data.vessel_speed_kn} vs "
            f"{result_power.propeller_data.vessel_speed_kn}")


def test_ship_model_with_mechanical_machinery_system_and_propeller(
        machinery_system_cases,
        calm_water_resistance_hddss_kilo_newton,
        added_resistance_stawave2_newton,
        propulsor_data_bseries,
        weather,
):
    power_limit = 6000.0
    power_limit_test_speeds = np.linspace(8.0, 17.4, 10)
    ship_description = ShipDescription(name="Test vessel", type=ShipType.bulk_handysize)

    for machinery in machinery_system_cases:
        ship_model_machinery_simple_power_limit_test = ShipModel(
            ship_description=ship_description,
            calm_water_resistance=calm_water_resistance_hddss_kilo_newton,
            added_resistance_wave=added_resistance_stawave2_newton,
            propulsor=propulsor_data_bseries,
            machinery_system=machinery,
        )

        speed_limited_power = ship_model_machinery_simple_power_limit_test.get_ship_performance_data_from_power(
            power_out_source_kw=power_limit, weather=weather, auxiliary_power_kw=0
        )

        for speed_kn in power_limit_test_speeds:
            operation_point = OperationPoint(
                speed_kn=speed_kn,
                power_limit_kw=power_limit,
                auxiliary_power=0,
                weather=weather,
            )
            result_power = ship_model_machinery_simple_power_limit_test.get_ship_performance_data_from_operating_point(
                operation_point=operation_point
            )
            # pprint(f"Speed setpoint: {speed_kn} - Speed achieved: {result_power.hull_data.vessel_speed_kn} - Power: {result_power.power_source_data.total.power_on_source_kw}")
            assert np.less_equal(
                result_power.power_source_data.total.power_on_source_kw, power_limit + 0.1
            ), f"Power achieved {result_power.power_source_data.total.power_on_source_kw} is not equal or less than the power limited {power_limit}"
            assert np.less_equal(
                result_power.hull_data.vessel_speed_kn,
                speed_limited_power.hull_data.vessel_speed_kn,
            ), f"Speed achieved {result_power.hull_data.vessel_speed_kn} is not equal to speed limited {speed_limited_power.hull_data.vessel_speed_kn}"


def test_ship_model_without_machinery_system (
        calm_water_resistance_hddss_kilo_newton,
        added_resistance_stawave2_newton,
        propulsor_data_bseries,
        weather,

):
    power_limit = 6000.0
    power_limit_test_speeds = np.linspace(8.0, 17.4, 10)
    ship_description = ShipDescription(name="Test vessel", type=ShipType.bulk_handysize)

    ship_model_machinery_simple_power_limit_test = ShipModel(
        ship_description=ship_description,
        calm_water_resistance=calm_water_resistance_hddss_kilo_newton,
        added_resistance_wave=added_resistance_stawave2_newton,
        propulsor=propulsor_data_bseries,
        machinery_system=None,
    )

    speed_limited_power = (
        ship_model_machinery_simple_power_limit_test.get_ship_performance_data_from_power(
            power_out_source_kw=power_limit, weather=weather, auxiliary_power_kw=0
        )
    )

    for speed_kn in power_limit_test_speeds:
        operation_point = OperationPoint(
            speed_kn=speed_kn,
            power_limit_kw=power_limit,
            auxiliary_power=0,
            weather=weather,
        )
        result_power = ship_model_machinery_simple_power_limit_test.get_ship_performance_data_from_operating_point(
            operation_point=operation_point
        )
        # pprint(f"Speed setpoint: {speed_kn} - Speed achieved: {result_power.hull_data.vessel_speed_kn} - Power: {result_power.propeller_data.shaft_power_kw}")
        assert np.less_equal(
            result_power.propeller_data.shaft_power_kw, power_limit + 0.1
        ), f"Power achieved {result_power.propeller_data.shaft_power_kw} is not equal or less than the power limited {power_limit}"
        assert np.less_equal(
            result_power.hull_data.vessel_speed_kn,
            speed_limited_power.hull_data.vessel_speed_kn,
        ), f"Speed achieved {result_power.hull_data.vessel_speed_kn} is not equal to speed limited {speed_limited_power.hull_data.vessel_speed_kn}"


def test_ship_model_get_performance_data_from_operating_point (
        calm_water_resistance_hddss_kilo_newton,
        rated_power_kw,
        weather,

):
    file_path = os.path.join(os.path.dirname(__file__), "..", "test_data", "df_filtered_voyages.pkl")
    file_path = os.path.abspath(file_path)  # Optional but recommended

    df_voyage = pd.read_pickle(file_path)
    voyage_list = df_voyage.iloc[0]["voyage_track_geometry"]

    time_stamp = voyage_list[0][1]
    time_object = datetime.strptime(time_stamp, "%Y-%m-%dT%H:%M:%S")

    one_operation_point = OperationPoint(
        timestamp_seconds=time_object.timestamp(),
        location=Location(latitude=voyage_list[0][3], longitude=voyage_list[0][2]),
        heading_deg=voyage_list[0][4],
        speed_kn=voyage_list[0][5],
        power_limit_kw=rated_power_kw,
        weather=weather,
        auxiliary_power=0.0
    )
    operation_list = []
    for i in range(1, len(df_voyage) + 1):
        voyage_list = df_voyage.iloc[i - 1]["voyage_track_geometry"]
        time_stamp = np.array(
            [
                datetime.strptime(point[1], "%Y-%m-%dT%H:%M:%S").timestamp()
                for point in voyage_list
            ]
        )
        longitude = np.array([point[2] for point in voyage_list])
        latitude = np.array([point[3] for point in voyage_list])
        heading = np.array([point[4] for point in voyage_list])
        speed = np.array([point[5] for point in voyage_list])
        location = Location(latitude=latitude, longitude=longitude)

        operation = OperationPoint(
            timestamp_seconds=time_stamp,
            location=location,
            heading_deg=heading,
            speed_kn=speed,
            power_limit_kw=rated_power_kw,
            weather=weather,
            auxiliary_power=0.0
        )
        operation_list.append(operation)
    operation_points_array = operation_list[0]

    ship_model_operation_point_hull_only = ShipModel(
        calm_water_resistance=calm_water_resistance_hddss_kilo_newton,
    )

    result_operation_point_hull_only_single_point = (
        ship_model_operation_point_hull_only.get_ship_performance_data_from_operating_point(
            operation_point=one_operation_point
        )
    )
    ship_model_operation_point_hull_only = ShipModel(
        calm_water_resistance=calm_water_resistance_hddss_kilo_newton,
    )

    result_operation_point_hull_only_array = (
        ship_model_operation_point_hull_only.get_ship_performance_data_from_operating_point(
            operation_point=operation_points_array
        )
    )
    # pprint(result_operation_point_hull_only_array)

