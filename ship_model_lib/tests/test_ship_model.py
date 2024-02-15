import random
import numpy as np
import pytest
from operation_profile_lib.operation_profile_structure import Weather

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
)
from test_machinery import get_machinery_system_nodel
from test_added_resistance import ship_dimension


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
    ship_model = ShipModel(
        calm_water_resistance=calm_water_resistance,
        propulsor=PropulsorDataScalar(efficiency=0.7),
        machinery_system=machinery_system,
    )
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
