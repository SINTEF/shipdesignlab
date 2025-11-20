import numpy as np
import pytest
import random
from scipy.interpolate import interp1d

from ship_model_lib.machinery import (
    MachinerySystem,
    MachinerySystemResult,
    MachinerySubsystemSimple,
    PowerSourceWithEfficiency,
    PowerSourceWithSpecificFuelConsumption,
    FuelByMassFraction,
    PowerLoad,
    LoadInput,
    PropulsionType,
    Curve,
    Point,
    FuelConsumption,
    EmissionType,
    EmissionFactor,
    MachineryResult,
    Emissions,
)


def get_machinery_system_nodel(
    propulsion_type: PropulsionType,
    efficiency_propulsion_drive: float,
    efficiency_power_source: float,
    efficiency_auxiliary_load: float,
    rated_power_source_kw: float,
    rated_power_auxiliary_kw: float = None,
) -> MachinerySystem:
    """Get a machinery system model for the test."""
    mechanical_system = (
        MachinerySubsystemSimple(
            power_source=PowerSourceWithEfficiency(
                fuel=FuelByMassFraction(hydrogen=1.0),
                efficiency=efficiency_power_source,
                rated_power_kw=rated_power_source_kw,
            ),
            propulsion_load=PowerLoad(efficiency=efficiency_propulsion_drive),
        )
        if propulsion_type == PropulsionType.MECHANICAL
        else None
    )
    electric_system = MachinerySubsystemSimple(
        power_source=PowerSourceWithEfficiency(
            fuel=FuelByMassFraction(hydrogen=1.0),
            efficiency=efficiency_power_source,
            rated_power_kw=(
                rated_power_source_kw
                if propulsion_type == PropulsionType.ELECTRIC
                else rated_power_auxiliary_kw
            ),
        ),
        propulsion_load=(
            PowerLoad(efficiency=efficiency_propulsion_drive)
            if propulsion_type == PropulsionType.ELECTRIC
            else None
        ),
        auxiliary_load=PowerLoad(efficiency=efficiency_auxiliary_load),
    )

    return MachinerySystem(
        propulsion_type=propulsion_type,
        mechanical_system=mechanical_system,
        electric_system=electric_system,
    )


def test_power_load_with_scalar_efficiency():
    """Test PowerLoad class with scalar efficiency."""
    # Test the scalar efficiency
    efficiency = 0.99
    power_load = PowerLoad(efficiency=efficiency)
    power_out_kw = np.linspace(0, 20000, 21)
    power_in_kw = power_load.get_power_in_kw(power_out_kw)
    assert power_in_kw == pytest.approx(power_out_kw / efficiency)


def test_power_load_with_efficiency_curve():
    """Test PowerLoad class with efficiency curve."""
    # Test the efficiency curve
    efficiency_array = np.linspace(0.5, 0.99, 21)
    load_array = np.linspace(0, 1, 21)
    efficiency_curve = Curve()
    for load, efficiency in zip(load_array, efficiency_array):
        efficiency_curve.add_point(Point(x=load, y=efficiency))
    with pytest.raises(ValueError):
        PowerLoad(efficiency=efficiency_curve)
    rated_power_kw = 10000.0
    power_load = PowerLoad(efficiency=efficiency_curve, rated_power_kw=rated_power_kw)
    power_out_kw = np.linspace(0, rated_power_kw, 21)
    power_in_kw = power_load.get_power_in_kw(power_out_kw)
    assert power_in_kw == pytest.approx(power_out_kw / efficiency_array)


def test_get_machinery_result_with_mechanical_propulsion():
    """Test get_machinery_result in MachinerySystem class."""
    # Create a machinery system
    efficiency_propulsion_drive = 0.99
    auxiliary_load = 1000.0
    propulsion_load_kw = np.linspace(0, 20000, 21)
    machinery_system = get_machinery_system_nodel(
        propulsion_type=PropulsionType.MECHANICAL,
        efficiency_propulsion_drive=efficiency_propulsion_drive,
        efficiency_power_source=0.45,
        efficiency_auxiliary_load=1.0,
        rated_power_source_kw=20000.0,
        rated_power_auxiliary_kw=auxiliary_load * 1.5,
    )

    # Test the wrong input
    mechanical_system = machinery_system.mechanical_system
    machinery_system.mechanical_system = None
    with pytest.raises(TypeError):
        machinery_system.get_machinery_result(
            mechanical_load=LoadInput(
                propulsion_load_kw=propulsion_load_kw,
                auxiliary_load_kw=auxiliary_load,
            )
        )
    machinery_system.mechanical_system = mechanical_system
    electric_system = machinery_system.electric_system
    machinery_system.electric_system = None
    with pytest.raises(TypeError):
        machinery_system.get_machinery_result(
            electric_load=LoadInput(
                propulsion_load_kw=0,
                auxiliary_load_kw=auxiliary_load,
            )
        )
    machinery_system.electric_system = electric_system
    with pytest.raises(ValueError):
        machinery_system.get_machinery_result(
            electric_load=LoadInput(
                propulsion_load_kw=propulsion_load_kw,
                auxiliary_load_kw=auxiliary_load,
            ),
            mechanical_load=LoadInput(
                propulsion_load_kw=propulsion_load_kw,
                auxiliary_load_kw=auxiliary_load,
            ),
        )
    with pytest.raises(AssertionError):
        machinery_system.get_machinery_result(
            electric_load=LoadInput(
                propulsion_load_kw=0,
                auxiliary_load_kw=auxiliary_load,
            ),
            mechanical_load=None,
        )
    with pytest.raises(AssertionError):
        machinery_system.get_machinery_result(
            electric_load=None,
            mechanical_load=LoadInput(
                propulsion_load_kw=propulsion_load_kw,
                auxiliary_load_kw=auxiliary_load,
            ),
        )

    # Test the correct input
    result = machinery_system.get_machinery_result(
        mechanical_load=LoadInput(
            propulsion_load_kw=propulsion_load_kw,
            auxiliary_load_kw=0,
        ),
        electric_load=LoadInput(
            propulsion_load_kw=0,
            auxiliary_load_kw=auxiliary_load,
        ),
    )
    assert result.mechanical_system.power_on_source_kw == pytest.approx(
        propulsion_load_kw / efficiency_propulsion_drive
    )
    assert result.electric_system.power_on_source_kw == pytest.approx(auxiliary_load)


def test_get_machinery_result_with_electric_propulsion():
    """Test get_machinery_result in MachinerySystem class with electric system"""
    # Create a machinery system
    efficiency_propulsion_drive = 0.9
    auxiliary_load = 1000.0
    propulsion_load_kw = np.linspace(0, 20000, 21)
    machinery_system = get_machinery_system_nodel(
        propulsion_type=PropulsionType.ELECTRIC,
        efficiency_propulsion_drive=efficiency_propulsion_drive,
        efficiency_power_source=0.6,
        efficiency_auxiliary_load=1.0,
        rated_power_source_kw=25000.0,
    )
    electric_power_consumption = LoadInput(
        propulsion_load_kw=np.linspace(0, 20000, 21),
        auxiliary_load_kw=auxiliary_load,
    )
    result = machinery_system.get_machinery_result(
        electric_load=electric_power_consumption
    )
    power_at_source = propulsion_load_kw / efficiency_propulsion_drive + auxiliary_load
    fuel_consumption_calculated = (
        machinery_system.electric_system.power_source.get_fuel_consumption_kg_per_h(
            power_at_source
        ).total_fuel_consumption
    )
    assert np.allclose(
        fuel_consumption_calculated,
        result.total.fuel_consumption.total_fuel_consumption,
    )


# This section has been added from nbdev tests
def create_random_fuel(number_fuel_type: int = 2) -> FuelByMassFraction:
    assert number_fuel_type > 0, "Number of fuel types must be greater than 1"
    all_fuel_types = [
        fuel_type
        for fuel_type in FuelByMassFraction.__dataclass_fields__
        if not fuel_type.startswith("_")
    ]
    fuel_types = np.random.choice(all_fuel_types, size=number_fuel_type, replace=False)
    mass_fraction_arg = {fuel_type: 0 for fuel_type in all_fuel_types}
    mass_fraction_left = 1.0
    for index, fuel_type in enumerate(fuel_types):
        if index == number_fuel_type - 1:
            mass_fraction_arg.__setitem__(fuel_type, mass_fraction_left)
        else:
            mass_fraction = mass_fraction_left * random.random()
            mass_fraction_arg.__setitem__(fuel_type, mass_fraction)
            mass_fraction_left -= mass_fraction
    return FuelByMassFraction(**mass_fraction_arg)


# Test FuelConsumption class
fuel = create_random_fuel()
fuel_consumption_each = 10
fuel_consumption_kg_per_h = FuelConsumption(
    total_fuel_consumption=fuel_consumption_each, fuel_by_mass_fraction=fuel
)

fuel_consumption_kg_per_h_new = fuel_consumption_kg_per_h + fuel_consumption_kg_per_h
assert np.isclose(
    fuel_consumption_kg_per_h_new.total_fuel_consumption, 2 * fuel_consumption_each
)
for fuel_type in fuel_consumption_kg_per_h.__dict__:
    if not fuel_type.startswith("_"):
        assert np.isclose(
            getattr(fuel_consumption_kg_per_h_new, fuel_type),
            2 * getattr(fuel_consumption_kg_per_h, fuel_type),
        )


def test_emission_factor():
    # | hide
    # Test EmissionFactor
    rated_power_kw = random.random() * 1000
    power_kw = (0.1 + 0.9 * random.random()) * rated_power_kw
    power_load_ratio = power_kw / rated_power_kw
    emission_curve = Curve(
        points=[
            Point(x=0.1, y=4),
            Point(x=0.25, y=6),
            Point(x=0.5, y=8),
            Point(x=0.75, y=7),
            Point(x=1.0, y=6),
        ]
    )
    emission_factor_interpolated = interp1d(
        x=emission_curve.to_x_array(), y=emission_curve.to_y_array(), kind="cubic"
    )(power_load_ratio)
    emissions_ref = emission_factor_interpolated * power_kw / 1000
    emission_factor = EmissionFactor(
        factor=emission_curve,
        emission_type=EmissionType.NOX,
        rated_power_kw=rated_power_kw,
    )
    emission_factor.get_emission_plot().show(renderer="svg")
    emission_calculated = emission_factor.get_emission_kg_per_h(power_kw)
    assert np.isclose(
        emission_calculated, emissions_ref
    ), f"Emission value is not correct: {emission_calculated} vs {emissions_ref}"


def test_emissions():
    # | hide
    # Test Emissions
    import random

    emissions_types1 = np.random.choice(EmissionType, size=2, replace=False)
    emissions_types2 = np.random.choice(EmissionType, size=2, replace=False)
    arg_to_emissions1 = {
        each_type.value: random.random() * 10 for each_type in emissions_types1
    }
    arg_to_emissions2 = {
        each_type.value: random.random() * 10 for each_type in emissions_types2
    }
    emissions1 = Emissions(**arg_to_emissions1)
    emissions2 = Emissions(**arg_to_emissions2)
    emissions_total = emissions1 + emissions2
    for key in EmissionType:
        emission_name = key.value
        assert np.isclose(
            getattr(emissions_total, emission_name),
            getattr(emissions1, emission_name) + getattr(emissions2, emission_name),
        ), f"{emission_name} emissions not added correctly"

    # Test MachineryResult
    power_kw1 = random.random() * 100
    power_kw2 = random.random() * 100
    fuel_consumption1 = FuelConsumption(
        total_fuel_consumption=random.random() * 100,
        fuel_by_mass_fraction=FuelByMassFraction(diesel=1.0),
    )
    fuel_consumption2 = FuelConsumption(
        total_fuel_consumption=random.random() * 100,
        fuel_by_mass_fraction=FuelByMassFraction(diesel=1.0),
    )
    machinery_result1 = MachineryResult(
        power_on_source_kw=power_kw1,
        fuel_consumption=fuel_consumption1,
        emissions=emissions1,
    )
    machinery_result2 = MachineryResult(
        power_on_source_kw=power_kw2,
        fuel_consumption=fuel_consumption2,
        emissions=emissions2,
    )
    machinery_result_total = machinery_result1 + machinery_result2
    assert np.isclose(
        machinery_result_total.power_on_source_kw, power_kw1 + power_kw2
    ), "Power not added correctly"
    assert np.isclose(
        machinery_result_total.fuel_consumption.total_fuel_consumption,
        fuel_consumption1.total_fuel_consumption
        + fuel_consumption2.total_fuel_consumption,
    ), "Fuel consumption not added correctly"
    for key in EmissionType:
        emission_name = key.value
        assert np.isclose(
            getattr(machinery_result_total.emissions, emission_name),
            getattr(emissions1, emission_name) + getattr(emissions2, emission_name),
        ), f"{emission_name} emissions not added correctly"


def test_power_sources_with_scalar_efficiency():
    fuel = create_random_fuel()
    efficiency = random.random()
    specific_fuel_consumption_ref_g_per_kwh = 1 / efficiency / fuel.lhv_mj_per_kg * 3600
    emission_factor_value = random.random()
    rated_power_kw = random.random() * 1000
    emission_factor = EmissionFactor(
        emission_type=EmissionType.CO2,
        factor=emission_factor_value,
        rated_power_kw=rated_power_kw,
    )
    power_source = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency,
        rated_power_kw=rated_power_kw,
        emission_factors=[emission_factor],
    )
    # Test scalar efficiency property
    assert (
        power_source._has_scalar_efficiency
    ), "Power source should have scalar efficiency."
    # Test specific fuel consumption
    assert np.isclose(
        power_source.get_specific_fuel_consumption_g_per_kwh(),
        specific_fuel_consumption_ref_g_per_kwh,
    ), "Specific fuel consumption should be equal to reference value."
    # Test fuel consumption with scalar input
    power_load_scalar = random.random() * rated_power_kw
    assert np.isclose(
        power_source.get_fuel_consumption_kg_per_h(
            power_load_scalar
        ).total_fuel_consumption,
        specific_fuel_consumption_ref_g_per_kwh * power_load_scalar / 1000,
    ), "Fuel consumption should be equal to reference value."
    # Test fuel consumption with array input
    power_load_array = np.random.random(100) * rated_power_kw
    assert np.allclose(
        power_source.get_fuel_consumption_kg_per_h(
            power_load_array
        ).total_fuel_consumption,
        specific_fuel_consumption_ref_g_per_kwh * power_load_array / 1000,
    ), "Fuel consumption should be equal to reference value."
    # Test emissions with scalar input
    emission_calculated = power_source.get_emissions_kg_per_h(power_load_scalar)
    emission_ref = emission_factor_value * power_load_scalar / 1000
    assert np.isclose(
        emission_calculated.co2, emission_ref
    ), f"Emissions should be equal to reference value ({emission_calculated.co2} vs {emission_ref})."
    # Test emissions with array input
    emission_calculated = power_source.get_emissions_kg_per_h(power_load_array)
    emission_ref = emission_factor_value * power_load_array / 1000
    assert np.allclose(
        emission_calculated.co2, emission_ref
    ), "Emissions should be equal to reference value."

    # Testing power source with curve efficiency
    efficiency_curve = Curve(
        points=[
            Point(x=0.0, y=0.0),
            Point(x=0.1, y=0.4),
            Point(x=0.25, y=0.5),
            Point(x=0.5, y=0.4),
            Point(x=0.75, y=0.3),
            Point(x=1.0, y=0.2),
        ]
    )
    specific_fuel_consumption_ref_g_per_kwh = (
        1
        / interp1d(
            x=efficiency_curve.to_x_array(),
            y=efficiency_curve.to_y_array(),
            kind="cubic",
        )(power_load_array / rated_power_kw)
        / fuel.lhv_mj_per_kg
        * 3600
    )
    power_source = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=efficiency_curve,
        rated_power_kw=rated_power_kw,
        emission_factors=[emission_factor],
    )
    # Test curve efficiency property
    assert (
        not power_source._has_scalar_efficiency
    ), "Power source should have curve efficiency."
    # Test specific fuel consumption
    assert np.allclose(
        power_source.get_specific_fuel_consumption_g_per_kwh(power_load_array),
        specific_fuel_consumption_ref_g_per_kwh,
    ), "Specific fuel consumption should be equal to reference value."
    # Test fuel consumption
    fuel_consumption_calculated = power_source.get_fuel_consumption_kg_per_h(
        power_load_array
    ).total_fuel_consumption
    fuel_consumption_ref = (
        specific_fuel_consumption_ref_g_per_kwh * power_load_array / 1000
    )
    assert np.allclose(fuel_consumption_calculated, fuel_consumption_ref)
    # Test emiisions
    emission_calculated = power_source.get_emissions_kg_per_h(power_load_array)
    emission_ref = emission_factor_value * power_load_array / 1000
    assert np.allclose(
        emission_calculated.co2, emission_ref
    ), "Emissions should be equal to reference value."
    # Test machinery result
    machinery_result = power_source.get_machinery_result(power_load_array)
    assert np.allclose(power_load_array, machinery_result.power_on_source_kw)
    assert np.allclose(
        fuel_consumption_ref, machinery_result.fuel_consumption.total_fuel_consumption
    )
    assert np.allclose(emission_ref, machinery_result.emissions.co2)


def test_power_sources_with_scalar_specific_fuel_consumption():
    # | hide
    specific_fuel_consumption_ref_g_per_kwh = 100 + random.random() * 100
    emission_factor_value = random.random()
    rated_power_kw = random.random() * 1000
    emission_factor = EmissionFactor(
        emission_type=EmissionType.CO2,
        factor=emission_factor_value,
        rated_power_kw=rated_power_kw,
    )
    power_source = PowerSourceWithSpecificFuelConsumption(
        fuel=fuel,
        specific_fuel_consumption=specific_fuel_consumption_ref_g_per_kwh,
        rated_power_kw=rated_power_kw,
        emission_factors=[emission_factor],
    )
    # Test scalar efficiency property
    assert (
        power_source._has_scalar_specific_fuel_consumption
    ), "Power source should have scalar specific fuel consumption."
    # Test fuel consumption with scalar input
    power_load_scalar = random.random() * rated_power_kw
    fuel_consumption_calculated = power_source.get_fuel_consumption_kg_per_h(
        power_load_scalar
    ).total_fuel_consumption
    fuel_consumption_ref = (
        specific_fuel_consumption_ref_g_per_kwh * power_load_scalar / 1000
    )
    assert np.isclose(
        fuel_consumption_calculated, fuel_consumption_ref
    ), "Fuel consumption should be equal to reference value."
    # Test fuel consumption with array input
    power_load_array = np.random.random(100) * rated_power_kw
    fuel_consumption_calculated = power_source.get_fuel_consumption_kg_per_h(
        power_load_array
    ).total_fuel_consumption
    fuel_consumption_ref = (
        specific_fuel_consumption_ref_g_per_kwh * power_load_array / 1000
    )
    assert np.allclose(
        fuel_consumption_calculated, fuel_consumption_ref
    ), "Fuel consumption should be equal to reference value."

    # Testing power source with curve specific fuel consumption
    specific_fuel_consumption_curve = Curve(
        points=[
            Point(x=0.0, y=300),
            Point(x=0.1, y=274),
            Point(x=0.25, y=250),
            Point(x=0.5, y=220),
            Point(x=0.75, y=210),
            Point(x=1.0, y=220),
        ]
    )
    specific_fuel_consumption_ref_g_per_kwh = interp1d(
        x=specific_fuel_consumption_curve.to_x_array(),
        y=specific_fuel_consumption_curve.to_y_array(),
        kind="cubic",
    )(power_load_array / rated_power_kw)
    power_source = PowerSourceWithSpecificFuelConsumption(
        fuel=fuel,
        specific_fuel_consumption=specific_fuel_consumption_curve,
        rated_power_kw=rated_power_kw,
        emission_factors=[emission_factor],
    )
    # Test curve efficiency property
    assert (
        not power_source._has_scalar_specific_fuel_consumption
    ), "Power source should have curve specific fuel consumption."
    # Test specific fuel consumption
    assert np.allclose(
        power_source.get_specific_fuel_consumption_g_per_kwh(power_load_array),
        specific_fuel_consumption_ref_g_per_kwh,
    ), "Specific fuel consumption should be equal to reference value."
    # Test fuel consumption
    fuel_consumption_calculated = power_source.get_fuel_consumption_kg_per_h(
        power_load_array
    ).total_fuel_consumption
    fuel_consumption_ref = (
        specific_fuel_consumption_ref_g_per_kwh * power_load_array / 1000
    )
    assert np.allclose(fuel_consumption_calculated, fuel_consumption_ref)
    # Test machinery result
    machinery_result = power_source.get_machinery_result(power_load_array)
    assert np.allclose(power_load_array, machinery_result.power_on_source_kw)
    assert np.allclose(
        fuel_consumption_ref, machinery_result.fuel_consumption.total_fuel_consumption
    )


def test_machinery_system():
    # | hide
    rated_power_kw = 1000 * random.random()
    efficiency = random.random()
    fuel = create_random_fuel()
    emission_factors = []
    for emission_type in np.random.choice(EmissionType, size=2, replace=False):
        emission_factors.append(
            EmissionFactor(
                emission_type=emission_type,
                factor=random.random() * 10,
                rated_power_kw=rated_power_kw,
            )
        )
    power_source = PowerSourceWithEfficiency(
        fuel=fuel,
        efficiency=random.random(),
        rated_power_kw=1000 * random.random(),
        emission_factors=emission_factors,
    )
    power_load_efficiency = 0.9854
    power_load = PowerLoad(efficiency=efficiency)
    subsystem = MachinerySubsystemSimple(
        power_source=power_source,
        propulsion_load=power_load,
    )

    power_load_scalar = rated_power_kw * random.random()
    power_out_power_source_ref = power_load_scalar / efficiency
    fuel_consumption = power_source.get_fuel_consumption_kg_per_h(
        power_out_power_source_ref
    )
    emissions = power_source.get_emissions_kg_per_h(power_out_power_source_ref)
    load_input = LoadInput(propulsion_load_kw=power_load_scalar)
    machinery_result = subsystem.get_machinery_result(load_input)
    # Test if load is included in the result
    assert np.isclose(power_out_power_source_ref, machinery_result.power_on_source_kw)

    # Test the fuel consumptions and emissions
    assert np.isclose(
        fuel_consumption.total_fuel_consumption,
        machinery_result.fuel_consumption.total_fuel_consumption,
    )
    for key in emissions.__dict__:
        assert np.isclose(
            emissions.__dict__[key], machinery_result.emissions.__dict__[key]
        )

    # | hide
    # Test Machinery system
    power_load_for_electric_system = PowerLoad(efficiency=1.0)
    electric_system = MachinerySubsystemSimple(
        power_source=power_source, auxiliary_load=power_load_for_electric_system
    )
    rated_power_for_mechanical_system = 1000 * random.random()
    specific_fuel_consumption_ref_g_per_kwh = 180
    shaft_efficiency = 0.99
    electric_power_output = (
        electric_system.power_source.rated_power_kw * random.random()
    )
    mechanical_power_output_kw = rated_power_for_mechanical_system * random.random()
    mechanical_load_input = LoadInput
    power_source_for_mechanical_system = PowerSourceWithSpecificFuelConsumption(
        rated_power_kw=rated_power_for_mechanical_system,
        fuel=fuel,
        specific_fuel_consumption=specific_fuel_consumption_ref_g_per_kwh,
        emission_factors=emission_factors,
    )
    power_load_for_mechanical_system = PowerLoad(efficiency=shaft_efficiency)
    mechanical_system = MachinerySubsystemSimple(
        power_source=power_source_for_mechanical_system,
        propulsion_load=power_load_for_mechanical_system,
    )
    load_input_for_mechanical_system = LoadInput(
        propulsion_load_kw=mechanical_power_output_kw
    )
    load_input_for_electric_system = LoadInput(auxiliary_load_kw=electric_power_output)
    result_mechanical_system = mechanical_system.get_machinery_result(
        load_input_for_mechanical_system
    )
    result_electric_system = electric_system.get_machinery_result(
        load_input_for_electric_system
    )
    machinery_system = MachinerySystem(
        propulsion_type=PropulsionType.MECHANICAL,
        mechanical_system=mechanical_system,
        electric_system=electric_system,
    )
    result = machinery_system.get_machinery_result(
        mechanical_load=load_input_for_mechanical_system,
        electric_load=load_input_for_electric_system,
    )
    print(emission_factors)
    print(result)

    # Test
    assert np.isclose(
        result.mechanical_system.fuel_consumption.total_fuel_consumption,
        result_mechanical_system.fuel_consumption.total_fuel_consumption,
    )
    assert np.allclose(
        result.mechanical_system.power_on_source_kw,
        result_mechanical_system.power_on_source_kw,
    )
    for key in EmissionType:
        assert np.isclose(
            getattr(result.mechanical_system.emissions, key.value),
            getattr(result_mechanical_system.emissions, key.value),
        )
    assert np.isclose(
        result.electric_system.fuel_consumption.total_fuel_consumption,
        result_electric_system.fuel_consumption.total_fuel_consumption,
    )
    assert np.allclose(
        result.electric_system.power_on_source_kw,
        result_electric_system.power_on_source_kw,
    )
    for key in EmissionType:
        assert np.isclose(
            getattr(result.electric_system.emissions, key.value),
            getattr(result_electric_system.emissions, key.value),
        )
    assert np.isclose(
        result.total.fuel_consumption.total_fuel_consumption,
        result_electric_system.fuel_consumption.total_fuel_consumption
        + result_mechanical_system.fuel_consumption.total_fuel_consumption,
    )
    assert np.allclose(
        result.total.power_on_source_kw,
        result_mechanical_system.power_on_source_kw
        + result_electric_system.power_on_source_kw,
    )
    for key in EmissionType:
        assert np.isclose(
            getattr(result.total.emissions, key.value),
            getattr(result_electric_system.emissions, key.value)
            + getattr(result_mechanical_system.emissions, key.value),
        )
