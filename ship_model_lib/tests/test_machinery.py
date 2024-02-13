import numpy as np
import pytest

from ship_model_lib.machinery import (
    MachinerySystem,
    MachinerySystemResult,
    MachinerySubsystemSimple,
    PowerSourceWithEfficiency,
    FuelByMassFraction,
    PowerLoad,
    LoadInput,
    PropulsionType,
    Curve,
    Point,
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
