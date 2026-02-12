# Ship Model Library

A comprehensive Python library for ship performance modeling, resistance calculations, propulsion analysis, and fuel consumption estimation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

`ship_model_lib` is a professional-grade library for naval architects, marine engineers, and researchers working on ship design and performance analysis. It provides validated implementations of industry-standard methods for:

- **Calm Water Resistance**: Speed-power/resistance curves and Hollenbach empirical method
- **Added Resistance**: Wave and wind resistance using STAwave-2 and SNNM methods
- **Propulsion**: Open water propeller performance and B-series calculations
- **Machinery**: Fuel consumption and emissions modeling
- **Wave Spectra**: Pierson-Moskowitz and JONSWAP implementations
- **Integration**: Complete ship performance simulation and voyage analysis

## Key Features

✨ **Validated Methods**: Implementations based on ITTC recommendations and peer-reviewed research

🔬 **Theoretical Foundation**: Comprehensive documentation including mathematical formulations and background theory

📊 **Flexible Data Input**: Support for empirical data, parametric models, and hybrid approaches

⚡ **Efficient Calculations**: Optimized numerical methods with intelligent interpolation and extrapolation

🌊 **Wave Modeling**: Industry-standard wave spectra for realistic sea state simulation

🛠️ **Easy Integration**: Clean API design for seamless integration into larger systems

## Installation

> **Note**: This project now uses [uv](https://github.com/astral-sh/uv) for package management and builds. See [UV_MIGRATION.md](UV_MIGRATION.md) for the full migration guide.

### Recommended: Using uv (Fast & Modern)

```bash
# Install uv first
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/SINTEF/shipdesignlab.git
cd shipdesignlab
uv sync
```

### Alternative: Using pip

```bash
pip install ship-model-lib
```

**Note**: Package name is `ship-model-lib` (with hyphens), but import as `ship_model_lib` (with underscores):
```python
import ship_model_lib
from ship_model_lib.calm_water_resistance import CalmWaterResistanceBySpeedPowerCurve
```

### From Source with pip

```bash
git clone https://github.com/SINTEF/shipdesignlab.git
cd shipdesignlab/ship_model_lib
pip install -e .
```

### Dependencies

The library requires:
- Python ≥ 3.12
- NumPy ≥ 1.26
- SciPy ≥ 1.11
- Pandas ≥ 2.0

Optional for examples:
- Matplotlib ≥ 3.8 (for visualization)
- Plotly ≥ 5.13 (for interactive plots)

## Quick Start

### Example 1: Basic Resistance Calculation

```python
import numpy as np
from ship_model_lib.calm_water_resistance import CalmWaterResistanceBySpeedPowerCurve

# Define speed-power relationship from sea trials
speed_kn = np.array([5, 10, 15, 20])
power_kw = np.array([500, 2000, 5000, 10000])

# Create resistance model
model = CalmWaterResistanceBySpeedPowerCurve(
    speed_ref_kn=speed_kn,
    power_ref_kw=power_kw
)

# Interpolate power at any speed
power_at_12kn = model.get_power_from_speed(speed_kn=12.0)
print(f"Power at 12 knots: {power_at_12kn.value:.2f} kW")
```

### Example 2: Added Resistance in Waves

```python
from ship_model_lib.added_resistance import AddedResistanceByStaWave2
from ship_model_lib.ship_dimensions import ShipDimensionsAddedResistance
from ship_model_lib.operation_profile_structure import Weather
from ship_model_lib.ship_types import WaveSpectrumType

# Define ship dimensions
ship = ShipDimensionsAddedResistance(
    lpp_length_between_perpendiculars_m=150.0,
    b_beam_m=25.0,
    ta_draft_aft_m=8.0,
    tf_draft_forward_m=8.0,
    cb_block_coefficient=0.65,
    kyy_radius_gyration_in_lateral_direction_non_dim=0.25,
)

# Define sea state
weather = Weather(
    significant_wave_height_m=3.0,
    mean_wave_period_s=8.0,
    wave_spectrum_type=WaveSpectrumType.JONSWAP_ITTC1984,
)

# Calculate added resistance
speed_ms = 15 * 0.514444  # Convert knots to m/s
added_resistance = AddedResistanceByStaWave2(
    ship_dimension=ship,
    weather=weather,
    speed_ms=speed_ms,
)

raw_kn = added_resistance.get_added_resistance_in_waves()
print(f"Added resistance in waves: {raw_kn:.2f} kN")
```

### Example 3: Wave Spectrum Analysis

```python
import numpy as np
from ship_model_lib.added_resistance import JONSWAPSpectrumITTC1984

# Create JONSWAP spectrum for North Sea conditions
spectrum = JONSWAPSpectrumITTC1984(
    significant_wave_height_m=4.0,
    mean_wave_period_s=9.0,
    gamma=3.3  # North Sea standard
)

# Calculate spectral density
omega = np.linspace(0.1, 2.0, 100)  # Wave frequencies (rad/s)
S_omega = spectrum.get_spectral_density_omega(omega_rad_per_s=omega)

# Find peak frequency
peak_idx = np.argmax(S_omega)
print(f"Peak frequency: {omega[peak_idx]:.3f} rad/s")
```

## Documentation

### 📚 [Complete API Reference](DOCS/API_REFERENCE.md)

Comprehensive documentation including:
- Detailed descriptions of all classes and methods
- Mathematical formulations and theory
- Parameter definitions and units
- Usage guidelines and best practices
- References to published literature

### 💡 [Examples](examples/)

Standalone Python scripts demonstrating:
1. **Calm Water Resistance**: Speed-power and speed-resistance curves
2. **Wave Spectra**: Comparison of Pierson-Moskowitz and JONSWAP
3. **Added Resistance**: Wave impact on ship performance
4. **Propulsion**: Propeller performance calculations
5. **Machinery**: Fuel consumption and emissions
6. **Ship Model Integration**: Complete performance simulation

Run any example:
```bash
cd examples
python example_01_calm_water_resistance_speed_power.py
```

## Module Overview

### 🌊 Calm Water Resistance (`calm_water_resistance`)

Calculate ship resistance in calm water using:
- **Speed-Power Curves**: Interpolate from empirical or CFD data
- **Speed-Resistance Curves**: Direct resistance modeling
- **Hollenbach Method**: Empirical method based on 433 ship models

**Key Classes:**
- `CalmWaterResistanceBySpeedPowerCurve`
- `CalmWaterResistanceBySpeedResistanceCurve`
- `CalmWaterResistanceHollenbachSingleScrew`
- `CalmWaterResistanceHollenbachTwinScrew`

### 🌊 Added Resistance (`added_resistance`)

Calculate additional resistance from environmental conditions:
- **STAwave-2**: ITTC-recommended method for wave resistance (±45° bow sector)
- **SNNM**: Advanced method by Liu & Papanikolaou for all wave directions
- **Wave Spectra**: Pierson-Moskowitz (ITTC 1978) and JONSWAP (ITTC 1984)

**Key Classes:**
- `AddedResistanceByStaWave2`
- `AddedResistanceBySNNM`
- `PiersonMoskowitzSpectrumITTC1978`
- `JONSWAPSpectrumITTC1984`

### ⚓ Propulsor (`propulsor`)

Model propeller performance with:
- **Open Water Data**: Use measured Kt, Kq curves
- **B-Series**: Parametric propeller calculations with Reynolds corrections
- **Efficiency Calculations**: Open water, hull, and total propulsive efficiency

**Key Classes:**
- `PropulsorDataOpenWater`
- `PropulsorDataBseries`
- `PropulsorDataScalar`

### ⚙️ Machinery (`machinery`)

Calculate fuel consumption and emissions using:
- **Efficiency Curves**: Based on load-dependent efficiency
- **Specific Fuel Consumption**: Direct SFC curves
- **Multi-Engine Systems**: Main engines, auxiliaries, and hotel loads
- **Emissions**: CO₂, NOx, SOx, and PM calculations

**Key Classes:**
- `PowerSourceWithEfficiency`
- `PowerSourceWithSpecificFuelConsumption`
- `MachinerySystem`
- `FuelByMassFraction`

### 🚢 Ship Model (`ship_model`)

Integrate all components for complete ship performance analysis:
- Total resistance calculation (calm + waves + wind)
- Required power estimation
- Fuel consumption and emissions
- Voyage simulation

**Key Class:**
- `ShipModel`

### 📐 Ship Dimensions (`ship_dimensions`)

Data classes for ship parameters required by different methods:
- `ShipDimensionsHollenbachSingleScrew`
- `ShipDimensionsHollenbachTwinScrew`
- `ShipDimensionsAddedResistance`

### 🌍 Operation Profile (`operation_profile_structure`)

Data structures for operating conditions:
- `Weather`: Environmental conditions (waves, wind, current, temperature)
- `Location`: Geographical position
- `OperationPoint`: Complete operating state

### 🛠️ Utilities (`utility`)

Helper functions for:
- Unit conversions (knots ↔ m/s)
- Froude number calculations
- Interpolation with extrapolation detection

## Theoretical Background

This library implements methods from authoritative sources:

1. **ITTC Recommendations**
   - Resistance and propulsion test procedures
   - Wave spectrum definitions
   - Seakeeping experiments guidelines

2. **Hollenbach Method**
   - Empirical resistance prediction
   - Based on 433 ship model tests
   - Suitable for commercial vessels

3. **STAwave-2 Method**
   - STA-JIP empirical method
   - Mean resistance increase in waves
   - Valid for Froude numbers 0.10-0.30

4. **SNNM Method (Liu & Papanikolaou, 2020)**
   - Semi-empirical approach
   - All wave encounter angles
   - Ship-type specific parameters

5. **Wave Spectra**
   - Pierson-Moskowitz (1964)
   - JONSWAP (1973)
   - ITTC standardizations (1978, 1984)

All formulations are documented in [API_REFERENCE.md](DOCS/API_REFERENCE.md) with complete mathematical derivations.

## Validation

The library has been validated against:
- Published test data from ITTC
- Full-scale measurements from sea trials
- Reference implementations in literature
- Test suites in `tests/` directory

Run tests:
```bash
pytest tests/
```

## Use Cases

### ✅ Ship Design

- Estimate power requirements during early design stages
- Compare different hull forms and propeller designs
- Optimize for fuel efficiency

### ✅ Performance Analysis

- Predict speed loss in waves
- Analyze fuel consumption on specific routes
- Estimate emissions for regulatory compliance

### ✅ Voyage Planning

- Simulate complete voyages with varying weather
- Optimize routes for fuel savings
- Calculate arrival times accounting for weather

### ✅ Research

- Develop new resistance prediction methods
- Study wave-ship interaction
- Analyze environmental impact of shipping

## Project Structure

```
shipdesignlab/
├── ship_model_lib/               # Package source code
│   ├── __init__.py
│   ├── calm_water_resistance.py
│   ├── added_resistance.py
│   ├── propulsor.py
│   ├── machinery.py
│   ├── ship_model.py
│   ├── ship_dimensions.py
│   ├── operation_profile_structure.py
│   ├── utility.py
│   └── ship_types.py
├── tests/                        # Test suite
│   ├── test_calm_water_resistance.py
│   ├── test_added_resistance.py
│   ├── test_propulsor.py
│   ├── test_machinery.py
│   └── ...
├── examples/                     # Standalone examples
│   ├── example_01_calm_water_resistance_speed_power.py
│   ├── example_02_calm_water_resistance_speed_resistance.py
│   ├── example_03_wave_spectra_and_added_resistance.py
│   └── README.md
├── DOCS/                        # Documentation
│   └── API_REFERENCE.md
├── README.md                    # This file
├── LICENSE
├── requirements.txt
└── setup.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Ensure all tests pass (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

#### Using uv (Recommended)

```bash
# Clone repository
git clone https://github.com/SINTEF/shipdesignlab.git
cd shipdesignlab

# Install with all dev dependencies
uv sync --all-extras

# Or activate environment manually
source .venv/bin/activate

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=ship_model_lib tests/

# Run linter
uv run ruff check .

# Format code
uv run ruff format .

# Or using make commands
make test
make lint
make format
```

#### Using pip (Traditional)

```bash
# Clone repository
git clone https://github.com/SINTEF/shipdesignlab.git
cd shipdesignlab/ship_model_lib

# Install in development mode with test dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=ship_model_lib tests/
```

See [UV_MIGRATION.md](UV_MIGRATION.md) for complete guide on using uv, ruff, and the new build system.

## Migration from nbdev

This project was originally developed using nbdev. The migration to a standard Python package structure:

- ✅ Preserved all functional logic
- ✅ Maintained mathematical formulations
- ✅ Extracted theoretical context into documentation
- ✅ Created standalone examples
- ✅ Established comprehensive test suite
- ✅ Improved code organization and modularity

See [MIGRATION.md](MIGRATION.md) for details on the migration process.

## Version History

See [CHANGELOG.md](ship_model_lib/CHANGELOG.md) for detailed version history.

**Current Version:** 1.0.2

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in academic work, please cite:

```bibtex
@software{ship_model_lib,
  title={Ship Model Library: A Python Package for Ship Performance Modeling},
  author={Kevin Koosup Yum},
  year={2026},
  url={https://github.com/yourusername/shipdesignlab}
}
```

## References

1. ITTC (2014). "Recommended Procedures and Guidelines 7.5-04-01-01.2 Rev. 1"
2. ITTC (2021). "Recommended Procedures and Guidelines for Speed/Power Trials, Rev 06"
3. Liu, S., & Papanikolaou, A. (2020). "Fast approach to the estimation of the added resistance of ships in head waves", Ocean Engineering
4. Hollenbach, K.U. (1998). "Estimating Resistance and Propulsion for Single-Screw and Twin-Screw Ships"
5. Lee, U.-J., et al. (2022). "Estimation and Analysis of JONSWAP Spectrum Parameter", J. Mar. Sci. Eng.

## Contact

- **Issues**: [GitHub Issues](https://github.com/SINTEF/shipdesignlab/issues)
- **Discussions**: [GitHub Discussions](https://github.com/SINTEF/shipdesignlab/discussions)
- **Email**: kevin.koosup.yum@gmail.com

## Acknowledgments

- ITTC for standardized procedures and guidelines
- Maritime research community for published methods
- Contributors and users of this library

---

**Built with ❤️ for the maritime engineering community**

*Last updated: February 2026*
