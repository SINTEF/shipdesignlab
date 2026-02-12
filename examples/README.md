# Ship Model Library - Examples

This directory contains examples demonstrating the use of the `ship_model_lib` package in two formats:

📜 **Python Scripts** (`.py`) - Standalone runnable scripts  
📓 **Jupyter Notebooks** (`.ipynb`) - Interactive notebooks with explanations

Both fs:** 
- Python: `example_01_calm_water_resistance_speed_power.py`
- Notebook: `example_01_calm_water_resistance_speed_power.ipynb`

**Demonstrates:**Examples

### Example 1: Calm Water Resistance - Speed-Power Curve
**File:** `example_01_calm_water_resistance_speed_power.py`

Demonstrates:
# Run Python script:
python example_01_calm_water_resistance_speed_power.py

# Or open Jupyter notebook:
jupyter notebook example_01_calm_water_resistance_speed_power.ipynb
```

---

### Example 2: Calm Water Resistance - Speed-Resistance Curve
**Files:**
- Python: `example_02_calm_water_resistance_speed_resistance.py`
- Notebook: `example_02_calm_water_resistance_speed_resistance.ipynb`

**Demonstrates:**

# Run Python script:
python example_02_calm_water_resistance_speed_resistance.py

# Or open Jupyter notebook:
jupyter notebook example_02_calm_water_resistance_speed_resistance.ipynb
```

---

### Example 3: Wave Spectra and Added Resistance
**Files:**
- Python: `example_03_wave_spectra_and_added_resistance.py`
- Notebook: `example_03_wave_spectra_and_added_resistance.ipynb`

**Demonstrates:**wer from resistance and speed
- Understanding resistance vs. power

**Usage:**
```bash
python example_02_calm_water_resistance_speed_resistance.py
```
# Run Python script:
python example_03_wave_spectra_and_added_resistance.py

# Or open Jupyter notebook:
jupyter notebook example_03_wave_spectra_and_added_resistance.ipynb
```

---

### Example 4: Propeller Performance Analysis
**Files:**
- Python: `example_04_propeller_performance.py`
- Notebook: `example_04_propeller_performance.ipynb`

**Demonstrates:**ifferent wave spectrum models
- Calculating added resistance in waves using STAwave-2 method
- Understanding how wave conditions affect ship performance

# Run Python script:
python example_04_propeller_performance.py

# Or open Jupyter notebook:
jupyter notebook example_04_propeller_performance.ipynb
```

---

### Example 5: Machinery Fuel Consumption and Emissions
**Files:**
- Python: `example_05_machinery_fuel_emissions.py`
- Notebook: `example_05_machinery_fuel_emissions.ipynb`

**Demonstrates:**ample_04_propeller_performance.py`

Demonstrates:
- Creating open water propeller data
- Analyzing thrust and torque coefficients (Kt, Kq)
# Run Python script:
python example_05_machinery_fuel_emissions.py

# Or open Jupyter notebook:
jupyter notebook example_05_machinery_fuel_emissions.ipynb
```

---

### Example 6: Integrated Ship Performance Model ⭐
**Files:**
- Python: `example_06_integrated_ship_model.py`
- Notebook: `example_06_integrated_ship_model.ipynb
python example_04_propeller_performance.py
```

---

### Example 5: Machinery Fuel Consumption and Emissions
**File:** `example_05_machinery_fuel_emissions.py`

Demonstrates:
- Defining fuel properties (HFO/LSFO)
- Creating efficiency-based power sources
- Using specific fuel consumption (SFC) curves
- Multi-engine machinery systems
- Calculating emissions (CO₂, NOx, PM, SOx)
- Analyzing fuel consumption at different loads
- Daily and voyage fuel calculations

**Usage:**
```bash
# Run Python script:
python example_06_integrated_ship_model.py

# Or open Jupyter notebook (RECOMMENDED):
jupyter notebook example_06_integrated_ship_model.ipynb
```

**This is the recommended starting point** for understanding how all components work together!  
**The Jupyter notebook format is especially recommended** for this example to explore interactively.

### Example 6: Integrated Ship Performance Model ⭐
**File:** `example_06_integrated_ship_model.py`

**THE COMPREHENSIVE EXAMPLE** - Shows complete integration of ALL library components!
- jupyter (for notebooks only)

### Install for Python Scripts:
```bash
pip install ship_model_lib matplotlib numpy
```

### Install for Jupyter Notebooks:
```bash
pip install ship_model_lib matplotlib numpy jupyter
```

### Or if working from source:
```bash
cd /path/to/shipdesignlab/ship_model_lib
### Python Scripts produce:
- Console output with calculated results
- PNG image files with visualizations
- Summary statistics

### Jupyter Notebooks produce:
- Interactive cells with results
- Inline visualizations
- PNG image files saved to disk
- Step-by-step explanations with LaTeX equations

## Choosing Between Python Scripts and Notebooks

**Use Python Scripts (.py) when:**
- Running automated analyses
- Batch processing or scripting
- Quick command-line execution
- Integration with other tools

**Use Jupyter Notebooks (.ipynb) when:**
- Learning the library interactively
- Exploring data and results
- Creating presentations or reports
- Teaching or documentation
- Experimenting with parameters

Both formats contain identical calculations and produce the same results!
**Key Features:**
- Integrates: Resistance + Propulsion + Machinery + Waves
- Analyzes 5 sea states from calm to very rough
- Simulates realistic 3000nm voyage
- Compares calm water vs wave performance
- Calculates total fuel consumption and CO₂ emissions
- Generates publication-quality comparative plots

**Usage:**
```bash
python example_06_integrated_ship_model.py
```

**This is the recommended starting point** for understanding how all components work together!

---

## Prerequisites

All examples require:
- Python 3.8 or higher
- ship_model_lib package installed
- matplotlib for visualization
- numpy for numerical calculations

Install dependencies:
```bash
pip install ship_model_lib matplotlib numpy
```

Or if working from source:
```bash
cd /path/to/shipdesignlab/ship_model_lib
pip install -e .
pip install matplotlib
```

## Output

Each example generates:
- Console output with calculated results
- PNG image files with visualizations
- Detailed explanations and summaries

## Additional Resources

All examples are now included! For detailed API documentation, theoretical background, and method references:

- **API Reference:** `../DOCS/API_REFERENCE.md` - Complete technical documentation
- **Migration Guide:** `../MIGRATION.md` - Understanding the project structure
- **Main README:** `../README.md` - Project overview and quick start

For alternative resistance calculation methods (e.g., Hollenbach), see the API documentation.

## Example Structure

Each example follows a consistent structure:

1. **Imports** - Required libraries and modules
2. **Introduction** - Description of what the example demonstrates
3. **Data Setup** - Reference data and parameters
4. **Calculations** - Step-by-step calculations with explanations
5. **Visualization** - Plots and graphs
6. **Summary** - Key results and takeaways

## Contributing

To add new examples:

1. Follow the naming convention: `example_XX_descriptive_name.py`
2. Include comprehensive docstrings
3. Add console output for key results
4. Generate visualizations when appropriate
5. Update this README with the new example

## Support

For questions or issues:
- Check the API Reference: `../DOCS/API_REFERENCE.md`
- Review the main README: `../README.md`
- Open an issue on GitHub

---

*Last updated: February 2026*
