# Ship Model Library - API Reference

Complete API reference for the `ship_model_lib` package with theoretical background and implementation details.

---

## Table of Contents

1. [Calm Water Resistance](#calm-water-resistance)
2. [Added Resistance](#added-resistance)
3. [Propulsor](#propulsor)
4. [Machinery](#machinery)
5. [Ship Model](#ship-model)
6. [Ship Dimensions](#ship-dimensions)
7. [Operation Profile Structure](#operation-profile-structure)
8. [Utility Functions](#utility-functions)

---

## Calm Water Resistance

Module: `ship_model_lib.calm_water_resistance`

### Overview

The calm water resistance module provides methods for calculating ship resistance in calm water conditions using prescribed speed-power or speed-resistance curves, as well as empirical methods like the Hollenbach method.

### Methods

#### Speed Power Curve

The propulsion power is defined by a prescribed curve between speed in kn (nautical mile per hour) and power in kW. This approach allows for direct representation of experimental or numerical results.

#### Hollenbach Method

Hollenbach's model is an estimate of the power requirement of a ship, which focuses on the prediction of the resistance in calm water. The model is based on regression analysis of 433 ship models, and depends on the vessel's main dimensions. The important details in the hull will therefore not have much of an effect in the resistance prediction. 

Hollenbach is the most recent empirical method for commercial vessels, and more accurate than other methods like Holtrop and Guldhammer-Harvald, as it uses a wider set of data and more complex formulas. The method separates the results in best and poor, with a difference of 5%. This gives the user the opportunity to decide whether the hull has good or poor hull lines.

**Total Resistance Formula:**

The total resistance is expressed as follows:

$$R_{T} = \frac{1}{2}\rho V^2 S C_{T}$$

where:
- $R_T$ is the total resistance (N)
- $\rho$ is density of the sea water (kg/m³)
- $V$ is the ship velocity (m/s)
- $S$ is the wetted surface area (m²)
- $C_{T}$ is a total resistance coefficient (dimensionless)

**Total Resistance Coefficient:**

$$C_{T} = C_{F} + C_{R}$$

where the viscous friction coefficient $C_F$ can be calculated as follows:

$$C_{F} = \frac{0.075}{(\log Re - 2)^2}$$

$$Re = \frac{6 F_n \sqrt{6g}}{1.1395} \cdot 10^6$$

where:
- $Re$ is the Reynolds number
- $F_n$ is the Froude number
- $g$ is gravitational acceleration (m/s²)

**Residual Friction Coefficient:**

The term for the residual friction coefficient $C_R$ is calculated:

$$C_{R, std} = b_{11} + b_{13} F_n + b_{13} F_n^2 + C_B (b_{21} + b_{22}F_n + b_{23} F_n^2) + C_B^2 (b_{31} + b_{32}F_n + b_{33} F_n^2)$$

$$C_{R} = C_{R, std}C_{R, FnKrit} k_L \left(\frac{T}{B}\right)^{a_1} \left(\frac{B}{L}\right)^{a_2} \left(\frac{L_{OS}}{L_{WL}}\right)^{a_3} \left(\frac{L_{WL}}{L_{pp}}\right)^{a_4} \left(\frac{D_p}{T_A}\right)^{a_6}$$

where the coefficients $b_{ij}$ and $a_i$ are determined from the regression analysis and depend on ship type and hull characteristics.

### Classes

#### `CalmWaterResistanceBase`

Abstract base class for calm water resistance models.

**Attributes:**
- `speed_ref_kn` (np.ndarray): Reference speeds in knots
- `draft_m` (float): Draft in meters (default: 0)
- `trim_deg` (float): Trim in degrees (default: 0)

**Methods:**

##### `_speed_to_y_interpolation_function(kind='cubic')`

Returns an interpolation function for propulsion speed in relation to a given power or resistance.

**Args:**
- `kind` (str | int): Specifies the kind of interpolation as a string or as an integer specifying the order of the spline interpolator to use. Refer to the documentation for `scipy.interpolate.interp1d`. Default is 'cubic'.

**Returns:**
- `Callable`: Interpolation function that takes a speed and returns an Interpolated1DValue

##### `_y_to_speed_interpolation_function(kind='cubic')`

Return an interpolation function for propulsion power in relation to a given speed.

**Args:**
- `kind` (str | int): Specifies the kind of interpolation. Default is 'cubic'.

**Returns:**
- `Callable`: Interpolation function that takes power/resistance and returns an Interpolated1DValue with speed

---

#### `CalmWaterResistanceBySpeedPowerCurve`

Class for calm water resistance that is specified by the speed power curve.

**Args:**
- `speed_ref_kn` (np.ndarray): Array of reference speeds in knots
- `power_ref_kw` (np.ndarray): Array of reference powers in kW (must be same length as speed_ref_kn)
- `draft_m` (float, optional): Draft in meters. Default is 0
- `trim_deg` (float, optional): Trim in degrees. Default is 0

**Raises:**
- `TypeError`: When power_ref_kw array is different in length from speed_ref_kn

**Properties:**
- `power_ref_kw` (np.ndarray): Reference power values in kW

**Methods:**

##### `get_power_from_speed(speed_kn)`

Get interpolated power for a given speed.

**Args:**
- `speed_kn` (float | np.ndarray): Speed in knots

**Returns:**
- `Interpolated1DValue`: Interpolated power in kW with metadata

##### `get_speed_from_power(power_kw)`

Get interpolated speed for a given power.

**Args:**
- `power_kw` (float | np.ndarray): Power in kW

**Returns:**
- `Interpolated1DValue`: Interpolated speed in knots with metadata

---

#### `CalmWaterResistanceBySpeedResistanceCurve`

Class for calm water resistance that is specified by the speed resistance curve.

**Args:**
- `speed_ref_kn` (np.ndarray): Array of reference speeds in knots
- `resistance_ref_k_n` (np.ndarray): Array of reference resistances in kN (must be same length as speed_ref_kn)
- `draft_m` (float, optional): Draft in meters. Default is 0
- `trim_deg` (float, optional): Trim in degrees. Default is 0

**Raises:**
- `TypeError`: When resistance_ref_k_n array is different in length from speed_ref_kn

**Properties:**
- `resistance_ref_k_n` (np.ndarray): Reference resistance values in kN

**Methods:**

##### `get_resistance_from_speed(velocity_kn)`

Get interpolated resistance for a given speed.

**Args:**
- `velocity_kn` (float | np.ndarray): Velocity in knots

**Returns:**
- `Interpolated1DValue`: Interpolated resistance in kN with metadata

##### `get_speed_from_resistance(resistance_k_n)`

Get interpolated speed for a given resistance.

**Args:**
- `resistance_k_n` (float | np.ndarray): Resistance in kN

**Returns:**
- `Interpolated1DValue`: Interpolated speed in knots with metadata

---

#### `CalmWaterResistanceHollenbachBase`

Abstract base class for Hollenbach resistance calculation methods.

Implements the Hollenbach method for estimating ship resistance in calm water based on empirical regression analysis of ship model tests.

**Attributes:**
- `ship_dimension` (ShipDimensionsHollenbachSingleScrew | ShipDimensionsHollenbachTwinScrew): Ship dimension data
- `resistance_level` (ResistanceLevel): Resistance level (BEST or POOR)

**Methods:**

Methods include various coefficient calculations (`get_fn_krit`, `get_cr_coefficients`, etc.) and the main resistance calculation methods implementing the formulas described in the theory section above.

---

#### `CalmWaterResistanceHollenbachSingleScrewDesignDraft`

Hollenbach resistance calculation for single screw vessels at design draft.

This class implements the Hollenbach method with coefficients specifically calibrated for single screw ships operating at design draft condition. It provides the most accurate predictions for conventional single-screw cargo vessels in their typical operating condition.

**Args:**
- `ship_dimensions` (ShipDimensionsHollenbachSingleScrew): Ship dimension data including length, beam, draft, block coefficient, wetted surface, and propeller parameters

**Inherits:** All methods from `CalmWaterResistanceHollenbachBase`

**Key Characteristics:**
- Calibrated for design draft operation
- Single screw propulsion configuration
- Includes propeller effects and appendage corrections

---

#### `CalmWaterResistanceHollenbachSingleScrewBallastDraft`

Hollenbach resistance calculation for single screw vessels at ballast draft.

This variant uses different empirical coefficients optimized for ships operating in ballast (light load) condition, where the hull form characteristics and resistance components differ significantly from design draft.

**Args:**
- `ship_dimensions` (ShipDimensionsHollenbachSingleScrew): Ship dimension data

**Inherits:** All methods from `CalmWaterResistanceHollenbachBase`

**Key Characteristics:**
- Calibrated for ballast draft operation
- Different residual resistance coefficients than design draft
- Accounts for reduced wetted surface and changed trim

---

#### `CalmWaterResistanceHollenbachTwinScrewDesignDraft`

Hollenbach resistance calculation for twin screw vessels at design draft.

Implements the Hollenbach method with coefficients and corrections specific to twin screw configurations, which have different wake patterns, appendage arrangements (twin rudders, brackets, bossings), and propeller-hull interactions compared to single screw vessels.

**Args:**
- `ship_dimensions` (ShipDimensionsHollenbachTwinScrew): Ship dimension data including additional parameters for twin screw configuration (number of rudders, brackets, bossings, thrusters)

**Inherits:** All methods from `CalmWaterResistanceHollenbachBase`

**Key Characteristics:**
- Calibrated for twin screw propulsion
- Additional correction factors for multiple appendages
- Different shape factor calculation with/without bulb
- Accounts for interference effects between twin propellers

**Additional Correction Parameters:**
- Number of rudders (typically 2)
- Number of brackets (propeller shaft supports)
- Number of bossings (propeller hub housings)
- Number of bow thrusters

---

## Added Resistance

Module: `ship_model_lib.added_resistance`

### Overview

The added resistance module calculates additional resistance experienced by ships in waves. Two main methods are implemented: STAwave-2 and SNNM (Liu and Papanikolaou).

### Theory

#### STAwave-2 Method

**Reference:** ITTC - Recommended Procedures and Guidelines 7.5-04-01-01.2 Rev. 1 (2014)

The empirical method has been developed by STA-JIP to approximate the transfer function of the mean resistance increase in heading regular waves by using the main parameters such as ship dimensions and speed. The method is applicable to the mean resistance increase in long crested irregular head waves $R_{AWL}$. 

The wave corrections are restricted to wave directions in the bow sector to ±45 degrees off bow. Waves within the sector are corrected as head waves. Waves outside the ±45° sector are not corrected for.

**Validity Range:**

The method is valid for the following conditions:

1. $75 \text{ m} \le L_{pp} \le 350 \text{ m}$
2. $4.0 \le \frac{L_{pp}}{B} \le 9.0$
3. $2.2 \le \frac{B}{T} \le 5.5$
4. $0.10 \le Fr \le 0.30$
5. $0.5 \le C_B \le 0.9$
6. Wave direction within 0 to ±45 degrees from bow

Where:
- $L_{pp}$ is the length between perpendiculars (m)
- $B$ is the beam (m)
- $T$ is the draft (m)
- $Fr$ is the Froude number (dimensionless)
- $C_B$ is the block coefficient (dimensionless)

**Motion Induced Resistance:**

This empirical transfer function covers both the mean resistance increase due to wave reflection $R_{AWR}$ and the motion induced resistance $R_{AWM}$.

$$R_{AWM} = 4 \rho g \zeta_A^2 \frac{B^2}{L_{pp}} \overline{raw}(\omega)$$

$$\overline{raw}(\omega) = \overline{\omega}^{b_1} \exp\left\{\frac{b_1}{d_1} (1 - \overline{\omega}^{d_1})\right\} a_1 Fr^{1.5} \exp(-3.5Fr)$$

$$\overline{\omega} = \frac{\sqrt{\frac{L_{pp}}{g}} \sqrt[3]{k_{yy}}}{1.17Fr^{-0.143}} \omega$$

$$a_1 = 60.3 C_B^{1.34}$$

$$b_1 = \begin{cases}
    11.0 & \text{for } \overline{\omega} \le 1 \\
    -8.50 & \text{elsewhere}
\end{cases}$$

$$d_1 = \begin{cases}
    14.0 & \text{for } \overline{\omega} \le 1 \\
    -566 \left(\frac{L_{pp}}{B}\right)^{-2.66} & \text{elsewhere}
\end{cases}$$

where:
- $\rho$ is seawater density (kg/m³)
- $g$ is gravitational acceleration (m/s²)
- $\zeta_A$ is wave amplitude (m)
- $\omega$ is wave frequency (rad/s)
- $k_{yy}$ is non-dimensional radius of gyration in lateral direction

**Wave Reflection Resistance:**

$$R_{AWR} = \frac{1}{2} \rho g \zeta_A^2 B \alpha_1(\omega)$$

$$\alpha_1(\omega) = \frac{\pi^2 I_1^2(1.5kT_m)}{\pi^2 I_1^2(1.5kT_m) + K_1^2(1.5kT_m)}f_1$$

$$f_1 = 0.692 \frac{V_s}{\sqrt{T_m g}}^{0.769} + 1.81C_B^{6.95}$$

where:
- $T_m$ is draught at midship (m)
- $k$ is wave number (rad/m)
- $I_1$ is modified Bessel function of the first kind of order 1
- $K_1$ is modified Bessel function of the second kind of order 1
- $V_s$ is ship speed (m/s)

**Total Added Resistance in Irregular Waves:**

$$R_{AWL} = 2 \int_0^\infty \frac{R_{AWR} + R_{AWM}}{\zeta_A^2}S_f(\omega) d\omega$$

where $S_f(\omega)$ is the wave spectrum.

---

### Wave Spectrum Theory

**References:**
- The Specialist Committee on Waves: Final Report and Recommendations to the 23rd ITTC, Proceedings of the 23rd ITTC – Volume II, 2003
- ITTC – Recommended Procedures and Guidelines: Seakeeping Experiments 7.5-02 07-02.1, 2014
- Lee et al., J. Mar. Sci. Eng. 2022

#### Pierson-Moskowitz Spectrum (ITTC 1978)

From the recommendation of ITTC for wave parameters, the spectral energy density proposed by Pierson and Moskowitz is expressed as:

$$S(\omega) = \frac{A}{\omega^5}\exp\left(-\frac{B}{\omega^4}\right)$$

In the original work by Pierson and Moskowitz both $A$ and $B$ were related to the wind speed 19.5 m above the mean sea surface, but today $A$ and $B$ are related to the main sea state parameters. Recommendation from ITTC in 1978 gives:

$$A = 173\frac{h_s^2}{T_1^4}$$

$$B = \frac{691}{T_1^4}$$

where:
- $T_1$ is mean wave period (s)
- $h_s$ is significant wave height (m)

#### JONSWAP Spectrum (ITTC 1984)

For JONSWAP spectrum, the following formulation and parameters are recommended in 1984:

$$S(\omega) = \frac{A}{\omega^5} \exp\left(-\frac{B}{\omega^4}\right) \gamma^p$$

where:

$$A = 155\frac{h_s^2}{T_1^4}$$

$$B = \frac{944}{T_1^4}$$

$$p = \exp\left[-\frac{(0.191 \omega T_1 - 1)^2}{2 \sigma_0^2}\right]$$

The default value for $\gamma$ is 3.3 for North Sea. It can be used as a control parameter for the shape of the spectrum. The parameter $\sigma$ determines the sharpness of the peak:

$$\sigma \approx \begin{cases}
    0.07, & f \leq f_p \\
    0.09, & f > f_p
\end{cases}$$

where:
- $f_p$ is the peak frequency (Hz)
- $\gamma$ is the peak enhancement factor (dimensionless)

From the work by Lee et al. (2022), the JONSWAP spectrum can also be expressed in terms of frequency $f$ (Hz):

$$S(f) = \frac{\alpha g^2}{(2\pi)^4} f^{-5} \exp\left[-1.25 (T_p f)^{-4}\right] \gamma^p$$

where the Philips parameter $\alpha = 0.0081$. This spectrum can be expressed in terms of $h_s$ and $f_p$ according to work by Goda:

$$S(f)=B h_s^2 T_p^{-4} f^{-5} \exp\left[-1.25 (T_p f)^{-4}\right] \gamma^{p*}$$

$$B = \frac{0.0624(1.094 - 0.01915 \log \gamma)}{0.230 + 0.0336 \gamma - 0.185(1.9 + \gamma)^{-1}}$$

$$T_p = \frac{T_{1/3}}{1 - 0.132(\gamma + 0.2)^{-0.559}}$$

where $\gamma = 3.3$ for North Sea and $T_{1/3}$ is the significant wave period.

---

### Classes

#### `AddedResistanceBySeaMarginCurve`

Simplified added resistance model using sea margin curves based on significant wave height.

This class provides a simple interpolation-based approach to estimate sea margin percentage from significant wave height, which can be used to approximate added resistance in waves without detailed wave spectrum calculations.

**Attributes:**
- `sea_margin_perc` (np.ndarray): Array of sea margin percentages
- `significant_wave_height_m` (np.ndarray): Array of corresponding significant wave heights in meters

**Methods:**

##### `get_sea_margin_percent(significant_wave_height_m)`

Get sea margin percentage for a given significant wave height.

**Args:**
- `significant_wave_height_m` (float | np.ndarray): Significant wave height in meters

**Returns:**
- `float | np.ndarray`: Sea margin percentage

**Example:**
```python
curve = AddedResistanceBySeaMarginCurve(
    sea_margin_perc=np.array([0, 10, 20, 30]),
    significant_wave_height_m=np.array([0, 2, 4, 6])
)

margin = curve.get_sea_margin_percent(significant_wave_height_m=3.0)  # Returns interpolated value
```

---

#### `AddedResistanceByStaWave2`

Implementation of the STAwave-2 method for calculating added resistance in waves.

**Args:**
- `ship_dimension` (ShipDimensionsAddedResistance): Ship dimension data including length, beam, draft, block coefficient, etc.
- `wave_spectrum_type` (WaveSpectrumType): Type of wave spectrum (JONSWAP_ITTC1984 or PIERSON_MOSKOWITZ_ITTC1978)
- `gamma` (float, optional): Peak enhancement factor for JONSWAP spectrum. Default is 3.3

**Methods:**

##### `get_added_resistance_newton(vessel_speed_kn, weather, heading_deg=None, verbose=False)`

Calculate added resistance for given speed and weather conditions.

**Args:**
- `vessel_speed_kn` (float | np.ndarray): Vessel speed in knots
- `weather` (Weather): Weather conditions with wave height, period, and optional direction
- `heading_deg` (float | np.ndarray, optional): Ship heading in degrees (required if weather has wave direction)
- `verbose` (bool, optional): Enable verbose output. Default is False

**Returns:**
- `float | np.ndarray`: Added resistance in Newtons

**Notes:**
- Only considers waves within ±45° of bow
- Returns 0 for waves outside this sector or when Hs = 0

##### `get_resistance_component_in_frequency(omega_rad_per_s, speed_m_per_s, weather)`

Calculate resistance component in frequency domain for integration.

**Args:**
- `omega_rad_per_s` (float): Wave frequency in rad/s
- `speed_m_per_s` (float): Ship speed in m/s
- `weather` (Weather): Weather conditions

**Returns:**
- `float`: Resistance contribution at this frequency

##### `get_resistance_due_to_reflection(omega_rad_per_s, speed_m_per_s, weather)`

Calculate resistance due to wave reflection.

**Args:**
- `omega_rad_per_s` (float): Wave frequency in rad/s
- `speed_m_per_s` (float): Ship speed in m/s
- `weather` (Weather): Weather conditions

**Returns:**
- `float`: Reflection resistance component in Newtons

##### `get_resistance_due_to_motion(omega_rad_per_s, speed_m_per_s, weather)`

Calculate resistance due to ship motions (heave, pitch).

**Args:**
- `omega_rad_per_s` (float): Wave frequency in rad/s
- `speed_m_per_s` (float): Ship speed in m/s
- `weather` (Weather): Weather conditions

**Returns:**
- `float`: Motion-induced resistance component in Newtons

---

#### `AddedResistanceBySNNM`

Implementation of the SNNM method (Liu and Papanikolaou, 2020) for calculating added resistance.

**Reference:** ITTC - Recommended Procedures and Guidelines for the Preparation, Conduct and Analysis of Speed/Power Trials, 2021, Rev 06

This method provides a more comprehensive approach to added resistance calculation considering different wave encounter angles and includes calculations for various ship types.

**Args:**
- `ship_dimension` (ShipDimensionsAddedResistance): Ship dimension data
- `weather` (Weather): Weather conditions
- `speed_ms` (float): Ship speed in m/s
- `ship_type` (ShipType): Type of ship (tanker, container, bulk carrier, etc.)

**Methods:**

Methods include calculations for length of entrance and run based on ship type, shadowed parts (forward and aft), angular distribution of waves, draft coefficients, and non-dimensional wave resistance due to motion and reflection.

---

#### Wave Spectrum Classes

##### `PiersonMoskowitzSpectrumITTC1978`

Pierson-Moskowitz wave spectrum implementation following ITTC 1978 recommendations.

**Args:**
- `hs` (float): Significant wave height in meters
- `t1` (float): Mean wave period in seconds

**Methods:**

###### `get_spectrum(omega)`

Get spectral energy density at given frequency.

**Args:**
- `omega` (float | np.ndarray): Wave frequency in rad/s

**Returns:**
- `float | np.ndarray`: Spectral energy density in m²·s

---

##### `JONSWAPSpectrumITTC1984`

JONSWAP wave spectrum implementation following ITTC 1984 recommendations.

**Args:**
- `hs` (float): Significant wave height in meters
- `t1` (float): Mean wave period in seconds
- `gamma` (float, optional): Peak enhancement factor. Default is 3.3 (North Sea)

**Methods:**

###### `get_spectrum(omega)`

Get spectral energy density at given frequency.

**Args:**
- `omega` (float | np.ndarray): Wave frequency in rad/s

**Returns:**
- `float | np.ndarray`: Spectral energy density in m²·s

---

#### `AddedResistanceWindITTC`

Calculates added resistance due to wind (air resistance) using ITTC recommended procedures.

**Reference:** ITTC (2021) Recommended Procedures and Guidelines: Preparation, Conduct and Analysis of Speed/Power Trials

**Attributes:**
- `ship_type` (ShipType): Type of ship (tanker, container, bulk carrier, etc.) - affects drag coefficients
- `transverse_area_m2` (float): Transverse projected area above waterline in m²
- `is_laden` (bool, optional): Whether ship is in laden condition. Default is True

**Methods:**

##### `get_added_resistance_newton(vessel_speed_kn, weather, heading_deg=None)`

Calculate the added resistance due to wind.

This method accounts for:
- Relative wind speed and direction based on vessel speed and heading
- Ship type-specific drag coefficients from ITTC data
- Different coefficients for laden vs. ballast condition
- Wind angle relative to vessel heading

**Args:**
- `vessel_speed_kn` (float | np.ndarray): Vessel speed in knots
- `weather` (Weather): Weather conditions including wind speed and direction
- `heading_deg` (float | np.ndarray, optional): Vessel heading in degrees (0=North, 90=East)

**Returns:**
- `float | np.ndarray`: Added resistance due to wind in Newtons

**Formula:**

The wind resistance is calculated using:

$$R_{wind} = \frac{1}{2} \rho_{air} A_T \left[C_{DA}(\psi) V_{rel}^2 + C_{DA}(0) V_s^2\right]$$

where:
- $\rho_{air}$ is air density (kg/m³)
- $A_T$ is transverse projected area (m²)
- $C_{DA}$ is drag coefficient (function of relative wind angle)
- $V_{rel}$ is relative wind speed (m/s)
- $V_s$ is ship speed (m/s)
- $\psi$ is relative wind angle (degrees)

**Example:**
```python
from ship_model_lib import AddedResistanceWindITTC, Weather, ShipType

wind_resistance = AddedResistanceWindITTC(
    ship_type=ShipType.CONTAINER,
    transverse_area_m2=850,  # Typical for large container ship
    is_laden=True
)

weather = Weather(
    wind_speed_m_per_s=15,  # 15 m/s wind
    wind_direction_deg=45   # 45° from North
)

# Calculate wind resistance at 20 knots, heading North
resistance = wind_resistance.get_added_resistance_newton(
    vessel_speed_kn=20,
    weather=weather,
    heading_deg=0  # North
)
```

---

## Propulsor

Module: `ship_model_lib.propulsor`

### Overview

The propulsor module handles propeller performance calculations using open water test data or B-series polynomial calculations.

### Theory

#### Basic Propeller Performance

The propeller performance is characterized by non-dimensional coefficients:

**Advance Ratio:**

$$J = \frac{V_a}{nD}$$

where:
- $V_a$ is the advance velocity (m/s)
- $n$ is the propeller rotational speed (rev/s)
- $D$ is the propeller diameter (m)

**Thrust Coefficient:**

$$K_T = \frac{T}{\rho n^2 D^4}$$

where $T$ is the thrust (N).

**Torque Coefficient:**

$$K_Q = \frac{Q}{\rho n^2 D^5}$$

where $Q$ is the torque (N·m).

**Open Water Efficiency:**

$$\eta_0 = \frac{K_T \cdot J}{K_Q \cdot 2\pi}$$

#### Behind-Ship Performance

**Wake Fraction Factor:**

$$w = 1 - \frac{V_a}{V_s}$$

where $V_s$ is the ship speed.

**Thrust Deduction Factor:**

The thrust deduction factor $t$ accounts for the reduction in thrust effectiveness due to hull interaction:

$$t = \frac{R_T - T}{R_T}$$

where $R_T$ is the total ship resistance.

**Hull Efficiency:**

$$\eta_H = \frac{1-t}{1-w}$$

### Classes

#### `PropulsorDataOpenWater`

Propulsor performance based on open water test data with $K_T$ and $K_Q$ as functions of advance ratio $J$.

**Args:**
- `propeller_curve_points` (list[OpenWaterPropellerCurvePoint]): List of open water curve data points
- `dp_diameter_propeller_m` (float): Propeller diameter in meters
- `wake_thrust_reduction` (list[WakeFractionThrustDeductionFactorPoint] | None, optional): Wake and thrust reduction data points
- `pitch_diameter_ratio` (float, optional): Pitch to diameter ratio, default 0.7
- `rho` (float, optional): Water density in kg/m³, default 1025

**Helper Classes:**

`OpenWaterPropellerCurvePoint`:
- `j` (float): Advance ratio
- `kt` (float): Thrust coefficient
- `kq` (float): Torque coefficient

`WakeFractionThrustDeductionFactorPoint`:
- `wake_fraction_factor` (float): Wake fraction factor
- `thrust_deduction_factor` (float): Thrust deduction factor
- `vessel_speed_kn` (float): Vessel speed in knots

**Methods:**

##### `get_kt(j)`

Get thrust coefficient for given advance ratio.

**Args:**
- `j` (float | np.ndarray): Advance ratio

**Returns:**
- `Interpolated1DValue`: Thrust coefficient

##### `get_kq(j)`

Get torque coefficient for given advance ratio.

**Args:**
- `j` (float | np.ndarray): Advance ratio

**Returns:**
- `Interpolated1DValue`: Torque coefficient

##### `get_efficiency_open_water(j)`

Get open water efficiency for given advance ratio.

**Args:**
- `j` (float | np.ndarray): Advance ratio

**Returns:**
- `float | np.ndarray`: Open water efficiency (0-1)

##### `plot_open_water_curves()`

Plot open water curves for $K_T$, $K_Q$, and $\eta_0$ against advance ratio $J$.

**Returns:**
- `None`

##### `get_thrust_newton(n_rps, vessel_speed_kn)`

Get the thrust force produced by the propeller.

**Args:**
- `n_rps` (float): Propeller rotation rate in revolutions per second
- `vessel_speed_kn` (float): Vessel speed in knots

**Returns:**
- `float`: Thrust in Newtons

**Formula:**
$$T = K_T \rho n^2 D^4$$

##### `get_thrust_reduced(n_rps, vessel_speed_kn)`

Get the reduced thrust after accounting for thrust deduction factor.

**Args:**
- `n_rps` (float): Propeller rotation rate in revolutions per second  
- `vessel_speed_kn` (float): Vessel speed in knots

**Returns:**
- `float`: Reduced thrust in Newtons

**Formula:**
$$T_{reduced} = T (1 - t)$$
where $t$ is the thrust deduction factor.

##### `get_torque_newton_meter(n_rps, vessel_speed_kn)`

Get the torque required by the propeller.

**Args:**
- `n_rps` (float): Propeller rotation rate in revolutions per second
- `vessel_speed_kn` (float): Vessel speed in knots

**Returns:**
- `float`: Torque in Newton-meters

**Formula:**
$$Q = K_Q \rho n^2 D^5$$

##### `get_propulsor_data_from_vessel_speed_rps(vessel_speed_kn, n_rps)`

Calculate complete propulsor operating point from vessel speed and propeller RPM.

**Args:**
- `vessel_speed_kn` (float | np.ndarray): Vessel speed in knots
- `n_rps` (float | np.ndarray): Propeller rotation rate in revolutions per second

**Returns:**
- `PropulsorOperatingPoint`: Complete operating point data with attributes:
  - `vessel_speed_kn`: Speed in knots
  - `n_rpm`: Propeller RPM
  - `j`: Advance ratio
  - `wake_velocity_kn`: Wake velocity
  - `propeller_thrust_newton`: Propeller thrust
  - `thrust_deduction_factor`: Thrust deduction factor
  - `resistance_newton`: Resistance (reduced thrust)
  - `torque_newton_meter`: Torque
  - `efficiency_open_water`: Open water efficiency
  - `shaft_power_kw`: Shaft power required

##### `get_propulsor_data_from_vessel_speed_thrust(vessel_speed_kn, thrust_resistance_newton)`

Calculate complete propulsor operating point from vessel speed and required thrust. Iteratively solves for the propeller RPM that produces the required thrust.

**Args:**
- `vessel_speed_kn` (float | np.ndarray): Vessel speed in knots
- `thrust_resistance_newton` (float | np.ndarray): Required thrust to overcome resistance in Newtons

**Returns:**
- `PropulsorOperatingPoint`: Complete operating point data (same structure as `get_propulsor_data_from_vessel_speed_rps`)

---

#### `PropulsorDataBseries`

B-series propeller calculations with Reynolds number corrections.

**Args:**
- `pd_pitch_diameter_ratio` (float): Pitch to diameter ratio (0.5 to 1.4)
- `ear_blade_area_ratio` (float): Expanded blade area ratio (0.3 to 1.05)
- `dp_diameter_propeller_m` (float): Propeller diameter in meters
- `z_blade_number` (int): Number of propeller blades (2 to 7)
- `wake_thrust_reduction` (WakeFractionThrustDeductionFactorPoint, optional): Wake and thrust deduction data. Default is None
- `rho` (float, optional): Water density in kg/m³. Default is 1025
- `re` (float, optional): Reynolds number. Default is 2e6
- `kp` (float, optional): Roughness of full scale propeller in meters. Default is 30e-6
- `re_correction` (ReCorrection, optional): Reynolds correction method (ReCorrection.ITTC78 or ReCorrection.Bseries). Default is ReCorrection.ITTC78

**Methods:**

Similar to `PropulsorDataOpenWater` with additional Reynolds number correction capabilities.

---

#### `PropulsorDataScalar`

Simplified propulsor model using a single scalar efficiency value for quick calculations.

**Args:**
- `efficiency` (float): Total propulsive efficiency (0-1), representing the combined efficiency of all propulsion components

**Methods:**

##### `get_propulsor_data_from_vessel_speed_thrust(vessel_speed_kn, thrust_resistance_newton)`

Calculate propulsor operating point from vessel speed and required thrust.

**Args:**
- `vessel_speed_kn` (float | np.ndarray): Vessel speed in knots
- `thrust_resistance_newton` (float | np.ndarray): Required thrust in Newtons

**Returns:**
- `PropulsorOperatingPoint`: Operating point with shaft power calculated as:
  $$P_{shaft} = \frac{T \times V_s}{\eta \times 1000}$$
  where $T$ is thrust (N), $V_s$ is speed (m/s), $\eta$ is efficiency

##### `get_propulsor_thrust_from_shaft_power(shaft_power_kw, vessel_speed_kn)`

Calculate thrust from given shaft power.

**Args:**
- `shaft_power_kw` (float | np.ndarray): Shaft power in kW
- `vessel_speed_kn` (float): Vessel speed in knots

**Returns:**
- `PropulsorOperatingPoint`: Operating point with thrust calculated as:
  $$T = \frac{\eta \times P_{shaft} \times 1000}{V_s}$$

**Note:** This class does not implement `get_propulsor_data_from_vessel_speed_rps` as it uses scalar efficiency rather than detailed propeller curves.

---

## Machinery

Module: `ship_model_lib.machinery`

### Overview

The machinery module calculates fuel consumption and emissions based on engine efficiency or specific fuel consumption curves.

### Theory

#### Fuel Consumption

Fuel consumption is calculated based on either:

1. **Specific Fuel Consumption (SFC)** in g/kWh
2. **Efficiency** ($\eta$) and fuel properties

The relationship between SFC and efficiency is:

$$\text{SFC} = \frac{3600}{\eta \times \text{LHV}}$$

where:
- $\eta$ is the thermal efficiency (0-1)
- LHV is the Lower Heating Value in MJ/kg
- 3600 converts from MJ to kWh

**Fuel Consumption Rate:**

$$\dot{m}_f = \frac{\text{SFC} \times P}{1000}$$

where:
- $\dot{m}_f$ is fuel consumption rate (kg/s)
- SFC is specific fuel consumption (g/kWh)  
- $P$ is power (kW)
- 1000 converts g/kWh to kg/kWh

#### Emissions

Emissions are calculated based on fuel consumption and emission factors:

$$E_i = \dot{m}_f \times \text{EF}_i$$

where:
- $E_i$ is the emission rate of species $i$ (kg/s)
- $\text{EF}_i$ is the emission factor for species $i$ (kg pollutant/kg fuel)

Common emission factors:
- **CO₂**: Varies by fuel type (e.g., 3.17 kg CO₂/kg fuel for marine diesel)
- **NOx**: Depends on engine technology and load (typically 0.01-0.08 kg/kg fuel)
- **SOx**: Proportional to sulfur content in fuel
- **PM** (Particulate Matter): Depends on fuel quality and combustion efficiency

### Classes

#### `PowerSourceWithEfficiency`

Power source characterized by efficiency.

**Args:**
- `efficiency` (float | Curve): Efficiency as scalar or as function of load (0-1). When using Curve, x is normalized power (power/rated_power) and y is efficiency
- `rated_power_kw` (float): Rated power output in kW
- `fuel` (FuelByMassFraction): Fuel properties
- `emission_factors` (list[EmissionFactor], optional): List of emission factors

**Methods:**

##### `get_fuel_consumption_rate_kg_per_s(power_kw)`

Calculate fuel consumption rate for given power.

**Args:**
- `power_kw` (float): Power in kW

**Returns:**
- `float`: Fuel consumption rate in kg/s

##### `get_emissions_rate(power_kw)`

Calculate emission rates for given power.

**Args:**
- `power_kw` (float): Power in kW

**Returns:**
- `dict`: Dictionary with emission rates (kg/s) for each species

---

#### `PowerSourceWithSpecificFuelConsumption`

Power source characterized by specific fuel consumption.

**Args:**
- `specific_fuel_consumption` (float | Curve): SFC as scalar or as function of load (g/kWh). When using Curve, x is normalized power (power/rated_power) and y is SFC
- `rated_power_kw` (float): Rated power output in kW
- `fuel` (FuelByMassFraction): Fuel properties
- `emission_factors` (list[EmissionFactor], optional): List of emission factors

**Methods:**

Similar to `PowerSourceWithEfficiency`.

---

#### Supporting Classes

##### `Point`

Dataclass representing a single point in a curve.

**Attributes:**
- `x` (float): X-coordinate value
- `y` (float): Y-coordinate value

---

##### `Curve`

Class representing a curve defined by a list of points. Used for efficiency curves, SFC curves, and emission factor curves.

**Attributes:**
- `points` (list[Point]): List of Point objects defining the curve

**Methods:**

###### `add_point(point)`

Add a point to the curve.

**Args:**
- `point` (Point): Point to add

###### `to_x_array()`

Convert curve points to numpy array of x-values.

**Returns:**
- `np.ndarray`: Array of x-coordinates

###### `to_y_array()`

Convert curve points to numpy array of y-values.

**Returns:**
- `np.ndarray`: Array of y-coordinates

###### `get_curve_plot(name, y_label, x_label)`

Generate a Plotly plot of the curve.

**Args:**
- `name` (str): Name/title of the plot
- `y_label` (str): Label for y-axis  
- `x_label` (str): Label for x-axis

**Returns:**
- `PlotlyFigure`: Interactive plot

---

##### `LoadInput`

Dataclass for specifying power loads on machinery subsystems.

**Attributes:**
- `propulsion_load_kw` (float | np.ndarray, optional): Propulsion power demand in kW. Default is 0
- `auxiliary_load_kw` (float | np.ndarray, optional): Auxiliary power demand in kW. Default is 0

---

##### `MachineryResult`

NamedTuple containing machinery calculation results.

**Attributes:**
- `power_on_source_kw` (float | np.ndarray): Power drawn from source in kW
- `fuel_consumption` (FuelConsumption): Fuel consumption breakdown
- `emissions` (Emissions): Emissions breakdown

**Methods:**

###### `__add__(other)`

Add two MachineryResult objects (supports `+` operator).

---

##### `MachinerySystemResult`

NamedTuple containing complete machinery system results.

**Attributes:**
- `total` (MachineryResult): Combined result from all subsystems
- `mechanical_system` (MachineryResult | None): Result from mechanical subsystem
- `electric_system` (MachineryResult | None): Result from electric subsystem

---

##### `FuelConsumption`

Class representing fuel consumption breakdown by fuel type (all in kg/h).

**Attributes:**
- `diesel`, `hfo`, `natural_gas`, `hydrogen`, `ammonia`, `lpg`, `marine_gas_oil`, `methanol` (float): Consumption rates

**Property:**
- `total_fuel_consumption` (float): Total consumption in kg/h

**Methods:**

###### `__add__(other)`

Add two FuelConsumption objects.

---

##### `Emissions`

Dataclass representing emission rates by pollutant type (all in kg/h).

**Attributes:**
- `co2` (float): CO₂ emission rate
- `sox` (float): SOₓ emission rate
- `nox` (float): NOₓ emission rate
- `pm` (float): Particulate matter emission rate

**Methods:**

###### `__add__(other)`

Add two Emissions objects.

---

##### `EmissionFactor`

Dataclass for emission factor configuration.

**Attributes:**
- `rated_power_kw` (float): Engine rated power
- `factor` (Curve | float): Emission factor in g/kWh
- `emission_type` (EmissionType): Type (CO2, NOX, SOX, PM)

**Property:**
- `has_scalar_factor` (bool): True if scalar factor

**Methods:**

###### `get_emission_kg_per_h(power_kw)`

Calculate emission rate for given power.

###### `get_emission_plot()`

Generate plot (None if scalar).

---

##### `PowerLoad`

Dataclass representing a power load with efficiency losses (e.g., shaft system, generator).

**Attributes:**
- `efficiency` (float | Curve): Efficiency (0-1) or curve
- `rated_power_kw` (float | None): Rated power (required for curve)

**Methods:**

###### `get_power_in_kw(power_out_kw)`

Calculate input power required for given output: $P_{in} = P_{out} / \eta$

---

##### `MachinerySubsystemSimple`

Simple machinery subsystem with one power source and optional loads.

**Attributes:**
- `power_source` (PowerSourceWithEfficiency | PowerSourceWithSpecificFuelConsumption): Power source
- `propulsion_load` (PowerLoad, optional): Propulsion system losses
- `auxiliary_load` (PowerLoad, optional): Auxiliary system losses

**Methods:**

###### `get_machinery_result(load)`

Calculate results for given load.

**Args:**
- `load` (LoadInput): Power loads

**Returns:**
- `MachineryResult`: Fuel and emissions results

---

#### `MachinerySystem`

Complete machinery system with mechanical and/or electric subsystems.

**Attributes:**
- `propulsion_type` (PropulsionType): MECHANICAL or ELECTRIC
- `mechanical_system` (MachinerySubsystemSimple, optional): Mechanical subsystem
- `electric_system` (MachinerySubsystemSimple, optional): Electric subsystem

**Properties:**

##### `has_mechanical_system`

Check if mechanical subsystem exists.

**Returns:**
- `bool`: True if defined

##### `has_electric_system`

Check if electric subsystem exists.

**Returns:**
- `bool`: True if defined

**Methods:**

##### `get_machinery_result(mechanical_load=None, electric_load=None)`

Calculate complete machinery system results.

**Args:**
- `mechanical_load` (LoadInput, optional): Mechanical subsystem loads
- `electric_load` (LoadInput, optional): Electric subsystem loads

**Returns:**
- `MachinerySystemResult`: Complete results

**Notes:**
- For MECHANICAL propulsion: mechanical_load contains propulsion, electric_load contains auxiliary
- For ELECTRIC propulsion: electric_load contains both propulsion and auxiliary

---

#### `FuelByMassFraction`

Fuel blend composition by mass fraction of different fuel types.

**Args:**
- `diesel` (float, optional): Diesel mass fraction (0-1), default 0.0
- `hfo` (float, optional): Heavy fuel oil mass fraction (0-1), default 0.0
- `natural_gas` (float, optional): Natural gas mass fraction (0-1), default 0.0
- `hydrogen` (float, optional): Hydrogen mass fraction (0-1), default 0.0
- `ammonia` (float, optional): Ammonia mass fraction (0-1), default 0.0
- `lpg` (float, optional): LPG mass fraction (0-1), default 0.0
- `marine_gas_oil` (float, optional): Marine gas oil mass fraction (0-1), default 0.0
- `methanol` (float, optional): Methanol mass fraction (0-1), default 0.0

**Note:** All mass fractions must sum to 1.0.

**Properties:**

##### `lhv_mj_per_kg`

Returns the lower heating value (LHV) of the fuel blend based on mass fractions.

**Returns:**
- `float`: Lower heating value in MJ/kg

**Methods:**

##### `get_co_2_curve(efficiency)`

Calculate CO₂ emission curve based on fuel blend and efficiency curve.

**Args:**
- `efficiency` (Curve): Engine efficiency curve

**Returns:**
- `Curve`: CO₂ emission curve (g CO₂/kWh shaft power)

---

## Ship Model

Module: `ship_model_lib.ship_model`

### Overview

The ship model module integrates all other modules to provide comprehensive ship performance calculations. It combines calm water resistance, added resistance from waves and wind, propulsor performance, and machinery fuel consumption/emissions.

### Theory

#### Integration of Components

The ship model integrates resistance, propulsion, and machinery components:

**Total Resistance:**

$$R_{total} = R_{calm} + R_{waves} + R_{wind} + R_{other}$$

**Required Power:**

$$P_{required} = \frac{R_{total} \times V_s}{\eta_{total}}$$

where:
- $V_s$ is ship speed (m/s)
- $\eta_{total}$ is total propulsive efficiency

**Fuel Consumption:**

$$\dot{m}_f = \frac{P_{required} \times \text{SFC}}{1000}$$

### Classes

#### `ShipDescription`

Dataclass containing ship identification information.

**Args:**
- `name` (str): Ship name
- `type` (ShipType): Ship type from ShipType enum
- `imo_number` (int, optional): IMO number. Default is 0
- `mmsi` (int, optional): MMSI number. Default is 0

**Example:**
```python
from ship_model_lib.ship_model import ShipDescription
from ship_model_lib.ship_types import ShipType

ship_desc = ShipDescription(
    name="MV Example",
    type=ShipType.container,
    imo_number=1234567,
)
```

---

#### `ShipModel`

Main class integrating all ship performance calculations.

**Args:**
- `ship_description` (ShipDescription, optional): Ship description with name and type
- `calm_water_resistance` (CalmWaterModel, optional): Calm water resistance model
- `added_resistance_wave` (AddedResistanceWaveModel, optional): Added resistance due to waves
- `added_resistance_wind` (AddedResistanceWindITTC, optional): Added resistance due to wind
- `propulsor` (PropulsorModel, optional): Propulsor performance data
- `machinery_system` (MachinerySystem, optional): Machinery system

**Methods:**

##### `get_ship_performance_data_from_speed(vessel_speed_kn, weather=None, heading_deg=None, auxiliary_power_kw=0)`

Calculate complete ship performance at given speed and conditions.

**Args:**
- `vessel_speed_kn` (float): Vessel speed in knots
- `weather` (Weather, optional): Weather conditions
- `heading_deg` (float, optional): Heading in degrees
- `auxiliary_power_kw` (float, optional): Auxiliary power in kW. Default is 0

**Returns:**
- `ShipPerformanceData`: Performance data with attributes:
  - `ship_description` (ShipDescription): Ship information
  - `hull_data` (HullOperatingPoint): Hull resistance and power data
    - `vessel_speed_kn`: Speed in knots
    - `calm_water_resistance_newton`: Calm water resistance
    - `added_resistance_wave_newton`: Wave added resistance
    - `added_resistance_wind_newton`: Wind added resistance
    - `total_resistance_newton`: Total resistance
    - `total_towing_power_kw`: Required towing power
  - `propeller_data` (PropulsorOperatingPoint): Propeller performance
    - `shaft_power_kw`: Shaft power required
    - `n_rpm`: Propeller rotation rate
    - Additional propeller metrics
  - `power_source_data` (MachinerySystemResult): Machinery results
    - `total` (MachineryResult):
      - `power_on_source_kw`: Total power from sources
      - `fuel_consumption` (FuelConsumption): Breakdown by fuel type
        - `total_fuel_consumption`: Total in kg/h
      - `emissions` (Emissions): Emissions in kg/h
        - `co2`, `sox`, `nox`, `pm`: Emission rates

##### `get_ship_performance_data_from_power(power_out_source_kw, weather=None, heading_deg=None, auxiliary_power_kw=0)`

Calculate ship performance for a given power output.

**Args:**
- `power_out_source_kw` (float): Power output from sources in kW
- `weather` (Weather, optional): Weather conditions
- `heading_deg` (float, optional): Heading in degrees
- `auxiliary_power_kw` (float, optional): Auxiliary power in kW. Default is 0

**Returns:**
- `ShipPerformanceData`: Same structure as `get_ship_performance_data_from_speed`

##### `get_ship_performance_data_from_operating_point(operation_point)`

Calculate ship performance from an operation point specification.

**Args:**
- `operation_point` (OperationPoint): Operation point with speed, weather, and power settings

**Returns:**
- `ShipPerformanceData`: Same structure as `get_ship_performance_data_from_speed`

**Example:**
```python
from ship_model_lib.ship_model import ShipDescription, ShipModel
from ship_model_lib.ship_types import ShipType

# Create ship components (calm water, propulsor, machinery, etc.)
# ... (see respective sections for component creation)

# Create ship description
ship_desc = ShipDescription(name="MV Container", type=ShipType.container)

# Create integrated ship model
ship = ShipModel(
    ship_description=ship_desc,
    calm_water_resistance=calm_water_model,
    propulsor=propeller,
    machinery_system=machinery,
)

# Get performance at specific speed
perf_data = ship.get_ship_performance_data_from_speed(
    vessel_speed_kn=18.0,
    weather=None,
    auxiliary_power_kw=500,
)

# Access results
power_kw = perf_data.power_source_data.total.power_on_source_kw
fuel_kg_h = perf_data.power_source_data.total.fuel_consumption.total_fuel_consumption
co2_kg_h = perf_data.power_source_data.total.emissions.co2
```

---

## Ship Dimensions

Module: `ship_model_lib.ship_dimensions`

### Overview

The ship dimensions module provides data classes for storing and validating ship dimensional parameters required by different resistance and performance calculation methods.

### Design Philosophy

The ship dimensions classes are designed to allow combining different pre-defined ship dimension data classes that will satisfy all resistance classes in use. Each ship dimension data class is designed to fit one resistance model class, or other classes needing ship dimensions.

By creating a new ship dimension class specific for a project and inheriting from the pre-defined ship dimension classes, the resulting class will ensure that all required fields are filled, and there is no need to fill in the same data twice.

### Classes

#### `ShipDimensionsHollenbachSingleScrew`

Ship dimensions required for Hollenbach single screw resistance calculations.

**Attributes:**
- `lpp_m` (float): Length between perpendiculars (m)
- `breadth_m` (float): Beam (m)
- `draft_design_m` (float): Design draft (m)
- `block_coefficient` (float): Block coefficient (dimensionless, 0-1)
- `wetted_surface_m2` (float): Wetted surface area (m²)
- `displacement_t` (float): Displacement (tonnes)
- Additional hull form parameters...

---

#### `ShipDimensionsHollenbachTwinScrew`

Ship dimensions required for Hollenbach twin screw resistance calculations.

Similar to single screw with additional parameters for twin screw arrangements.

---

#### `ShipDimensionsAddedResistance`

Ship dimensions required for added resistance calculations (STAwave-2, SNNM).

**Attributes:**
- `lpp_m` (float): Length between perpendiculars (m)
- `breadth_m` (float): Beam (m)
- `draft_m` (float): Draft (m)
- `block_coefficient` (float): Block coefficient (dimensionless)
- `kyy` (float): Non-dimensional radius of gyration in lateral direction
- Additional parameters for wave resistance calculations...

---

## Operation Profile Structure

Module: `ship_model_lib.operation_profile_structure`

### Overview

Data structures for representing ship operating conditions including weather, location, and operation points.

### Classes

#### `Weather`

Environmental conditions for vessel operation.

**Attributes:**
- `significant_wave_height_m` (float | np.ndarray): Significant wave height (m)
- `mean_wave_period_s` (float | np.ndarray): Mean wave period (s)
- `wave_direction_deg` (float | np.ndarray): Wave direction (degrees, 0=following waves, 180=head waves)
- `wind_speed_m_per_s` (float | np.ndarray): Wind speed (m/s)
- `wind_direction_deg` (float | np.ndarray): Wind direction (degrees, meteorological convention)
- `ocean_current_speed_m_per_s` (float | np.ndarray): Ocean current speed (m/s)
- `ocean_current_direction_deg` (float | np.ndarray): Ocean current direction (degrees)
- `air_temperature_deg_c` (float | np.ndarray): Air temperature (°C)
- `sea_water_temperature_deg_c` (float | np.ndarray): Sea water temperature (°C)

**Methods:**

##### `get_weather_at_index(index)`

Returns a new Weather object with values at the given index (for array attributes).

**Args:**
- `index` (int): Array index

**Returns:**
- `Weather`: New Weather object with scalar values at the specified index

##### `to_dict()`

Converts the Weather object to a dictionary.

**Returns:**
- `dict`: Dictionary containing all weather attributes

##### `to_dict_scalar()`

Converts the Weather object to a dictionary with scalar values. If attributes are arrays, the first value is returned.

**Returns:**
- `dict`: Dictionary with scalar values

---

#### `Location`

Geographical position.

**Attributes:**
- `longitude` (float | np.ndarray): Longitude (degrees East)
- `latitude` (float | np.ndarray): Latitude (degrees North)

**Methods:**

##### `to_dict()`

Converts the Location object to a dictionary.

**Returns:**
- `dict`: Dictionary containing longitude and latitude

##### `to_dict_scalar()`

Converts the Location object to a dictionary with scalar values. If attributes are arrays, the first value is returned.

**Returns:**
- `dict`: Dictionary with scalar values

---

#### `OperationPoint`

Complete operating state of the vessel including speed, heading, power limits, weather, and location.

**Attributes:**
- `timestamp_seconds` (float | np.ndarray): Timestamp in seconds since epoch
- `speed_kn` (float | np.ndarray): Speed (knots)
- `power_limit_kw` (float | np.ndarray): Maximum power limit (kW), default is 1e6
- `heading_deg` (float | np.ndarray): Heading (degrees, 0=North, 90=East)
- `auxiliary_power` (float | np.ndarray): Auxiliary power requirement (kW)
- `weather` (Weather, optional): Weather conditions
- `location` (Location, optional): Position

**Methods:**

##### `to_dict()`

Converts the OperationPoint to a dictionary.

**Returns:**
- `dict`: Dictionary containing all operation point attributes, including nested Weather and Location dictionaries if present

##### `to_dict_scalar()`

Converts the OperationPoint to a dictionary with scalar values. If attributes are arrays, the first value is used.

**Returns:**
- `dict[str, Any]`: Dictionary with scalar values

---

## Utility Functions

Module: `ship_model_lib.utility`

### Overview

Utility functions for unit conversions, interpolation, and common calculations.

### Functions

#### `kn_to_m_per_s(speed_kn)`

Convert speed from knots to meters per second.

**Args:**
- `speed_kn` (float | np.ndarray): Speed in knots

**Returns:**
- `float | np.ndarray`: Speed in m/s

**Formula:**
$$v_{m/s} = v_{kn} \times 0.514444$$

---

#### `m_per_s_to_kn(speed_ms)`

Convert speed from meters per second to knots.

**Args:**
- `speed_ms` (float | np.ndarray): Speed in m/s

**Returns:**
- `float | np.ndarray`: Speed in knots

**Formula:**
$$v_{kn} = v_{m/s} / 0.514444$$

---

#### `get_froude_number(velocity_ms, length_m)`

Calculate Froude number.

**Args:**
- `velocity_ms` (float): Velocity in m/s
- `length_m` (float): Characteristic length in m

**Returns:**
- `float`: Froude number (dimensionless)

**Formula:**
$$Fr = \frac{V}{\sqrt{gL}}$$

where $g = 9.81$ m/s² is gravitational acceleration.

---

#### `get_speed_kn_from_froude_number(froude_number, lpp_m)`

Calculate speed from Froude number.

Inverse of `get_froude_number` - calculates the vessel speed in knots that corresponds to a given Froude number and ship length.

**Args:**
- `froude_number` (float | np.ndarray): Froude number (dimensionless)
- `lpp_m` (float): Length between perpendiculars in meters

**Returns:**
- `float | np.ndarray`: Speed in knots

**Formula:**
$$V_{kn} = Fr \times \sqrt{gL} \times \frac{1}{0.514444}$$

where $g = 9.81$ m/s² is gravitational acceleration.

---

#### `get_interpolation_1d_function(x, y, kind='cubic', add_origo=False)`

Create a 1D interpolation function with extrapolation detection.

**Args:**
- `x` (np.ndarray): x-coordinates
- `y` (np.ndarray): y-coordinates
- `kind` (str | int, optional): Interpolation kind. Default is 'cubic'
- `add_origo` (bool, optional): If True, adds origin (0,0) point. Default is False

**Returns:**
- `Callable`: Interpolation function that returns Interpolated1DValue with extrapolation warnings

---

### Classes

#### `Interpolated1DValue`

Return value from interpolation functions with metadata.

**Attributes:**
- `value` (float | np.ndarray): Interpolated value(s)
- `is_extrapolated` (bool | np.ndarray): Whether value(s) were extrapolated
- `extrapolation_type` (str | list): Type of extrapolation ('below', 'above', or 'both')

---

## Constants

Various physical constants used throughout the library:

- `GRAVITY = 9.81` m/s²
- `KINEMATIC_VISCOSITY_WATER = 1.189e-6` m²/s
- `SEAWATER_DENSITY = 1025` kg/m³

---

## Types and Enumerations

Module: `ship_model_lib.ship_types`

### Enumerations

#### `ResistanceLevel`

Quality of hull design for resistance calculations.

**Values:**
- `BEST`: Best hull lines (5% better than average)
- `POOR`: Poor hull lines (5% worse than average)

---

#### `WaveSpectrumType`

Type of wave spectrum.

**Values:**
- `PIERSON_MOSKOWITZ_ITTC1978`
- `JONSWAP_ITTC1984`

---

#### `ShipType`

Type of ship for empirical calculations.

**Values:**
- `TANKER`
- `LIQUID_GAS`
- `BULK_CARRIER`
- `GENERAL_CARGO`
- `CONTAINER`
- `RO_PAX`

---

## Usage Examples

See the `examples/` directory for complete working examples demonstrating:

1. Basic calm water resistance calculations
2. Added resistance in waves
3. Propeller performance analysis
4. Fuel consumption and emissions calculations
5. Complete ship model integration
6. Voyage simulation

---

## References

1. ITTC - Recommended Procedures and Guidelines 7.5-04-01-01.2 Rev. 1 (2014)
2. ITTC - Recommended Procedures and Guidelines for Speed/Power Trials, 2021, Rev 06
3. The Specialist Committee on Waves: Final Report, 23rd ITTC, 2003
4. Liu, S., & Papanikolaou, A. (2020). "Fast approach to the estimation of the added resistance of ships in head waves"
5. Lee, U.-J., et al. (2022): "Estimation and Analysis of JONSWAP Spectrum Parameter", J. Mar. Sci. Eng.
6. Hollenbach, K.U.: "Estimating Resistance and Propulsion for Single-Screw and Twin-Screw Ships"

---

*Last updated: February 2026*
