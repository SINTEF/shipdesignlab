# AUTOGENERATED! DO NOT EDIT! File to edit: ../05_ship_model.ipynb.

# %% auto 0
__all__ = [
    "Numeric",
    "CalmWaterModel",
    "PropulsorModel",
    "AddedResistanceWaveModel",
    "PowerSystem",
    "ShipDescription",
    "HullOperatingPoint",
    "HullData",
    "PropulsorData",
    "ShipPerformanceData",
    "ShipModel",
]

# %% ../05_ship_model.ipynb 2
from dataclasses import dataclass
from typing import Union, TypeVar

import numpy as np

from scipy import optimize, interpolate

from ship_model_lib.ship_dimensions import (
    ShipDimensionsHollenbachSingleScrew,
    ShipDimensionsHollenbachTwinScrew,
    ShipDimensionsAddedResistance,
)
from ship_model_lib.calm_water_resistance import (
    CalmWaterResistanceBySpeedResistanceCurve,
    CalmWaterResistanceBySpeedPowerCurve,
    CalmWaterResistanceHollenbachSingleScrewDesignDraft,
    CalmWaterResistanceHollenbachTwinScrewDesignDraft,
    CalmWaterResistanceHollenbachSingleScrewBallastDraft,
)
from ship_model_lib.propulsor import (
    PropulsorOperatingPoint,
    PropulsorDataBseries,
    PropulsorDataOpenWater,
    PropulsorDataScalar,
    WakeFractionThrustDeductionFactorPoint,
    rps_to_rad_per_s,
)
from ship_model_lib.added_resistance import (
    WaveSpectrumType,
    AddedResistanceByStaWave2,
    AddedResistanceBySeaMarginCurve,
    AddedResistanceBySNNM,
    AddedResistanceWindITTC,
)
from .operation_profile_structure import Weather, OperationPoint, Location
from .utility import kn_to_m_per_s, m_per_s_to_kn, Interpolated1DValue
from ship_model_lib.machinery import (
    Point,
    Curve,
    EmissionType,
    EmissionFactor,
    FuelByMassFraction,
    FuelConsumption,
    PowerSourceWithEfficiency,
    PowerSourceWithSpecificFuelConsumption,
    MachinerySystem,
    MachinerySubsystemSimple,
    MachineryResult,
    MachinerySystemResult,
    LoadInput,
    PropulsionType,
)
from .types import ShipType

# %% ../05_ship_model.ipynb 3
Numeric = TypeVar("Numeric", float, np.ndarray)


@dataclass
class ShipDescription:
    name: str
    type: ShipType
    imo_number: int = 0
    mmsi: int = 0


class HullOperatingPoint:
    def __init__(
        self,
        vessel_speed_kn: float,
        calm_water_resistance_newton: float,
        added_resistance_wave_newton: float,
        added_resistance_wind_newton: float,
    ):
        self.vessel_speed_kn = vessel_speed_kn
        self.calm_water_resistance_newton = calm_water_resistance_newton
        self.added_resistance_wave_newton = added_resistance_wave_newton
        self.added_resistance_wind_newton = added_resistance_wind_newton

    def __repr__(self):
        kws = [f"{key}={value!r}" for key, value in self.__dict__.items()]
        return "{}({})".format(type(self).__name__, ", ".join(kws))

    @property
    def total_resistance_newton(self):
        return (
            self.calm_water_resistance_newton
            + self.added_resistance_wave_newton
            + self.added_resistance_wind_newton
        )

    @property
    def total_towing_power_kw(self):
        vessel_speed_m_per_s = kn_to_m_per_s(self.vessel_speed_kn)
        return self.total_resistance_newton * vessel_speed_m_per_s / 1000


@dataclass
class HullData:
    b_beam_m: float
    lpp_length_between_perpendiculars_m: float
    los_length_over_surface_m: float
    lwl_length_water_line_m: float
    cb_block_coefficient: float
    ta_draft_aft_m: float
    tf_draft_forward_m: float
    wetted_surface_m2: float
    av_transverse_area_above_water_line_m2: float
    area_bilge_keel_m2: float


@dataclass
class PropulsorData:
    dp_diameter_propeller_m: float
    pd_pitch_diameter_ratio: float
    ear_blade_area_ratio: float
    z_blade_number: int


@dataclass
class ShipPerformanceData:
    ship_description: ShipDescription
    propeller_data: PropulsorOperatingPoint
    hull_data: HullOperatingPoint
    power_source_data: MachinerySystemResult


# %% ../05_ship_model.ipynb 5
CalmWaterModel = Union[
    CalmWaterResistanceHollenbachTwinScrewDesignDraft,
    CalmWaterResistanceHollenbachSingleScrewBallastDraft,
    CalmWaterResistanceBySpeedResistanceCurve,
    CalmWaterResistanceHollenbachSingleScrewDesignDraft,
    CalmWaterResistanceBySpeedPowerCurve,
]
PropulsorModel = Union[
    PropulsorDataBseries, PropulsorDataOpenWater, PropulsorDataScalar
]
AddedResistanceWaveModel = Union[
    AddedResistanceByStaWave2, AddedResistanceBySeaMarginCurve, AddedResistanceBySNNM
]
PowerSystem = Union[MachinerySystem, MachinerySubsystemSimple]


# %% ../05_ship_model.ipynb 7
class ShipModel:
    def __init__(
        self,
        ship_description: ShipDescription = None,
        calm_water_resistance: CalmWaterModel = None,
        added_resistance_wave: AddedResistanceWaveModel = None,
        added_resistance_wind: AddedResistanceWindITTC = None,
        propulsor: PropulsorModel = None,
        machinery_system: MachinerySystem = None,
    ):
        if not ship_description:
            self.ship_description = ShipDescription(
                name="ship_name_unknown", type=ShipType("ship_type_unknown")
            )
        else:
            self.ship_description = ship_description
        self.calm_water_resistance = calm_water_resistance
        self.added_resistance_wave = added_resistance_wave
        self.added_resistance_wind = added_resistance_wind
        self.propulsor = propulsor
        self.machinery_system = machinery_system

    def _get_calm_water_power_speed_curve_interpolator(self):
        power_list_kw = []
        speed_list_kn = [*range(1, 30)]
        for speed_kn in speed_list_kn:
            if isinstance(
                self.calm_water_resistance, CalmWaterResistanceBySpeedPowerCurve
            ):
                power_list_kw.append(
                    self.calm_water_resistance.get_power_from_speed(speed_kn).value
                )
            else:
                power_list_kw.append(
                    kn_to_m_per_s(speed_kn)
                    * self.calm_water_resistance.get_resistance_from_speed(speed_kn)
                )
        return interpolate.interp1d(
            np.array(power_list_kw),
            np.array(speed_list_kn),
            kind="linear",
            fill_value="extrapolate",
        )

    def _iterate_ship_power(
        self,
        vessel_velocity_kn: float,
        power_goal_kw: float,
        weather: Weather,
        heading_deg: float,
        auxiliary_power_kw: float,
    ) -> float:
        """This is the function to find the speed that gives the required power numerically."""
        ship_performance_data = self.get_ship_performance_data_from_speed(
            vessel_speed_kn=vessel_velocity_kn,
            weather=weather,
            heading_deg=heading_deg,
            auxiliary_power_kw=auxiliary_power_kw,
        )
        if self.machinery_system is not None:
            power = ship_performance_data.power_source_data.total.power_on_source_kw
        elif self.propulsor is not None:
            power = ship_performance_data.propeller_data.shaft_power_kw
        else:
            power = ship_performance_data.hull_data.total_towing_power_kw
        return power - power_goal_kw

    @staticmethod
    def _fill_weather_with_default(weather: Weather) -> Weather:
        """Fill weather with default values if not provided."""
        if weather is None:
            return None
        if weather.significant_wave_height_m is None:
            weather.significant_wave_height_m = 0.0
        if weather.wave_direction_deg is None:
            weather.wave_direction_deg = 0.0
        if weather.wind_speed_m_per_s is None:
            weather.wind_speed_m_per_s = 0.0
        if weather.wind_direction_deg is None:
            weather.wind_direction_deg = 0.0
        return weather

    def get_ship_performance_data_from_speed(
        self,
        vessel_speed_kn,
        weather: Weather = None,
        heading_deg: float = None,
        auxiliary_power_kw: float = 0,
    ) -> ShipPerformanceData:
        # Calm water implementation selection logic
        weather = self._fill_weather_with_default(weather)
        hull_operating_point = HullOperatingPoint(
            vessel_speed_kn=vessel_speed_kn,
            calm_water_resistance_newton=0.0,
            added_resistance_wave_newton=0.0,
            added_resistance_wind_newton=0.0,
        )
        propulsor_operating_point = PropulsorOperatingPoint(
            vessel_speed_kn=vessel_speed_kn,
            n_rpm=0.0,
            j=0.0,
            wake_velocity_kn=0.0,
            propeller_thrust_newton=0.0,
            thrust_deduction_factor=0.0,
            resistance_newton=0.0,
            torque_newton_meter=0.0,
            efficiency_open_water=1.0,
        )
        shaft_power_kw = (
            0.0 if np.isscalar(vessel_speed_kn) else np.zeros_like(vessel_speed_kn)
        )
        # For semi-empirical models, we need to calculate the calm water resistance,
        # added resistance and propeller performance
        if (
            not isinstance(
                self.calm_water_resistance, CalmWaterResistanceBySpeedPowerCurve
            )
            and self.calm_water_resistance is not None
        ):
            calm_water_resistance_kilo_newton = (
                self.calm_water_resistance.get_resistance_from_speed(
                    velocity_kn=vessel_speed_kn
                )
            )
            if isinstance(calm_water_resistance_kilo_newton, Interpolated1DValue):
                calm_water_resistance_kilo_newton = (
                    calm_water_resistance_kilo_newton.value
                )
            hull_operating_point.calm_water_resistance_newton = (
                calm_water_resistance_kilo_newton * 1000
            )
            added_resistance_by_sea_margin = False
            if self.added_resistance_wave is not None and weather is not None:
                added_resistance_by_sea_margin = isinstance(
                    self.added_resistance_wave, AddedResistanceBySeaMarginCurve
                )
                if not added_resistance_by_sea_margin:
                    added_resistance_wave_newton = (
                        self.added_resistance_wave.get_added_resistance_newton(
                            vessel_speed_kn=vessel_speed_kn,
                            weather=weather,
                            heading_deg=heading_deg,
                        )
                    )
                else:
                    sea_margin_percent = (
                        self.added_resistance_wave.get_sea_margin_percent(
                            significant_wave_height_m=weather.significant_wave_height_m,
                        )
                    )
                    added_resistance_wave_newton = (
                        calm_water_resistance_kilo_newton
                        * sea_margin_percent
                        / 100
                        * 1000
                    )
            else:
                added_resistance_wave_newton = 0
            if (
                self.added_resistance_wind is not None
                and weather is not None
                and not added_resistance_by_sea_margin
            ):
                added_resistance_wind_newton = (
                    self.added_resistance_wind.get_added_resistance_newton(
                        vessel_speed_kn=vessel_speed_kn,
                        weather=weather,
                        heading_deg=heading_deg,
                    )
                )
            else:
                added_resistance_wind_newton = 0
            hull_operating_point.added_resistance_wave_newton = (
                added_resistance_wave_newton
            )
            hull_operating_point.added_resistance_wind_newton = (
                added_resistance_wind_newton
            )
            if self.propulsor is not None:
                propulsor_operating_point = self.propulsor.get_propulsor_data_from_vessel_speed_thrust(
                    vessel_speed_kn=vessel_speed_kn,
                    thrust_resistance_newton=hull_operating_point.total_resistance_newton,
                )
            shaft_power_kw = propulsor_operating_point.shaft_power_kw
        # For power curve models, we need to calculate the shaft power directly
        elif isinstance(
            self.calm_water_resistance, CalmWaterResistanceBySpeedPowerCurve
        ):
            shaft_power_kw = self.calm_water_resistance.get_power_from_speed(
                speed_kn=vessel_speed_kn
            ).value
            if np.isscalar(vessel_speed_kn):
                n_rps = 1 if vessel_speed_kn > 0 else 0
                torque_newton_meter = (
                    shaft_power_kw / rps_to_rad_per_s(n_rps) * 1000
                    if shaft_power_kw > 0
                    else 0
                )
            else:
                n_rps = np.zeros_like(vessel_speed_kn)
                n_rps[vessel_speed_kn > 0] = 1
                torque_newton_meter = np.zeros_like(vessel_speed_kn)
                torque_newton_meter[shaft_power_kw > 0] = (
                    shaft_power_kw[shaft_power_kw > 0]
                    / rps_to_rad_per_s(n_rps[shaft_power_kw > 0])
                    * 1000
                )
            propulsor_operating_point.n_rpm = n_rps * 60
            propulsor_operating_point.torque_newton_meter = torque_newton_meter

        # Machinery implementation selection logic
        if self.machinery_system is None:
            fuel_consumption = FuelConsumption(
                total_fuel_consumption=0,
                fuel_by_mass_fraction=FuelByMassFraction(marine_gas_oil=1),
            )
            power_source_operating_point = MachineryResult(
                power_on_source_kw=0, fuel_consumption=fuel_consumption
            )
        else:
            if self.machinery_system.propulsion_type == PropulsionType.MECHANICAL:
                mechanical_load_input = LoadInput(propulsion_load_kw=shaft_power_kw)
                electric_load_input = LoadInput(auxiliary_load_kw=auxiliary_power_kw)
            elif self.machinery_system.propulsion_type == PropulsionType.ELECTRIC:
                mechanical_load_input = None
                electric_load_input = LoadInput(
                    propulsion_load_kw=shaft_power_kw,
                    auxiliary_load_kw=auxiliary_power_kw,
                )
            else:
                raise NotImplementedError(
                    "Getting the result from the machinery system is not implemented for "
                    "the propulsion type: {self.machinery_system.propulsion_type}"
                )
            power_source_operating_point = self.machinery_system.get_machinery_result(
                mechanical_load=mechanical_load_input, electric_load=electric_load_input
            )
        return ShipPerformanceData(
            ship_description=self.ship_description,
            propeller_data=propulsor_operating_point,
            hull_data=hull_operating_point,
            power_source_data=power_source_operating_point,
        )

    def get_ship_performance_data_from_power(
        self,
        power_out_source_kw: Numeric,
        weather: Weather = None,
        heading_deg: float = None,
        auxiliary_power_kw: Numeric = 0,
    ) -> ShipPerformanceData:
        """Calculates the ship performance data from the power requirement.

        In this calculation, the ship speed and the propeller speed are calculated from the power
        requirement. It is a useful function for when there is an upper limit on the power available
        for the ship. Note that the power_kw is the power at the power source, internal combustion
        engine, fuel cells, gensets, etc. If no machinery system is defined, the power_kw is the
        shaft power at the propeller.

        :param power_out_source_kw: The power requirement at the power source if machinery system is
            defined. It is a propulsion power otherwise.
        :param weather: The weather conditions
        :param auxiliary_power_kw: The auxiliary power consumption (other than propulsor) in kW
        :return: The ship performance data
        """
        if np.isscalar(power_out_source_kw):
            power_out_source_kw = np.array([power_out_source_kw])
        if np.isscalar(auxiliary_power_kw):
            auxiliary_power_kw = np.ones(power_out_source_kw.shape) * auxiliary_power_kw

        vessel_speed_kn = []
        speed_estimator = self._get_calm_water_power_speed_curve_interpolator()
        for index, power_kw_scalar in enumerate(power_out_source_kw):
            initial_speed_estimate = speed_estimator(power_kw_scalar)
            sol = optimize.root_scalar(
                f=self._iterate_ship_power,
                x0=initial_speed_estimate,
                x1=initial_speed_estimate * 1.05,
                args=(power_kw_scalar, weather, heading_deg, auxiliary_power_kw[index]),
            )
            if sol.converged:
                speed_found = sol.root if np.isscalar(sol.root) else sol.root[0]
                vessel_speed_kn.append(speed_found)
            else:
                raise ValueError(
                    f"Could not find a solution for the power requirement of {power_kw_scalar} "
                    f"with the initial speed estimate of {initial_speed_estimate}."
                )
        vessel_speed_kn = np.array(vessel_speed_kn)
        return self.get_ship_performance_data_from_speed(
            vessel_speed_kn=vessel_speed_kn,
            weather=weather,
            auxiliary_power_kw=auxiliary_power_kw,
        )

    def get_ship_performance_data_from_operating_point(
        self, operation_point: OperationPoint
    ) -> ShipPerformanceData:
        """Calculates the ship performance data from the operating point.

        :param operation_point: The operating point
        :return: The ship performance data
        """
        performance_data = self.get_ship_performance_data_from_speed(
            vessel_speed_kn=operation_point.speed_kn,
            weather=operation_point.weather,
            heading_deg=operation_point.heading_deg,
            auxiliary_power_kw=operation_point.auxiliary_power,
        )
        # Determine what power source data is available
        if self.machinery_system is not None:
            if self.machinery_system.propulsion_type is PropulsionType.MECHANICAL:
                power_on_source_kw = (
                    performance_data.power_source_data.mechanical_system.power_on_source_kw
                )
            elif self.machinery_system.propulsion_type is PropulsionType.ELECTRIC:
                power_on_source_kw = (
                    performance_data.power_source_data.electric_system.power_on_source_kw
                )
            else:
                raise NotImplementedError(
                    "The power source data is not implemented for the propulsion type: {self.machinery_system.propulsion_type}"
                )
        elif self.machinery_system is None and self.propulsor is not None:
            power_on_source_kw = performance_data.propeller_data.shaft_power_kw
        elif self.machinery_system is None and self.propulsor is None:
            power_on_source_kw = performance_data.hull_data.total_towing_power_kw
        else:
            power_on_source_kw = np.zeros([len(operation_point.speed_kn)])

        greater_than_power_limit = np.greater(
            power_on_source_kw, operation_point.power_limit_kw
        )
        if np.any(greater_than_power_limit):
            for index, greater_than_power_limit_each in enumerate(
                greater_than_power_limit
            ):
                if greater_than_power_limit_each:
                    if operation_point.weather.significant_wave_height_m:
                        significant_wave_height_m = (
                            operation_point.weather.significant_wave_height_m[index]
                        )
                    else:
                        significant_wave_height_m = None
                    if operation_point.weather.mean_wave_period_s:
                        mean_wave_period_s = operation_point.weather.mean_wave_period_s[
                            index
                        ]
                    else:
                        mean_wave_period_s = (None,)
                    if operation_point.weather.wave_direction_deg:
                        wave_direction_deg = operation_point.weather.wave_direction_deg[
                            index
                        ]
                    else:
                        wave_direction_deg = None
                    if operation_point.weather.wind_speed_m_per_s:
                        wind_speed_m_per_s = operation_point.weather.wind_speed_m_per_s[
                            index
                        ]
                    else:
                        wind_speed_m_per_s = None
                    if operation_point.weather.wind_direction_deg:
                        wind_direction_deg = operation_point.weather.wind_direction_deg[
                            index
                        ]
                    else:
                        wind_direction_deg = None
                    if operation_point.weather.ocean_current_speed_m_per_s:
                        ocean_current_speed_m_per_s = (
                            operation_point.weather.ocean_current_speed_m_per_s[index]
                        )
                    else:
                        ocean_current_speed_m_per_s = None
                    if operation_point.weather.ocean_current_direction_deg:
                        ocean_current_direction_deg = (
                            operation_point.weather.ocean_current_direction_deg[index]
                        )
                    else:
                        ocean_current_direction_deg = None
                    weather = Weather(
                        significant_wave_height_m=significant_wave_height_m,
                        mean_wave_period_s=mean_wave_period_s,
                        wave_direction_deg=wave_direction_deg,
                        wind_speed_m_per_s=wind_speed_m_per_s,
                        wind_direction_deg=wind_direction_deg,
                        ocean_current_speed_m_per_s=ocean_current_speed_m_per_s,
                        ocean_current_direction_deg=ocean_current_direction_deg,
                    )
                    if operation_point.power_limit_kw:
                        power_limit_kw = operation_point.power_limit_kw[index]
                    else:
                        power_limit_kw = 1e6
                    if operation_point.heading_deg:
                        heading_deg = operation_point.heading_deg[index]
                    else:
                        heading_deg = None
                    if operation_point.auxiliary_power:
                        auxiliary_power_kw = operation_point.auxiliary_power[index]
                    else:
                        auxiliary_power_kw = 0
                    performance_data_from_power = (
                        self.get_ship_performance_data_from_power(
                            power_out_source_kw=power_limit_kw,
                            weather=weather,
                            heading_deg=heading_deg,
                            auxiliary_power_kw=auxiliary_power_kw,
                        )
                    )

                    operation_point.speed_kn[index] = (
                        performance_data_from_power.hull_data.vessel_speed_kn
                    )

            performance_data = self.get_ship_performance_data_from_speed(
                vessel_speed_kn=operation_point.speed_kn,
                weather=operation_point.weather,
                heading_deg=operation_point.heading_deg,
                auxiliary_power_kw=operation_point.auxiliary_power,
            )

        return performance_data
