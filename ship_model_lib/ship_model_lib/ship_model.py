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
from dataclasses import dataclass


Numeric = TypeVar("Numeric", float, np.ndarray)


class ShipModel:
    description: "ShipDescription"
    hull_data: "HullData"
    propulsor_data: "PropulsorData"

    def __init__(
        self,
        description: "ShipDescription",
        hull_data: "HullData",
        propulsor_data: "PropulsorData",
        machinery_system: "MachinerySystem",
    ):
        self.description = description
        self.hull_data = hull_data
        self.propulsor_data = propulsor_data
        self.machinery_system = machinery_system




@dataclass
class ShipDescription:
    name: str
    type: ShipType


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


