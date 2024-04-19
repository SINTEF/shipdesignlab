# AUTOGENERATED! DO NOT EDIT! File to edit: ../07_operation_profile_structure.ipynb.

# %% auto 0
__all__ = ['Numeric', 'Location', 'Weather', 'OperationPoint']

# %% ../07_operation_profile_structure.ipynb 3
import numpy as np
from typing import Union, Optional, Dict, Any

Numeric = Union[float, np.ndarray]


class Location:
    longitude: Numeric = None
    latitude: Numeric = None
    """Wrapper class for the location"""

    def __init__(
        self,
        longitude: Optional[Numeric] = None,
        latitude: Optional[Numeric] = None,
    ):
        self.longitude = (
            np.array([longitude]) if isinstance(longitude, (float, int)) else longitude
        )
        self.latitude = (
            np.array([latitude]) if isinstance(latitude, (float, int)) else latitude
        )

    def to_dict(self):
        """Converts the object to a dictionary"""
        return {
            "longitude": self.longitude,
            "latitude": self.latitude,
        }

    def to_dict_scalar(self):
        """Converts the object to a dictionary with scalar values. If the object contains arrays,
        the first value is returned"""
        return {
            "longitude": self.longitude[0] if self.longitude is not None else None,
            "latitude": self.latitude[0] if self.latitude is not None else None,
        }


class Weather:
    """Wrapper class for the weather"""

    significant_wave_height_m: Numeric = None
    mean_wave_period_s: Numeric = None
    wave_direction_deg: Numeric = None
    wind_speed_m_per_s: Numeric = None
    wind_direction_deg: Numeric = None
    ocean_current_speed_m_per_s: Numeric = None
    ocean_current_direction_deg: Numeric = None
    air_temperature_deg_c: Numeric = None
    sea_water_temperature_deg_c: Numeric = None

    def __init__(
        self,
        significant_wave_height_m: Numeric = None,
        mean_wave_period_s: Numeric = None,
        wave_direction_deg: Numeric = None,
        wind_speed_m_per_s: Numeric = None,
        wind_direction_deg: Numeric = None,
        ocean_current_speed_m_per_s: Numeric = None,
        ocean_current_direction_deg: Numeric = None,
        air_temperature_deg_c: Numeric = None,
        sea_water_temperature_deg_c: Numeric = None,
    ):
        if sea_water_temperature_deg_c is not None:
            self.sea_water_temperature_deg_c = np.atleast_1d(
                sea_water_temperature_deg_c
            )
        if significant_wave_height_m is not None:
            self.significant_wave_height_m = np.atleast_1d(significant_wave_height_m)
        if mean_wave_period_s is not None:
            self.mean_wave_period_s = np.atleast_1d(mean_wave_period_s)
        if wave_direction_deg is not None:
            self.wave_direction_deg = np.atleast_1d(wave_direction_deg)
        if wind_speed_m_per_s is not None:
            self.wind_speed_m_per_s = np.atleast_1d(wind_speed_m_per_s)
        if wind_direction_deg is not None:
            self.wind_direction_deg = np.atleast_1d(wind_direction_deg)
        if ocean_current_speed_m_per_s is not None:
            self.ocean_current_speed_m_per_s = np.atleast_1d(
                ocean_current_speed_m_per_s
            )
        if ocean_current_direction_deg is not None:
            self.ocean_current_direction_deg = np.atleast_1d(
                ocean_current_direction_deg
            )
        if air_temperature_deg_c is not None:
            self.air_temperature_deg_c = np.atleast_1d(air_temperature_deg_c)
        if sea_water_temperature_deg_c is not None:
            self.sea_water_temperature_deg_c = np.atleast_1d(
                sea_water_temperature_deg_c
            )

    def get_weather_at_index(self, index):
        """Returns a new weather object with the values at the given index"""
        weather_index = Weather()
        if self.significant_wave_height_m is not None:
            weather_index.significant_wave_height_m = float(
                self.significant_wave_height_m[index]
            )
        if self.mean_wave_period_s is not None:
            weather_index.mean_wave_period_s = float(self.mean_wave_period_s[index])
        if self.wave_direction_deg is not None:
            weather_index.wave_direction_deg = float(self.wave_direction_deg[index])
        if self.wind_speed_m_per_s is not None:
            weather_index.wind_speed_m_per_s = float(self.wind_speed_m_per_s[index])
        if self.wind_direction_deg is not None:
            weather_index.wind_direction_deg = float(self.wind_direction_deg[index])
        if self.ocean_current_speed_m_per_s is not None:
            weather_index.ocean_current_speed_m_per_s = float(
                self.ocean_current_speed_m_per_s[index]
            )
        if self.ocean_current_direction_deg is not None:
            weather_index.ocean_current_direction_deg = float(
                self.ocean_current_direction_deg[index]
            )
        if self.air_temperature_deg_c is not None:
            weather_index.air_temperature_deg_c = float(
                self.air_temperature_deg_c[index]
            )
        if self.sea_water_temperature_deg_c is not None:
            weather_index.sea_water_temperature_deg_c = float(
                self.sea_water_temperature_deg_c[index]
            )
        return weather_index

    def to_dict(self) -> dict:
        """Converts the weather to a dictionary"""
        return {
            "significant_wave_height_m": getattr(
                self, "significant_wave_height_m", None
            ),
            "mean_wave_period_s": getattr(self, "mean_wave_period_s", None),
            "wave_direction_deg": getattr(self, "wave_direction_deg", None),
            "wind_speed_m_per_s": getattr(self, "wind_speed_m_per_s", None),
            "wind_direction_deg": getattr(self, "wind_direction_deg", None),
            "ocean_current_speed_m_per_s": getattr(
                self, "ocean_current_speed_m_per_s", None
            ),
            "ocean_current_direction_deg": getattr(
                self, "ocean_current_direction_deg", None
            ),
            "air_temperature_deg_c": getattr(self, "air_temperature_deg_c", None),
            "sea_water_temperature_deg_c": getattr(
                self, "sea_water_temperature_deg_c", None
            ),
        }

    def to_dict_scalar(self) -> Dict[str, np.ndarray]:
        """Converts the weather to a dictionary with scalar values. If the weather has timeseries,
        the first value is returned"""
        return {
            "significant_wave_height_m": (
                self.significant_wave_height_m[0]
                if self.significant_wave_height_m is not None
                else None
            ),
            "mean_wave_period_s": (
                self.mean_wave_period_s[0]
                if self.mean_wave_period_s is not None
                else None
            ),
            "wave_direction_deg": (
                self.wave_direction_deg[0]
                if self.wave_direction_deg is not None
                else None
            ),
            "wind_speed_m_per_s": (
                self.wind_speed_m_per_s[0]
                if self.wind_speed_m_per_s is not None
                else None
            ),
            "wind_direction_deg": (
                self.wind_direction_deg[0]
                if self.wind_direction_deg is not None
                else None
            ),
            "ocean_current_speed_m_per_s": (
                self.ocean_current_speed_m_per_s[0]
                if self.ocean_current_speed_m_per_s is not None
                else None
            ),
            "ocean_current_direction_deg": (
                self.ocean_current_direction_deg[0]
                if self.ocean_current_direction_deg is not None
                else None
            ),
            "air_temperature_deg_c": (
                self.air_temperature_deg_c[0]
                if self.air_temperature_deg_c is not None
                else None
            ),
            "sea_water_temperature_deg_c": (
                self.sea_water_temperature_deg_c[0]
                if self.sea_water_temperature_deg_c is not None
                else None
            ),
        }


class OperationPoint:
    """Operation point class"""

    timestamp_seconds: Numeric = None
    speed_kn: Numeric = None
    power_limit_kw: Numeric = None
    heading_deg: Numeric = None
    auxiliary_power: Numeric = None
    weather: Weather = None
    location: Location = None

    def __init__(
        self,
        speed_kn: Numeric,
        timestamp_seconds: Numeric = None,
        power_limit_kw: Numeric = 1e6,
        heading_deg: Numeric = None,
        auxiliary_power: Numeric = 0,
        weather: Weather = None,
        location: Location = None,
    ):
        self.timestamp_seconds = (
            np.array([timestamp_seconds])
            if isinstance(timestamp_seconds, (float, int))
            else timestamp_seconds
        )
        self.speed_kn = (
            np.array([speed_kn]) if isinstance(speed_kn, (float, int)) else speed_kn
        )
        self.power_limit_kw = (
            np.array([power_limit_kw])
            if isinstance(power_limit_kw, (float, int))
            else power_limit_kw
        )
        self.heading_deg = (
            np.array([heading_deg])
            if isinstance(heading_deg, (float, int))
            else heading_deg
        )
        self.auxiliary_power = (
            np.array([auxiliary_power])
            if isinstance(auxiliary_power, (float, int))
            else auxiliary_power
        )
        self.weather = weather
        self.location = location

    def to_dict(self) -> dict:
        """Converts the operation point to a dictionary"""
        return {
            "timestamp_seconds": getattr(self, "timestamp_seconds", None),
            "speed_kn": getattr(self, "speed_kn", None),
            "power_limit_kw": getattr(self, "power_limit_kw", None),
            "heading_deg": getattr(self, "heading_deg", None),
            "auxiliary_power": getattr(self, "auxiliary_power", None),
            "weather": self.weather.to_dict() if self.weather is not None else None,
            "location": self.location.to_dict() if self.location is not None else None,
        }

    def to_dict_scalar(self) -> Dict[str, Any]:
        """Converts the operation point to a dictionary with scalar values.
        If the values are arrays, the first value is used."""
        return {
            "timestamp_seconds": (
                self.timestamp_seconds[0]
                if self.timestamp_seconds is not None
                else None
            ),
            "speed_kn": self.speed_kn[0] if self.speed_kn is not None else None,
            "power_limit_kw": (
                self.power_limit_kw[0] if self.power_limit_kw is not None else None
            ),
            "heading_deg": (
                self.heading_deg[0] if self.heading_deg is not None else None
            ),
            "auxiliary_power": (
                self.auxiliary_power[0] if self.auxiliary_power is not None else None
            ),
            "weather": (
                self.weather.to_dict_scalar() if self.weather is not None else None
            ),
            "location": (
                self.location.to_dict_scalar() if self.location is not None else None
            ),
        }
