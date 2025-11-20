from random import random
import numpy as np
from typing import Union, Optional, Dict, Any
from ship_model_lib.operation_profile_structure import Weather, OperationPoint, Location

Numeric = Union[float, np.ndarray]

def test_operation_profile_structure():

    data = dict(
        significant_wave_height_m=random(),
        mean_wave_period_s=random(),
        wave_direction_deg=random(),
        wind_direction_deg=random(),
        wind_speed_m_per_s=random(),
        ocean_current_direction_deg=random(),
        ocean_current_speed_m_per_s=random(),
        sea_water_temperature_deg_c=random(),
        air_temperature_deg_c=random(),
    )

    weather = Weather(**data)
    for key, value in data.items():
        assert getattr(weather, key) == value

    operation_point = OperationPoint(
        timestamp_seconds=123,
        speed_kn=5.6,
        power_limit_kw=10000,
        heading_deg=34,
        auxiliary_power=4.6,
        weather=weather,
        location=Location(longitude=5.6, latitude=4.2),
    )

    print(operation_point.weather.to_dict())
    print(operation_point.location.to_dict())