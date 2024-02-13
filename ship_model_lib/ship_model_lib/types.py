"""This module contains the types used by the ship model library."""

from enum import Enum


class EmissionType(Enum):
    CO2 = "co2"
    SOX = "sox"
    NOX = "nox"
    PM = "pm"


class ShipType(Enum):
    unknown = "ship_type_unknown"
    bulk_handysize = "bulk_handysize"
    bulk_capesize = "bulk_capesize"
    container = "container"
    tanker = "tanker"
    fishing = "fishing"
    service = "service"
    roro = "roro"
    ro_pax = "ro_pax"
    liquid_gas = "liquid_gas"
    general_cargo = "general_cargo"
    car_carrier = "car_carrier"
    multi_purpose = "multi_purpose"


class WaveSpectrumType(Enum):
    PIERSON_MOSKOWITZ_ITTC_1978 = 1
    JONSWAP_ITTC_1984 = 2


class ResistanceLevel(Enum):
    MINIMUM = "min"
    MEAN = "mean"
    MAXIMUM = "max"


class PropulsionType(Enum):
    """Enum for machinery type"""

    MECHANICAL = "mechanical"
    ELECTRIC = "electric"
    HYBRID = "hybrid"
