"""Typed dataclasses describing radar system configuration.

These dataclasses define the structured schema for radar waveform,
antenna, geometry, and noise parameters. They are registered with Hydra's
`ConfigStore` so that YAML configs under `clutterfield/configs/radar/` are
validated against this schema at composition time.

No radar algorithms are implemented here; this module only defines
configuration structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from hydra.core.config_store import ConfigStore


class WaveformType(StrEnum):
    """Supported radar waveform types."""

    FMCW = "fmcw"
    PULSED = "pulsed"
    OFDM = "ofdm"


@dataclass(frozen=True)
class RadarWaveformConfig:
    """Waveform parameters for a radar transmitter.

    Attributes:
        waveform_type: The modulation scheme used by the radar.
        carrier_frequency_hz: Carrier (center) frequency in hertz.
        bandwidth_hz: Swept bandwidth in hertz.
        chirp_duration_s: Duration of a single chirp/pulse in seconds.
        num_chirps: Number of chirps per coherent processing interval.
        sample_rate_hz: ADC sample rate in hertz.
    """

    waveform_type: WaveformType = WaveformType.FMCW
    carrier_frequency_hz: float = 77.0e9
    bandwidth_hz: float = 1.0e9
    chirp_duration_s: float = 50.0e-6
    num_chirps: int = 128
    sample_rate_hz: float = 10.0e6


@dataclass(frozen=True)
class RadarAntennaConfig:
    """Transmit/receive antenna array parameters.

    Attributes:
        num_tx: Number of transmit antenna elements.
        num_rx: Number of receive antenna elements.
        tx_spacing_m: Spacing between transmit elements, in meters.
        rx_spacing_m: Spacing between receive elements, in meters.
    """

    num_tx: int = 1
    num_rx: int = 4
    tx_spacing_m: float = 0.0
    rx_spacing_m: float = 1.9e-3


@dataclass(frozen=True)
class RadarGeometryConfig:
    """Scene geometry and discretization parameters.

    Attributes:
        max_range_m: Maximum unambiguous range in meters.
        range_bins: Number of range bins.
        max_velocity_mps: Maximum unambiguous radial velocity in m/s.
        velocity_bins: Number of Doppler/velocity bins.
        azimuth_fov_deg: Azimuth field of view in degrees.
    """

    max_range_m: float = 100.0
    range_bins: int = 256
    max_velocity_mps: float = 30.0
    velocity_bins: int = 128
    azimuth_fov_deg: float = 120.0


@dataclass(frozen=True)
class RadarNoiseConfig:
    """Noise and clutter floor parameters.

    Attributes:
        noise_floor_dbm: Thermal noise floor in dBm.
        clutter_rcs_db: Nominal clutter radar cross-section in dB.
    """

    noise_floor_dbm: float = -90.0
    clutter_rcs_db: float = 10.0


@dataclass(frozen=True)
class RadarConfig:
    """Top-level radar system configuration.

    Composes waveform, antenna, geometry, and noise configuration into a
    single structured config used throughout the framework.

    Attributes:
        name: Human-readable identifier for this radar configuration.
        waveform: Waveform configuration.
        antenna: Antenna array configuration.
        geometry: Scene geometry configuration.
        noise: Noise and clutter floor configuration.
    """

    name: str = "default_radar"
    waveform: RadarWaveformConfig = field(default_factory=RadarWaveformConfig)
    antenna: RadarAntennaConfig = field(default_factory=RadarAntennaConfig)
    geometry: RadarGeometryConfig = field(default_factory=RadarGeometryConfig)
    noise: RadarNoiseConfig = field(default_factory=RadarNoiseConfig)


def register_radar_configs(group: str = "radar", name: str = "base_radar") -> None:
    """Register radar dataclasses with Hydra's ConfigStore.

    Args:
        group: The Hydra config group to register the schema under.
        name: The name to register the schema as within that group.
    """
    config_store = ConfigStore.instance()
    config_store.store(group=group, name=name, node=RadarConfig)
