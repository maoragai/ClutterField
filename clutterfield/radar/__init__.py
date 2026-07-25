"""Radar domain configuration, geometry, and signal definitions.

This subpackage holds the typed configuration dataclasses that describe
radar waveforms, antenna arrays, and scene geometry (see `config.py`), as
well as (eventually) domain utilities built on top of them.
"""

from __future__ import annotations

from clutterfield.radar.config import (
    RadarAntennaConfig,
    RadarConfig,
    RadarGeometryConfig,
    RadarNoiseConfig,
    RadarWaveformConfig,
    WaveformType,
    register_radar_configs,
)

__all__ = [
    "RadarAntennaConfig",
    "RadarConfig",
    "RadarGeometryConfig",
    "RadarNoiseConfig",
    "RadarWaveformConfig",
    "WaveformType",
    "register_radar_configs",
]
