"""Tests for the radar configuration dataclasses."""

from __future__ import annotations

import dataclasses

import pytest
from clutterfield.radar.config import (
    RadarAntennaConfig,
    RadarConfig,
    RadarGeometryConfig,
    RadarNoiseConfig,
    RadarWaveformConfig,
    WaveformType,
)


def test_radar_config_defaults() -> None:
    """RadarConfig should construct with sensible default sub-configs."""
    config = RadarConfig()

    assert config.name == "default_radar"
    assert isinstance(config.waveform, RadarWaveformConfig)
    assert isinstance(config.antenna, RadarAntennaConfig)
    assert isinstance(config.geometry, RadarGeometryConfig)
    assert isinstance(config.noise, RadarNoiseConfig)
    assert config.waveform.waveform_type == WaveformType.FMCW


def test_radar_config_is_frozen() -> None:
    """RadarConfig instances should be immutable."""
    config = RadarConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.name = "mutated"  # type: ignore[misc]


def test_radar_config_overrides() -> None:
    """RadarConfig sub-configs should accept explicit overrides."""
    config = RadarConfig(
        name="wideband_test",
        waveform=RadarWaveformConfig(bandwidth_hz=2.0e9, num_chirps=256),
    )

    assert config.name == "wideband_test"
    assert config.waveform.bandwidth_hz == 2.0e9
    assert config.waveform.num_chirps == 256
