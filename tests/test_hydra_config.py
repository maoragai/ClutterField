"""Tests that the Hydra configuration composes correctly."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir

_CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "clutterfield" / "configs")


def test_config_composes_with_expected_groups() -> None:
    """The root config should compose radar, experiment, and logging groups."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config")

    assert cfg.seed == 42
    assert cfg.radar.name == "default_radar"
    assert cfg.experiment.name == "baseline"
    assert cfg.logging.level == "INFO"


def test_config_overrides_apply() -> None:
    """Command-line-style overrides should apply during composition."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=["experiment.max_epochs=5"])

    assert cfg.experiment.max_epochs == 5
