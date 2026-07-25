"""Hydra entrypoint skeleton for ClutterField training experiments.

This script only wires up configuration loading and logging. No training
loop or radar algorithm is implemented yet.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from clutterfield.logging_config import get_logger, setup_logging
from clutterfield.radar.config import register_radar_configs

register_radar_configs()

logger = get_logger(__name__)


@hydra.main(config_path="../clutterfield/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Compose configuration and set up logging for an experiment run.

    Args:
        cfg: The fully composed Hydra configuration.
    """
    setup_logging(level=cfg.logging.level, log_dir=cfg.logging.log_dir)
    logger.info("Resolved configuration:\n%s", OmegaConf.to_yaml(cfg))
    logger.info("ClutterField skeleton initialized; no training loop implemented yet.")


if __name__ == "__main__":
    main()
