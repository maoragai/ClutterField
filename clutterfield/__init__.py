"""ClutterField: recursive predictive neural radar fields for clutter-aware perception.

This package implements a CPU-first, edge-oriented research framework for
learning neural clutter fields that predict stationary radar clutter, with
the goal of improving detection of slow, low-RCS targets through
innovation-driven recursive updates.

Subpackages:
    encoders: Sparse multiresolution hash-grid and other spatial encodings.
    fields: Neural field architectures for predictive clutter modeling.
    radar: Radar domain configuration, geometry, and signal definitions.
    simulator: Synthetic radar and clutter simulation environments.
    detector: Innovation-residual-based target saliency and detection.
    datasets: Dataset loading, generation, and streaming utilities.
    losses: Loss functions for predictive and recursive field training.
    visualization: Plotting and inspection utilities for radar fields.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
