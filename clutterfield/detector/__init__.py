"""Innovation-residual-based target saliency and detection.

This subpackage will hold logic that compares observed radar returns
against the predicted clutter field (the innovation residual) and
accumulates that signal over time to surface weak, low-RCS targets.
"""

from __future__ import annotations
