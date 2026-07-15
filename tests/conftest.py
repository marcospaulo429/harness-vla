"""Pytest configuration: make the vendored EmbodiedBench importable."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EB_ROOT = os.path.join(os.path.dirname(_HERE), "EmbodiedBench")
if _EB_ROOT not in sys.path:
    sys.path.insert(0, _EB_ROOT)
