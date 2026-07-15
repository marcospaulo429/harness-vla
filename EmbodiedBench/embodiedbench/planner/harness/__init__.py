"""Harness VLA beta extension for EmbodiedBench (EB-Manipulation).

A memory-guided agentic planner that composes a small fixed library of analytic
primitives with a single retryable contact-rich primitive (``vla_act``). This is
a simplified, in-process, zero-shot beta of the Harness VLA framework
(arXiv:2607.08448) used as a sanity check on top of EB-Manipulation.

See ``PROVENANCE.md`` and ``docs/HARNESS_VLA_NOT_IMPLEMENTED.md`` at the repo root
for what is intentionally left out of this beta.
"""

from embodiedbench.planner.harness.primitives import (
    PrimitiveLibrary,
    PrimitiveError,
    PoseState,
)
from embodiedbench.planner.harness.global_memory import GlobalMemory

__all__ = [
    "PrimitiveLibrary",
    "PrimitiveError",
    "PoseState",
    "GlobalMemory",
]
