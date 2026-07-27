"""Generic NotaGen model runtime utilities.

This package contains generation, replay/logprob, and adapter code shared by
SFT sampling, GRPO, and PPO. Algorithm-specific training logic should live in
the corresponding script or package instead of importing generic NotaGen
runtime code from `grpo`.
"""
