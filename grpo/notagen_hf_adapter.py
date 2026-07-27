"""Compatibility wrapper for older imports.

Use `notagen_runtime.notagen_hf_adapter` in new code.
"""

from notagen_runtime import notagen_hf_adapter as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
__all__ = [name for name in globals() if not name.startswith("_")]
