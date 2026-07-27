"""Compatibility wrapper for older imports.

Use `notagen_runtime.notagen_replay` in new code.
"""

from notagen_runtime import notagen_replay as _impl

globals().update({name: value for name, value in vars(_impl).items() if not name.startswith("__")})
__all__ = [name for name in globals() if not name.startswith("_")]
