from typing import Any

# The shipped mitsuba 3.8 stubs have an invalid syntax (`python.tensor_io as
# python.tensor_io`); this shim takes priority via mypy_path and makes mypy
# treat everything in `mitsuba` as `Any`.
def __getattr__(name: str) -> Any: ...
