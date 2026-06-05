"""
Misc functionality used in multiple modules.
"""

import logging
import atexit
from concurrent.futures import Future, ThreadPoolExecutor


logger = logging.getLogger("simplezarr")
logger.setLevel(logging.WARNING)


# Create executor to allow parallel reads and writes
# TODO: load it lazily, allow configuring number of workers
executor = ThreadPoolExecutor(max_workers=8)
atexit.register(lambda: executor.shutdown())


class ZarrFuture(Future):
    """A subclass of concurrent.futures.Future that has a friendly repr."""

    def __init__(self, description: str):
        super().__init__()
        self._for = description

    def __repr__(self):
        r = super().__repr__()
        return r.replace(" at ", f" for {self._for} at ")


# Define all possible dtypes for a Zarr array. Note that 'rx' (with x a multiple of 8) is also allowed.
DTYPES = (
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
)


def resolve_fill_value(fill_value, dtype: str) -> tuple[object, object]:
    """Resolve fill value to a value that matches the dtype, and to its json representation."""
    real_value = json_value = fill_value
    error = None

    if dtype in DTYPES:
        if dtype == "bool":
            if fill_value is None:
                real_value = json_value = False
            elif isinstance(fill_value, str):
                false_like = ("false", "no", "0")
                true_like = "true", "yes", "1"
                low_value = fill_value.lower()
                if low_value in false_like:
                    real_value = False
                elif low_value in true_like:
                    real_value = True
                else:
                    error = "not a bool-like str"
            elif fill_value not in (False, True):
                error = "not an expected type"
        elif dtype.startswith("float"):
            if fill_value is None:
                real_value = json_value = 0.0
            elif isinstance(fill_value, str):
                try:
                    real_value = float(fill_value)
                except Exception as err:
                    error = err.args[0]
            else:
                try:
                    real_value = json_value = float(fill_value)
                except Exception as err:
                    error = err.args[0]
        elif dtype.startswith("complex"):
            if fill_value is None:
                real_value = complex()
                json_value = str(real_value)
            elif isinstance(fill_value, str):
                try:
                    real_value = complex(fill_value)
                except Exception as err:
                    error = err.args[0]
            elif isinstance(fill_value, (tuple, list)) and len(fill_value) == 2:
                try:
                    real_value = complex(fill_value[0], fill_value[1])
                    json_value = str(real_value)
                except Exception as err:
                    error = err.args[0]
            elif isinstance(fill_value, complex):
                json_value = str(real_value)
            else:
                error = "not an expected type"
        else:  # int or uint
            if fill_value is None:
                real_value = json_value = 0
            elif isinstance(fill_value, str):
                try:
                    real_value = int(fill_value)  # check, but don't replace
                except Exception as err:
                    error = err.args[0]
            else:
                try:
                    real_value = json_value = int(fill_value)  # may raise ValueError
                except Exception as err:
                    error = err.args[0]

    elif dtype.startswith("r"):
        n = int(dtype[1:])
        if fill_value is None:
            json_value = [0] * n
            real_value = bytes(json_value)
        elif isinstance(fill_value, str):
            error = "str not expected for raw dtype"
        elif isinstance(fill_value, bytes):
            if len(fill_value) != n:
                error = f"number of bytes does not match dtype {dtype}"
            real_value = fill_value
            json_value = list(real_value)
        else:
            error = "not an expected type"

    else:  # no-cover
        pass  # extension type ... we currently don't check

    if error:
        raise ValueError(
            f"ZarrArray fill_value must match dtype {dtype}, but got {fill_value!r}: {error}"
        )

    return real_value, json_value
