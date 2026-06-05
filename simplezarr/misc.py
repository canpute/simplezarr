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
