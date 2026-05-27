import logging
import atexit
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger("simplezarr")
logger.setLevel(logging.WARNING)


# Create executor to allow parallel reads and writes
# TODO: load it lazily, allow configuring number of workers
executor = ThreadPoolExecutor(max_workers=8)
atexit.register(lambda: executor.shutdown())
