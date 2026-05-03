import sys
import logging
from contextlib import contextmanager


logger = logging.getLogger("simplezarr")
logger.setLevel(logging.WARNING)


@contextmanager
def log_exception(kind):
    """Context manager to log any exceptions, but only log a one-liner
    for subsequent occurrences of the same error to avoid spamming by
    repeating errors in e.g. a draw function or event callback.
    """
    try:
        yield
    except Exception as err:
        # Store exc info for postmortem debugging
        exc_info = list(sys.exc_info())
        exc_info[2] = exc_info[2].tb_next  # type: ignore | skip *this* function
        sys.last_type, sys.last_value, sys.last_traceback = exc_info
        # Provide the exception, so the default logger prints a stacktrace.
        # IDE's can get the exception from the root logger for PM debugging.
        logger.error(kind, exc_info=err)
