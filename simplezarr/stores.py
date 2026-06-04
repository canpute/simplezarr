# The store interface and implementations. This code follows the abstract store
# interface as defines in the Zarr spec:
#
#     https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
#
# Some quotes from the spec, for easy reference:
#
# * The store interface is intended to be simple to implement using a variety of
#   different underlying storage technologies.
# * It is assumed that the store holds (key, value) pairs, with only one such pair
#   for any given key. I.e., a store is a mapping from keys to values.
# * It is also assumed that keys are case sensitive, i.e., the keys “foo” and
#   “FOO” are different.
# * In the context of this interface, a key is a Unicode string, where the final
#   character is not a '/' character.
# * In the context of this interface, a prefix is a string containing only
#   characters that are valid for use in keys and ending with a trailing '/'
#   character.
# * The store operations are grouped into three sets of capabilities: readable,
#   writeable and listable. It is not necessary for a store implementation to
#   support all of these capabilities.
#
# Further, the spec seems to assume that there is no dash prefix; so "a/b", rather
# than "/a/b". We follow this in the stores.

from __future__ import annotations
from pathlib import Path
import time

__all__ = [
    "BaseStore",
    "ReadableStore",
    "WritableStore",
    "ListableStore",
    "MemoryStore",
    "LocalStore",
    "WrapperStore",
    "SlowStore",
]

List = list  # for typing


def check_key(ob: object, method: str, key: str) -> str:
    if not isinstance(key, str):
        raise TypeError(
            f"{ob.__class__.__name__}.{method}(): key must be a str, got {key!r}"
        )
    if key.startswith("/"):
        raise ValueError(
            f"{ob.__class__.__name__}.{method}(): key must not start with '/', got {key!r}"
        )
    if not key:
        raise ValueError(
            f"{ob.__class__.__name__}.{method}(): key must not be empty, got {key!r}"
        )
    if key.endswith("/"):
        raise ValueError(
            f"{ob.__class__.__name__}.{method}(): key must not end with '/', got {key!r}"
        )
    if any(name.count(".") == len(name) for name in key.split("/")):
        raise ValueError(
            f"{ob.__class__.__name__}.{method}(): prefix parts cannot only consists of period chars, got {key!r}"
        )
    return key


def check_prefix(ob: object, method: str, prefix: str) -> str:
    if not isinstance(prefix, str):
        raise TypeError(
            f"{ob.__class__.__name__}.{method}(): prefix must be a str, got {prefix!r}"
        )
    if prefix != "/" and prefix.startswith("/"):
        raise ValueError(
            f"{ob.__class__.__name__}.{method}(): prefix must not start with '/', got {prefix!r}"
        )
    if not prefix.endswith("/"):
        raise ValueError(
            f"{ob.__class__.__name__}.{method}(): prefix must end with '/', got {prefix!r}"
        )
    if len(prefix) > 1 and any(
        name.count(".") == len(name) for name in prefix[:-1].split("/")
    ):
        raise ValueError(
            f"{ob.__class__.__name__}.{method}(): prefix parts cannot only consists of period chars, got {prefix!r}"
        )
    return prefix


def check_key_range(ob: object, key_range) -> tuple[str, int, int | None]:
    if not (isinstance(key_range, tuple) and len(key_range) == 3):
        raise TypeError(
            "f{ob.__class__.__name__}.get_partial_values(): key_ranges entries must be 3-element tuples."
        )
    return key_range


def check_key_start_value(ob: object, key_start_value) -> tuple[str, int, bytes]:
    if not (isinstance(key_start_value, tuple) and len(key_start_value) == 3):
        raise TypeError(
            "f{ob.__class__.__name__}.set_partial_values(): key_start_values entries must be 3-element tuples."
        )
    return key_start_value


class BaseStore:
    """The base store class.

    Zarr data is stored as a set of key-value pairs. This can be a directory with
    files on your hard-drive. Or a Python dict in memory, or a remote resource
    accessible over the internet.

    Stores give access to that data in a consistent way, so that the code that
    reads/writes the Zarr data does not have to care how/where the data is stored.
    Multiple implementations are provided. But also wrapper stores for various
    purposes.
    """

    pass


class ReadableStore(BaseStore):
    """A store that can read keys.

    Partial getting is implemented in this base class by using ``.get()``, and
    then taking a slice.
    """

    def get(self, key: str) -> bytes:
        """Retrieve the value associated with a given key."""
        check_key(self, "get", key)
        return self._get(key)

    def _get(self, key: str) -> bytes:
        raise NotImplementedError()

    def get_partial_values(
        self, key_ranges: List[tuple[str, int, int | None]]
    ) -> list[bytes]:
        """Retrieve possibly partial values from given key_ranges.

        The ``key_ranges`` is an iterable of (key, range_start, range_length),
        where range_length may be None to indicate the full remaining length.
        """
        key_ranges2 = []
        for key_range in key_ranges:
            key, start, length = check_key_range(self, key_range)
            check_key(self, "get_partial_values", key)
            start = int(start)
            length = None if length is None else int(length)
            key_ranges2.append((key, start, length))
        return self._get_partial_values(key_ranges2)

    def _get_partial_values(
        self, key_ranges: List[tuple[str, int, int | None]]
    ) -> list[bytes]:
        # Default implementation, using .get()
        result = []
        for key, i1, length in key_ranges:
            i2 = None if length is None else i1 + length
            full_value = self.get(key)
            result.append(full_value[i1:i2])
        return result


class WritableStore(BaseStore):
    """A store that can write and delete keys.

    Partial setting is implemented in this base class by using ``.get()``,
    updating the value, and then ``.set()``. Similarly, ``erase_values()`` and
    ``erase_prefix()`` are implemented in the base class; they can be overridden
    if a subclass can implement it more efficiently.
    """

    def set(self, key: str, value: bytes) -> None:
        """Store a (key, value) pair."""
        check_key(self, "set", key)
        return self._set(key, value)

    def _set(self, key: str, value: bytes):
        raise NotImplementedError()

    def set_partial_values(
        self, key_start_values: List[tuple[str, int, bytes]]
    ) -> None:
        """Store values at a given key, starting at byte range_start."""
        key_start_values2 = []
        for key_start_value in key_start_values:
            key, start, value = check_key_start_value(self, key_start_value)
            check_key(self, "set_partial_values", key)
            start = int(start)
            key_start_values2.append((key, start, value))
        self._set_partial_values(key_start_values2)

    def _set_partial_values(self, key_start_values: List[tuple[str, int, bytes]]):
        # Default implementation, using .get()
        for key, i1, value in key_start_values:
            i2 = i1 + len(value)
            full_value = self.get(key)
            full_value = full_value[:i1] + value + full_value[i2:]
            self.set(key, full_value)

    def erase(self, key: str) -> None:
        """Erase the given key/value pair from the store."""
        check_key(self, "erase", key)
        self._erase(key)

    def _erase(self, key: str):
        raise NotImplementedError()

    def erase_values(self, keys: List[str]) -> None:
        """Erase the given key/value pairs from the store."""
        keys = list(keys)
        for key in keys:
            check_key(self, "erase_values", key)
        self._erase_values(keys)

    def _erase_values(self, keys: List[str]):
        # Default implementation, using .erase()
        for key in keys:
            self.erase(key)

    def erase_prefix(self, prefix: str):
        """Erase all keys with the given prefix from the store.

        The prefix represents a 'directory'; it must end with a '/'.
        """
        check_prefix(self, "erase_values", prefix)
        return self._erase_prefix(prefix)

    def _erase_prefix(self, prefix: str):
        # Default implementation, using .list_prefix()
        for key in self.list_prefix(prefix):
            self.erase(key)


class ListableStore(BaseStore):
    """A store that can list keys.

    Although ``list_prefix()`` and ``list_dir()`` are implemented in this base
    class, subclasses can likely implement them more efficiently.
    """

    def list(self) -> list[str]:
        """Retrieve all keys in the store."""
        return self._list()

    def _list(self) -> list[str]:
        raise NotImplementedError()

    def list_prefix(self, prefix: str) -> List[str]:
        """Retrieve all keys with a given prefix.

        The prefix represents a 'directory'; it must end with a '/'. This method
        lists the full (recursive) list of items in that directory.

        For example, if a store contains the keys “a/b”, “a/c/d” and “e/f/g”,
        then ``list_prefix("a/")`` would return “a/b” and “a/c/d”.
        """
        check_prefix(self, "list_prefix", prefix)
        return self._list_prefix(prefix)

    def _list_prefix(self, prefix: str) -> List[str]:
        # Default implementation, using .list()
        prefix = "" if prefix == "/" else prefix  # Special case
        return [key for key in self.list() if key.startswith(prefix)]

    def list_dir(self, prefix: str) -> List[str]:
        """Retrieve all keys within a given directory.

        The prefix represents a 'directory'; it must end with a '/'. This method
        lists only the keys in that directory and not in that of any
        subdirectories. But it does return prefixes (i.e. directories) within
        the given directory.

        For example, if a store contains the keys “a/b”, “a/c”, “a/d/e”,
        “a/f/g”, then ``list_dir("a/")`` would return keys “a/b” and “a/c” and
        prefixes “a/d/” and “a/f/”. ``list_dir("b/")`` would return the empty
        set.
        """
        check_prefix(self, "list_dir", prefix)
        return self._list_dir(prefix)

    def _list_dir(self, prefix: str) -> List[str]:
        # Default implementation, using .list()
        prefix = "" if prefix == "/" else prefix  # Special case
        n = len(prefix)
        keys = set()
        for key in self.list():
            if key.startswith(prefix):
                key, dash, _rest = key[n:].partition("/")
                keys.add(prefix + key + dash)
        return sorted(keys)


# %%%%% Implementations


class MemoryStore(ReadableStore, WritableStore, ListableStore):
    """Implementation of a readable, writable and listable store, based on an in-memory dict."""

    def __init__(self, fields: dict | None = None):
        self._store = {}
        if fields:
            for k, v in fields.items():
                self.set(k, v)

    def _get(self, key: str) -> bytes:
        try:
            return self._store[key]
        except KeyError:
            raise IOError(f"get(): key {key!r} does not exist.") from None

    def _set(self, key: str, value: bytes):
        dir = ""
        for d in key.split("/")[:-1]:
            dir += f"{d}/"
            if dir[:-1] in self._store:
                raise IOError(f"Cannot set {key!r} because {dir[:-1]!r} exists.")
        self._store[key] = value

    def _erase(self, key: str):
        try:
            self._store.pop(key)
        except KeyError:
            raise IOError(f"erase(): key {key!r} does not exist.") from None

    def _list(self) -> list[str]:
        return sorted(self._store.keys())


class LocalStore(ReadableStore, WritableStore, ListableStore):
    """Implementation of a readable, writable and listable store, based on the local file system.

    The given path represents the root of the store.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)

    def __repr__(self):
        return f"<LocalStore '{self._path}' at {hex(id(self))}>"

    def _get(self, key: str) -> bytes:
        p = self._path.joinpath(*key.split("/"))
        if p.is_file():
            return p.read_bytes()
        else:
            raise IOError(f"get(): key {key!r} does not exist.")

    def _get_partial_values(
        self, key_ranges: List[tuple[str, int, int | None]]
    ) -> list[bytes]:
        result = []
        for key, start, length in key_ranges:
            p = self._path.joinpath(*key.split("/"))
            if not p.is_file():
                raise IOError(f"Key {key!r} does not exist.")
            with p.open("rb") as f:
                f.seek(start)
                result.append(f.read(length))
        return result

    def _set(self, key: str, value: bytes):
        p = self._path.joinpath(*key.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(value)

    def _set_partial_values(self, key_start_values: List[tuple[str, int, bytes]]):
        for key, start, value in key_start_values:
            p = self._path.joinpath(*key.split("/"))
            p.parent.mkdir(parents=False, exist_ok=True)
            with p.open("r+b") as f:
                f.seek(start)
                f.write(value)

    def _erase(self, key: str):
        p = self._path.joinpath(*key.split("/"))
        if p.is_file():
            p.unlink()
        else:
            raise IOError(f"erase(): key {key!r} does not exist.")

    # def _erase_values(self, keys: List[str]):  ->  use default implementation
    # def _erase_prefix(self, prefix: str):  ->  use default implementation, could use shutil.rmtree if we want to optimize this

    def _list(self) -> list[str]:
        return sorted(
            [
                p.relative_to(self._path).as_posix()
                for p in self._path.rglob("*")
                if p.is_file()
            ]
        )

    def _list_prefix(self, prefix: str) -> list[str]:
        d = self._path.joinpath(*prefix.split("/"))
        return sorted(
            [p.relative_to(self._path).as_posix() for p in d.rglob("*") if p.is_file()]
        )

    def _list_dir(self, prefix: str) -> list[str]:
        d = self._path.joinpath(*prefix.split("/"))
        if not d.is_dir():
            return []
        keys = set()
        for p in d.iterdir():
            key = p.relative_to(self._path).as_posix()
            dash = "/" if p.is_dir() else ""
            keys.add(key + dash)
        return sorted(keys)


class WrapperStore(ReadableStore, WritableStore, ListableStore):
    """A store that wraps another store.

    Subclasses can implement a method ``_hook(method, args, result)`` to implement specific behaviour on each API call.
    """

    def __init__(self, store: ReadableStore | WritableStore | ListableStore):
        self._store = store

    def _hook(self, method, args, result):
        """The hook that gets called on each API call."""
        pass

    def _get(self, key: str) -> bytes:
        result = self._store.get(key)
        self._hook("get", (key,), result)
        return result

    def _get_partial_values(
        self, key_ranges: List[tuple[str, int, int | None]]
    ) -> list[bytes]:
        result = self._store._get_partial_values(key_ranges)
        self._hook("get_partial_values", (key_ranges,), result)
        return result

    def _set(self, key: str, value: bytes):
        result = self._store._set(key, value)
        self._hook("set", (key, value), result)
        return result

    def _set_partial_values(self, key_start_values: List[tuple[str, int, bytes]]):
        result = self._store._set_partial_values(key_start_values)
        self._hook("set_partial_values", (key_start_values,), result)
        return result

    def _erase(self, key: str):
        result = self._store._erase(key)
        self._hook("erase", (key,), result)
        return result

    def _erase_values(self, keys: List[str]):
        result = self._store._erase_values(keys)
        self._hook("erase_values", (keys,), result)
        return result

    def _erase_prefix(self, prefix: str):
        result = self._store._erase_prefix(prefix)
        self._hook("erase_prefix", (prefix), result)
        return result

    def _list(self) -> list[str]:
        result = self._store._list()
        self._hook("list", (), result)
        return result

    def _list_prefix(self, prefix: str) -> list[str]:
        result = self._store._list_prefix(prefix)
        self._hook("list_prefix", (prefix,), result)
        return result

    def _list_dir(self, prefix: str) -> list[str]:
        result = self._store._list_dir(prefix)
        self._hook("list_dir", (prefix,), result)
        return result


class SlowStore(WrapperStore):
    """A store that has a fixed time delay for reads and writes."""

    def __init__(
        self,
        store: ReadableStore | WritableStore | ListableStore,
        base_delay: float = 1.0,
        bits_per_second: float = 0.0,
    ):
        super().__init__(store)
        self._base_delay = base_delay
        self._bits_per_second = bits_per_second

    def _sleep(self, nbytes: int):
        delay = self._base_delay

        if self._bits_per_second > 0:
            delay += (nbytes * 8) / self._bits_per_second

        if delay > 0:
            time.sleep(delay)

    def _hook(self, method, args, result):
        # read delay
        if method == "get":
            self._sleep(len(result))

        elif method == "get_partial_values":
            self._sleep(sum(len(x) for x in result))

        # write delay
        elif method == "set":
            _key, value = args
            self._sleep(len(value))

        elif method == "set_partial_values":
            (key_start_values,) = args
            self._sleep(sum(len(value) for _, _, value in key_start_values))

        # base delay
        else:
            if self._base_delay > 0:
                time.sleep(self._base_delay)


# More store ideas:
#
# class LoggingStore:
# class ZipStore:
# class S3Store:
# class HttpStore:
