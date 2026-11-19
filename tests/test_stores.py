from pathlib import Path
import shutil
import time

from simplezarr.stores import (
    BaseStore,
    ReadableStore,
    WritableStore,
    ListableStore,
    SlowStore,
    RetryStore,
)
from simplezarr.stores import check_key, check_prefix
from simplezarr.stores import MemoryStore, LocalStore, WrapperStore

import pytest
import numpy as np


List = list

STORE = {
    "foo": b"hi foo",
    "bar": b"hi bar",
    "dir1/foo": b"hi foo",
    "dir1/bar": b"hi bar",
    "dir2/index": b"hi index",
    "dir2/sub/foo": b"hi foo",
    "dir2/sub/bar": b"hi bar",
}

test_dir1 = Path(__file__).absolute().parent / "test-data"


def setup_module():
    shutil.rmtree(test_dir1, ignore_errors=True)
    for k, v in STORE.items():
        f = test_dir1.joinpath(*k.split("/"))
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(v)


def teardown_module():
    shutil.rmtree(test_dir1)


# %%%%% Define classes that minimally extend the base classes so they can be tested easily


class StoreThatFillsTheGaps:
    def __init__(self):
        self._store = STORE.copy()

    def _get(self, key: str) -> bytes:
        try:
            return self._store[key]
        except KeyError:
            raise FileNotFoundError(f"get(): key {key!r} does not exist.") from None

    def _set(self, key: str, value: bytes):
        dir = ""
        for d in key.split("/")[:-1]:
            dir += f"{d}/"
            if dir[:-1] in self._store:
                raise NotADirectoryError(
                    f"Cannot set {key!r} because {dir[:-1]!r} exists."
                )
        key_as_dir = key + "/"
        if any(k.startswith(key_as_dir) for k in self._store.keys()):
            raise IsADirectoryError(f"Cannot set {key!r} because its a dir.")
        if not isinstance(value, bytes):
            value = value.tobytes()
        self._store[key] = value

    def _erase(self, key: str):
        try:
            self._store.pop(key)
        except KeyError:
            raise FileNotFoundError(f"erase(): key {key!r} does not exist.") from None

    def _list(self) -> list[str]:
        return sorted(self._store.keys())


class ReadableStoreTestable(StoreThatFillsTheGaps, ReadableStore):
    pass


class WritableStoreTestable(StoreThatFillsTheGaps, WritableStore):
    def _list_prefix(self, prefix: str) -> List[str]:
        prefix = "" if prefix == "/" else prefix  # Special case
        return [key for key in self._list() if key.startswith(prefix)]


class ListableStoreTestable(StoreThatFillsTheGaps, ListableStore):
    pass


class MemoryStoreTestable(MemoryStore):
    def __init__(self):
        super().__init__(STORE)


class LocalStoreTestable(LocalStore):
    def __init__(self):
        super().__init__(test_dir1)


class WrapperStoreTestable(WrapperStore):
    def __init__(self):
        super().__init__(MemoryStore(STORE))


class SlowStoreTestable(SlowStore):
    def __init__(self):
        super().__init__(
            MemoryStore(STORE),
            base_delay=0.001,
            bits_per_second=10_000,
            delay_variation=0.001,
        )


class RetryStoreTestable(RetryStore):
    def __init__(self):
        super().__init__(MemoryStore(STORE), base_sleep=0.001, max_sleep=0.002)


class FlakyStore(MemoryStore):
    """A MemoryStore that raises the given exception on the first ``fail_times``
    calls of each operation. For testing retry behavior.
    """

    def __init__(self, fail_times=0, exc=None, fields=None):
        self._fail_times = fail_times
        self._exc = TimeoutError("flaky store: too slow") if exc is None else exc
        self.calls = {}  # op name -> number of calls
        super().__init__()
        if fields:
            self._dict.update(fields)

    def _tick(self, op):
        count = self.calls.get(op, 0) + 1
        self.calls[op] = count
        if count <= self._fail_times:
            raise self._exc

    def _get(self, key):
        self._tick("get")
        return super()._get(key)

    def _set(self, key, value):
        self._tick("set")
        return super()._set(key, value)

    def _erase(self, key):
        self._tick("erase")
        return super()._erase(key)

    def _list(self):
        self._tick("list")
        return super()._list()

    # The methods below are implemented directly on the internal dict, because
    # the default implementations call other (also flaky) methods, which would
    # make the attempt counts depend on the nesting of the default methods.

    def _get_partial_values(self, key_ranges):
        self._tick("get_partial_values")
        result = []
        for key, i1, length in key_ranges:
            i2 = None if length is None else i1 + length
            result.append(self._dict[key][i1:i2])
        return result

    def _list_dir(self, prefix):
        self._tick("list_dir")
        prefix = "" if prefix == "/" else prefix  # Special case
        n = len(prefix)
        keys = set()
        for key in self._dict:
            if key.startswith(prefix):
                key2, dash, _rest = key[n:].partition("/")
                keys.add(prefix + key2 + dash)
        return sorted(keys)


# %%%%% Test all stores


store_classes = [
    ReadableStoreTestable,
    WritableStoreTestable,
    ListableStoreTestable,
    MemoryStoreTestable,
    LocalStoreTestable,
    WrapperStoreTestable,
    SlowStoreTestable,
    RetryStoreTestable,
]


@pytest.mark.parametrize("cls", store_classes)
def test_read(cls):
    assert isinstance(cls, type)
    assert issubclass(cls, BaseStore)
    if not issubclass(cls, ReadableStore):
        pytest.skip()

    store = cls()

    assert store.get("foo") == b"hi foo"
    assert store.get("dir1/bar") == b"hi bar"
    assert store.get("dir2/sub/bar") == b"hi bar"

    key_ranges = [("bar", 3, 2), ("dir2/sub/foo", 3, None)]

    assert store.get_partial_values(key_ranges[0:1]) == [b"ba"]
    assert store.get_partial_values(key_ranges[1:]) == [b"foo"]
    assert store.get_partial_values(key_ranges) == [b"ba", b"foo"]

    with pytest.raises(FileNotFoundError):
        store.get("doesnotexist")

    with pytest.raises(FileNotFoundError):
        store.get_partial_values([("doesnotexist", 1, 2)])

    with pytest.raises(TypeError):
        store.get_partial_values("foo")

    with pytest.raises(TypeError):
        store.get_partial_values(["foo"])


@pytest.mark.parametrize("cls", store_classes)
def test_list(cls):
    assert isinstance(cls, type)
    assert issubclass(cls, BaseStore)
    if not issubclass(cls, ListableStore):
        pytest.skip()

    store = cls()

    files = store.list()
    assert files == sorted(STORE.keys())

    assert store.list_prefix("/") == files
    assert store.list_prefix("dir1/") == ["dir1/bar", "dir1/foo"]
    assert store.list_prefix("dir2/") == ["dir2/index", "dir2/sub/bar", "dir2/sub/foo"]
    assert store.list_prefix("dir2/sub/") == ["dir2/sub/bar", "dir2/sub/foo"]
    assert store.list_prefix("doesnotexist/") == []

    assert store.list_dir("/") == ["bar", "dir1/", "dir2/", "foo"]
    assert store.list_dir("dir1/") == ["dir1/bar", "dir1/foo"]
    assert store.list_dir("dir2/") == ["dir2/index", "dir2/sub/"]
    assert store.list_dir("dir2/sub/") == ["dir2/sub/bar", "dir2/sub/foo"]
    assert store.list_dir("doesnotexist/") == []

    with pytest.raises(ValueError):
        store.list_prefix("dir1")  # must end with '/'

    with pytest.raises(ValueError):
        store.list_dir("dir1")  # must end with '/'


@pytest.mark.parametrize("cls", store_classes)
def test_write(cls):
    assert isinstance(cls, type)
    assert issubclass(cls, BaseStore)
    if not issubclass(cls, WritableStore):
        pytest.skip()

    store = cls()

    store.set("spam", b"hi spam")
    store.set("dir3/sub/spam", b"hi spam")

    mem1 = memoryview(b"012345")
    mem2 = mem1[::2]  # not contiguous
    store.set("spam2", mem1)
    store.set("spam3", mem2)

    with pytest.raises(TypeError):
        store.set("spam4", [b"cannot_write_a_list"])
    with pytest.raises(TypeError):
        store.set("spam4", np.array([1, 2, 3]))

    if True:  # isinstance(store, ReadableStore):
        assert store._get("spam") == b"hi spam"
        assert store._get("dir3/sub/spam") == b"hi spam"
        assert store._get("spam2") == b"012345"
        assert store._get("spam3") == b"024"
    if True:  # isinstance(store, ListableStore):
        assert "spam" in store._list()
        assert "dir3/sub/spam" in store._list()

    store.set_partial_values([("foo", 1, b"a-")])
    if True:  # isinstance(store, ReadableStore):
        assert store._get("foo") == b"ha-foo"

    store.set_partial_values([("foo", 3, mem2)])
    if True:  # isinstance(store, ReadableStore):
        assert store._get("foo") == b"ha-024"

    with pytest.raises(TypeError):
        store.set_partial_values([("spam", 1, b"a-", 0)])
    with pytest.raises(IOError):
        store.set_partial_values([("doesnotexist", 1, b"a-")])
    with pytest.raises(TypeError):
        store.set_partial_values([("spam", 1, "not-bytes")])

    # Cannot create a dir for a file with that name
    with pytest.raises(IOError):
        store.set("foo/eggs", b"hi eggs")
    with pytest.raises(IOError):
        store.set("dir2/sub/bar/eggs", b"hi eggs")
    with pytest.raises(IOError):
        store.set("dir2", b"hi dir2")

    # Erase

    store.erase("foo")
    with pytest.raises(FileNotFoundError):
        store._get("foo")
    store.set("foo/eggs", b"hi eggs")

    store.erase("dir2/sub/bar")
    with pytest.raises(FileNotFoundError):
        store._get("dir2/sub/bar")
    store.set("dir2/sub/bar/eggs", b"hi eggs")

    with pytest.raises(FileNotFoundError):
        store.erase("doesnotexist")
    with pytest.raises(FileNotFoundError):
        store.erase("foo")

    # Erase values

    store.erase_values(["bar", "dir1/bar"])
    assert "bar" not in store._list()
    assert "dir1/bar" not in store._list()

    with pytest.raises(FileNotFoundError):
        store.erase_values(["doesnotexist"])

    # Erase prefix

    store.erase_prefix("dir2/")
    store.erase_prefix("dir3/")
    assert store._list() == ["dir1/foo", "foo/eggs", "spam", "spam2", "spam3"]

    store.erase_prefix("foo/")
    assert store._list() == ["dir1/foo", "spam", "spam2", "spam3"]

    store.erase_prefix("/")
    assert store._list() == []

    store.erase_prefix("doesnotexist/")  # This is fine


def test_memory_store():
    store = MemoryStoreTestable()
    assert isinstance(store.dict, dict)
    assert store.dict == STORE
    assert store.nbytes > 10
    store.dict.clear()
    assert store.nbytes == 0


def test_local_store():
    store = LocalStoreTestable()
    assert isinstance(store.path, Path)
    assert "test-data" in str(store.path)
    assert "test-data" in repr(store)


def test_wrapper_store():
    store = WrapperStoreTestable()
    assert isinstance(store.sub, BaseStore)


# %%%%% More specific tests


def test_retry_store_retries_transient_errors():
    # TimeoutError and ConnectionError are retried until the store succeeds
    flaky = FlakyStore(fail_times=3, fields={"foo": b"hi foo"})
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    assert store.get("foo") == b"hi foo"
    assert flaky.calls["get"] == 4  # 3 failed tries + 1 success

    flaky = FlakyStore(fail_times=2, exc=ConnectionError("flaky store: no connection"))
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    store.set("foo", b"hi foo")
    assert flaky.calls["set"] == 3  # 2 failed tries + 1 success
    assert store.get("foo") == b"hi foo"

    flaky = FlakyStore(fail_times=1, fields=STORE)
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    assert store.list_dir("/") == ["bar", "dir1/", "dir2/", "foo"]
    assert flaky.calls["list_dir"] == 2

    flaky = FlakyStore(fail_times=1, fields=STORE)
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    assert store.list() == sorted(STORE)
    assert flaky.calls["list"] == 2

    flaky = FlakyStore(fail_times=1, fields=STORE)
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    store.erase("foo")
    assert flaky.calls["erase"] == 2
    assert "foo" not in store.list()

    # Partial reads and writes are retried too
    flaky = FlakyStore(fail_times=1, fields={"foo": b"hi foo"})
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    assert store.get_partial_values([("foo", 1, 2)]) == [b"i "]
    assert flaky.calls["get_partial_values"] == 2


def test_retry_store_raises_timeout():
    # When the store keeps failing, a TimeoutError is raised when the
    # total timeout expires
    flaky = FlakyStore(fail_times=1_000_000, fields={"foo": b"hi foo"})
    store = RetryStore(flaky, timeout=0.1, base_sleep=0.001, max_sleep=0.002)

    t0 = time.perf_counter()
    with pytest.raises(TimeoutError) as exc_info:
        store.get("foo")
    dt = time.perf_counter() - t0

    assert 0.09 <= dt < 5
    assert flaky.calls["get"] > 10  # It really tried again and again
    # The message says what operation failed and how often, and the store's
    # own error is the cause
    assert "get" in str(exc_info.value)
    assert "tries" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, TimeoutError)


def test_retry_store_non_transient_errors():
    # Errors that are not transient are not retried, but raised immediately
    flaky = FlakyStore(fail_times=2, exc=FileNotFoundError("flaky store: gone"))
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    with pytest.raises(FileNotFoundError):
        store.get("foo")
    assert flaky.calls["get"] == 1

    flaky = FlakyStore(fail_times=2, exc=PermissionError("flaky store: denied"))
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    with pytest.raises(PermissionError):
        store.get("foo")
    assert flaky.calls["get"] == 1


def test_retry_store_max_attempts():
    # A max number of attempts can be given, so we do not hammer a
    # fast-failing server
    flaky = FlakyStore(fail_times=1_000_000, fields={"foo": b"hi foo"})
    store = RetryStore(
        flaky, timeout=10, max_attempts=3, base_sleep=0.001, max_sleep=0.002
    )

    with pytest.raises(TimeoutError):
        store.get("foo")
    assert flaky.calls["get"] == 3


def test_retry_store_timeout_from_store():
    # Without an explicit timeout, the timeout of the wrapped store is used
    flaky = FlakyStore(fail_times=1, fields={"foo": b"hi foo"})
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)
    assert store._timeout == BaseStore.timeout

    flaky = FlakyStore(fail_times=1_000_000, fields={"foo": b"hi foo"})
    flaky.timeout = 0.1
    store = RetryStore(flaky, base_sleep=0.001, max_sleep=0.002)

    with pytest.raises(TimeoutError):
        store.get("foo")


def test_meta():
    classes = set()
    for name, ob in globals().items():
        if name.endswith("Testable"):
            classes.add(ob)

    assert classes == set(store_classes)


def test_check_key():
    check = lambda key: check_key(None, "test", key)

    # Ok
    for key in ["foo", "x", "x.y", "x.y.z", "foo/bar", "foo/bar/spam/eggs.xyz"]:
        check(key)

    # Not ok because not str
    for key in [b"foo", ("foo",), 42]:
        with pytest.raises(TypeError):
            check(key)

    # Not ok because path sep
    for key in ["/", "/foo", "/foo/bar/spam", "foo/", "foo/bar/spam/"]:
        with pytest.raises(ValueError):
            check(key)

    # Not ok because name part
    for key in [
        "",
        ".",
        "..",
        "...",
        "foo/bar/.",
        "foo/./bar",
        "./foo/bar",
        "foo//bar",
    ]:
        with pytest.raises(ValueError):
            check(key)


def test_check_prefix():
    check = lambda key: check_prefix(None, "test", key)

    # Ok
    for key in ["foo/", "x/", "x.y/", "x.y.z/", "foo/bar/", "foo/bar/spam/eggs.xyz/"]:
        check(key)

    # Not ok because not str
    for key in [b"foo", ("foo",), 42]:
        with pytest.raises(TypeError):
            check(key)

    # Not ok because path sep
    for key in ["/foo", "/foo/bar/spam", "foo", "foo/bar/spam"]:
        with pytest.raises(ValueError):
            check(key)

    # Not ok because name part
    for key in [
        "",
        "./",
        "../",
        ".../",
        "foo/bar/./",
        "foo/./bar/",
        "./foo/bar/",
        "foo//bar/",
    ]:
        with pytest.raises(ValueError):
            check(key)


if __name__ == "__main__":
    setup_module()
    try:
        for func in [test_read, test_list, test_write]:
            for cls in store_classes:
                print(f"{func.__name__}[{cls.__name__}] ... ", end="")
                try:
                    func(cls)
                except pytest.skip.Exception:
                    print("skip")
                else:
                    print("done")
        print("all done")

        test_memory_store()
        test_local_store()
        test_wrapper_store()

        test_meta()
        test_check_key()
        test_check_prefix()

    finally:
        teardown_module()
