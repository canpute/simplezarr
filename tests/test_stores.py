from pathlib import Path
import errno
import shutil
import stat
import threading
import time

from simplezarr.stores import (
    BaseStore,
    ReadableStore,
    WritableStore,
    ListableStore,
    SlowStore,
    SftpStore,
    RetryStore,
)
from simplezarr.stores import _map_sftp_error
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


# %%%%% An in-memory fake of an SFTP server, so that the SftpStore can be tested
# without paramiko or a server. It mimics the semantics of a real SFTP server:
# files are stored under absolute posix paths, and errors are raised as OSError
# instances (with an errno), like paramiko does.


class FakeSftpAttr:
    def __init__(self, filename, st_mode):
        self.filename = filename
        self.st_mode = st_mode


class FakeSftpFile:
    def __init__(self, fs, path, mode):
        self._fs = fs
        self._path = path
        self._mode = mode
        self._pos = 0
        if mode in ("w", "wb"):
            fs._files[path] = bytearray()

    def seek(self, pos):
        self._pos = pos
        return self._pos

    def read(self, length=None):
        data = self._fs._files[self._path]
        i2 = len(data) if length is None else min(len(data), self._pos + length)
        result = bytes(data[self._pos : i2])
        self._pos = i2
        return result

    def write(self, value):
        data = self._fs._files[self._path]
        value = bytes(value)
        data[self._pos : self._pos + len(value)] = value
        self._pos += len(value)
        return len(value)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
        return False


class FakeSftpClient:
    """In-memory stand-in for a paramiko SFTPClient."""

    def __init__(self, fields=None):
        self._files = {}  # absolute posix path -> bytearray
        self.closed = False
        if fields:
            for key, value in fields.items():
                self._files["/" + key] = bytearray(value)

    def close(self):
        self.closed = True

    def _is_dir(self, path):
        # Directories are registered implicitly, by the files that are in them
        return any(k.startswith(path.rstrip("/") + "/") for k in self._files)

    def _check_ancestors(self, path):
        # Raise if any ancestor of the given path is a file
        parts = [p for p in path.split("/") if p]
        for i in range(1, len(parts)):
            p = "/" + "/".join(parts[:i])
            if p in self._files:
                raise OSError(errno.ENOTDIR, "Not a directory", p)

    def open(self, path, mode="r"):
        self._check_ancestors(path)
        if mode in ("w", "wb"):
            if path in self._files or self._is_dir(path):
                raise OSError(errno.EISDIR, "Is a directory", path)
            return FakeSftpFile(self, path, mode)
        if self._is_dir(path):
            raise OSError(errno.EISDIR, "Is a directory", path)
        if path not in self._files:
            raise OSError(errno.ENOENT, "No such file", path)
        return FakeSftpFile(self, path, mode)

    def remove(self, path):
        if path not in self._files:
            raise OSError(errno.ENOENT, "No such file", path)
        del self._files[path]

    def mkdir(self, path):
        self._check_ancestors(path)
        if path in self._files:
            raise OSError(errno.EEXIST, "File exists", path)
        if self._is_dir(path):
            raise OSError(errno.EEXIST, "File exists", path)
        # Else no-op: directories are registered implicitly

    def stat(self, path):
        if path in self._files:
            return FakeSftpAttr(path, stat.S_IFREG | 0o644)
        if self._is_dir(path):
            return FakeSftpAttr(path, stat.S_IFDIR | 0o755)
        raise OSError(errno.ENOENT, "No such file", path)

    def listdir_attr(self, path):
        if path in self._files:
            raise OSError(errno.ENOTDIR, "Not a directory", path)
        if not self._is_dir(path):
            raise OSError(errno.ENOENT, "No such file or directory", path)
        prefix = path.rstrip("/") + "/"
        entries = {}  # filename -> is a dir
        for p in self._files:
            if p.startswith(prefix):
                rest = p[len(prefix) :]
                name, _, _ = rest.partition("/")
                entries[name] = "/" in rest
        return [
            FakeSftpAttr(name, (stat.S_IFDIR if is_dir else stat.S_IFREG) | 0o644)
            for name, is_dir in entries.items()
        ]


class SftpStoreTestable(SftpStore):
    def __init__(self):
        super().__init__("", "/", sftp_client=FakeSftpClient(STORE))


class FlakySftpClient(FakeSftpClient):
    """A fake SFTP client that raises on the first ``fail_times`` open() calls."""

    def __init__(self, fail_times=0, exc=None):
        super().__init__(STORE)
        self._fail_times = fail_times
        self._exc = TimeoutError("flaky sftp: too slow") if exc is None else exc
        self.calls = 0

    def open(self, path, mode="r"):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return super().open(path, mode)


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
    SftpStoreTestable,
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


def test_sftp_store_paths():
    # The keys map to files on the server, relative to the root path
    fake = FakeSftpClient()
    store = SftpStore("", "/", sftp_client=fake)
    store.set("a/b", b"hi ab")
    assert sorted(fake._files) == ["/a/b"]
    assert store.list() == ["a/b"]
    assert "SftpStore" in repr(store)

    # And with a non-root path
    fake = FakeSftpClient()
    store = SftpStore("", "data/store", sftp_client=fake)
    assert store.path == "/data/store"
    store.set("a/b", b"hi ab")
    store.set("a/c", b"hi ac")
    assert sorted(fake._files) == ["/data/store/a/b", "/data/store/a/c"]
    assert store.list() == ["a/b", "a/c"]
    assert store.list_prefix("a/") == ["a/b", "a/c"]
    assert store.list_dir("/") == ["a/"]
    assert store.list_dir("a/") == ["a/b", "a/c"]

    # Keys are always server paths, also for partial reads/writes
    assert store.get_partial_values([("a/b", 3, 2)]) == [b"ab"]
    store.set_partial_values([("a/b", 0, b"HI")])
    assert fake._files["/data/store/a/b"] == b"HI ab"


def test_sftp_store_errors():
    store = SftpStoreTestable()

    # Missing keys map to FileNotFoundError (retryable errors map to
    # TimeoutError/ConnectionError, see the other tests)
    with pytest.raises(FileNotFoundError):
        store.get("doesnotexist")
    with pytest.raises(FileNotFoundError):
        store.get_partial_values([("doesnotexist", 0, 2)])
    with pytest.raises(FileNotFoundError):
        store.set_partial_values([("doesnotexist", 0, b"x")])
    with pytest.raises(FileNotFoundError):
        store.erase("doesnotexist")

    # Cannot treat a file as a dir
    with pytest.raises(NotADirectoryError):
        store.set("foo/eggs", b"hi eggs")
    store._dirs.clear()  # Also try again without the dir cache
    with pytest.raises(NotADirectoryError):
        store.set("foo/eggs", b"hi eggs")

    # Cannot treat a dir as a file
    with pytest.raises(IsADirectoryError):
        store.set("dir1", b"hi dir1")


def test_sftp_store_map_errors():
    # Builtin errors (including TypeError etc.) are returned unchanged
    for err in [
        TimeoutError("too slow"),
        ConnectionError("broken"),
        FileNotFoundError("nope"),
        PermissionError("no"),
        TimeoutError(),
        TypeError("wrong type"),
        ValueError("wrong value"),
        OSError("something generic"),
    ]:
        assert _map_sftp_error(err) is err

    # EOFError means the connection was closed
    assert isinstance(_map_sftp_error(EOFError()), ConnectionError)

    # OSErrors with an errno map to the builtin error of the taxonomy
    cases = [
        (errno.ENOENT, FileNotFoundError),
        (errno.EACCES, PermissionError),
        (errno.EPERM, PermissionError),
        (errno.ENOTDIR, NotADirectoryError),
        (errno.EISDIR, IsADirectoryError),
        (errno.ETIMEDOUT, TimeoutError),
        (errno.ECONNREFUSED, ConnectionError),
        (errno.ECONNRESET, ConnectionError),
        (errno.EPIPE, ConnectionError),
        (errno.EBADF, OSError),  # anything else stays an OSError
    ]
    for errno_, cls in cases:
        mapped = _map_sftp_error(OSError(errno_, "fake"))
        assert isinstance(mapped, cls), f"errno {errno_} should map to {cls.__name__}"
        assert isinstance(mapped, OSError)


def test_sftp_store_retry_integration():
    # A transient error from the SFTP transport is retried by the RetryStore
    flaky = FlakySftpClient(fail_times=1)
    store = RetryStore(SftpStore("", "/", sftp_client=flaky), base_sleep=0.001)
    assert store.get("foo") == b"hi foo"
    assert flaky.calls == 2

    # And with all attempts failing, the RetryStore raises a TimeoutError
    flaky = FlakySftpClient(fail_times=10_000)
    store = RetryStore(
        SftpStore("", "/", sftp_client=flaky),
        timeout=0.05,
        base_sleep=0.001,
        max_sleep=0.002,
    )
    with pytest.raises(TimeoutError):
        store.get("foo")
    assert flaky.calls > 10


def _make_pooled_store(state, connections, client_factory):
    """Create a SftpStore with a fake SSH connection, so that the channel
    pool can be tested without paramiko. The factory is called to create
    each SFTP channel; ``state`` counts what happens to them.
    """

    class FakeSshClient:
        def __init__(self):
            self._is_active = True
            self.channels = []

        def get_transport(self):
            return self

        def is_active(self):
            return self._is_active

        def open_sftp(self):
            sftp = client_factory()
            state["channels"] += 1
            self.channels.append(sftp)
            return sftp

        def close(self):
            self._is_active = False

    class PooledSftpStore(SftpStore):
        def _create_ssh_client(self):
            state["ssh_connections"] += 1
            return FakeSshClient()

    return PooledSftpStore("", "/", connections=connections)


def _make_instrumented_factory(state, op_duration=0.0, flaky_first=False):
    guard = threading.Lock()
    flaky_done = []  # mutable, to allow sharing between factory calls

    def factory():
        if flaky_first and not flaky_done:
            flaky_done.append(True)
            return FlakySftpClient(fail_times=1)
        sftp = FakeSftpClient(STORE)
        orig_open = sftp.open

        def open_(path, mode="r"):
            with guard:
                state["n_open"] += 1
                state["max_open"] = max(state["max_open"], state["n_open"])
            if op_duration:
                time.sleep(op_duration)
            try:
                return orig_open(path, mode)
            finally:
                with guard:
                    state["n_open"] -= 1

        sftp.open = open_
        return sftp

    return factory


def test_sftp_store_single_channel_reused():
    # With connections=1 (the default), all operations use a single channel
    state = {"channels": 0, "ssh_connections": 0, "n_open": 0, "max_open": 0}
    store = _make_pooled_store(
        state, connections=1, client_factory=_make_instrumented_factory(state)
    )
    for _i in range(6):
        assert store.get("foo") == b"hi foo"
    assert state["channels"] == 1
    assert state["ssh_connections"] == 1
    store.close()


def test_sftp_store_parallel_channels():
    # With connections=4, multiple operations are in flight at the same time,
    # and the store never uses more channels than that
    state = {"channels": 0, "ssh_connections": 0, "n_open": 0, "max_open": 0}
    store = _make_pooled_store(
        state,
        connections=4,
        client_factory=_make_instrumented_factory(state, op_duration=0.005),
    )

    def worker():
        for _i in range(15):
            assert store.get("foo") == b"hi foo"

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["channels"] <= 4
    assert state["max_open"] >= 2  # Without parallelism this would be 1
    assert state["max_open"] <= 4
    store.close()


def test_sftp_store_broken_channel_replaced():
    # A transient error closes the channel on which it occurred; the next
    # operation uses a new one (or another one from the pool)
    state = {"channels": 0, "ssh_connections": 0, "n_open": 0, "max_open": 0}
    store = _make_pooled_store(
        state,
        connections=2,
        client_factory=_make_instrumented_factory(state, flaky_first=True),
    )

    with pytest.raises(TimeoutError):
        store.get("foo")
    # The broken channel is closed, and the next attempt works
    assert _first_channel(store).closed is True
    assert store.get("foo") == b"hi foo"
    assert store.get("bar") == b"hi bar"
    store.close()


def _first_channel(store):
    # The store's fake SSH client keeps track of the channels that it opened
    return store._client.channels[0]


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
