import numpy as np
import pytest

from simplezarr import MemoryStore, ZarrArray, ZarrGroup, open_zarr
from simplezarr.utils.write_multiscale import (
    _get_levels_per_step,
    _step_nbytes,
    write_ome_zarr_pyramid,
    PyramidData,
)


def test_write_multiscale_even():
    for max_memory in [0, 1]:  # trigger simple and advanced alg, respectively
        s = MemoryStore()

        a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, np.float32)
        write_ome_zarr_pyramid(
            s, "", "yx", PyramidData(a), nlevels=3, max_memory=max_memory
        )

        a0 = open_zarr(s, "level0")[...].get_now()
        a1 = open_zarr(s, "level1")[...].get_now()
        a2 = open_zarr(s, "level2")[...].get_now()

        with pytest.raises(IOError):
            open_zarr(s, "level3")

        assert a0.shape == (8, 8)
        assert a1.shape == (4, 4)
        assert a2.shape == (2, 2)

        assert np.all(a0[0] == range(1, 9))
        assert np.all(a1[0] == [1.5, 3.5, 5.5, 7.5])
        assert np.all(a2[0] == [2.5, 6.5])


def test_write_multiscale_uneven():
    for max_memory in [0, 1]:  # trigger simple and advanced alg, respectively
        s = MemoryStore()

        a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] * 11, np.float32)
        write_ome_zarr_pyramid(
            s, "", "yx", PyramidData(a), nlevels=3, max_memory=max_memory
        )

        a0 = open_zarr(s, "level0")[...].get_now()
        a1 = open_zarr(s, "level1")[...].get_now()
        a2 = open_zarr(s, "level2")[...].get_now()

        with pytest.raises(IOError):
            open_zarr(s, "level3")

        assert a0.shape == (11, 11)
        assert a1.shape == (5, 5)
        assert a2.shape == (2, 2)

        assert np.all(a0[0] == range(1, 12))
        assert np.all(a1[0] == [1.5, 3.5, 5.5, 7.5, 9.5])
        assert np.all(a2[0] == [2.5, 6.5])


def _downsample_reference(a, zyx_ndim):
    """Reference local-mean downsampling of the last `zyx_ndim` dimensions."""
    # Drop the trailing odd pixels, just like the downsampling in write_ome_zarr_pyramid
    slices = tuple(slice(0, s // 2 * 2, 1) for s in a.shape[-zyx_ndim:])
    b = a[..., *slices]
    # Fold each 2x2 (or 2x2x2) block into a mean
    shape = list(b.shape[:-zyx_ndim])
    for s in b.shape[-zyx_ndim:]:
        shape += [s // 2, 2]
    b = b.reshape(shape)
    # The size-2 axes are interleaved after the leading dims; mean over those
    axes = tuple(range(b.ndim - 2 * zyx_ndim + 1, b.ndim, 2))
    return b.mean(axis=axes)


def _expected_auto_nlevels(zyx_shape, zyx_chunk):
    """Same rule that write_ome_zarr_pyramid uses to pick the number of levels."""
    n = 1
    while not all(s <= c for s, c in zip(zyx_shape, zyx_chunk, strict=True)):
        next_zyx_shape = tuple(s // 2 for s in zyx_shape)
        if any(s == 0 for s in next_zyx_shape):
            break  # cannot halve the shape any further
        zyx_shape = next_zyx_shape
        n += 1
    return n


def test_write_multiscale_variety():
    cases = [
        # (array_shape, chunk_shape, axes)
        ((8, 8), (4, 4), "yx"),
        ((16, 16), (4, 4), "yx"),
        ((15, 17), (4, 4), "yx"),
        ((11, 11), (3, 5), "yx"),
        ((9, 5), (2, 2), "yx"),
        ((64, 40), (16, 8), "yx"),
        ((5, 16), (4, 8), "yx"),  # spatial dim reducing to 1
        ((7, 5), (16, 16), "yx"),  # chunk larger than the array
        ((33, 65), (32, 32), "yx"),
        ((257, 384), (64, 64), "yx"),
        ((1000, 513), (128, 64), "yx"),
        ((1500, 900), (128, 256), "yx"),
        ((8, 8, 8), (4, 4, 4), "zyx"),
        ((9, 11, 13), (4, 3, 5), "zyx"),
        ((20, 15, 17), (8, 7, 9), "zyx"),
        ((67, 45, 100), (16, 8, 32), "zyx"),
        ((129, 200, 260), (32, 64, 32), "zyx"),
        ((513, 257, 129), (64, 32, 16), "zyx"),
        ((2, 9, 11, 13), (1, 4, 3, 5), "tcyx"),
        ((2, 3, 9, 11, 13), (1, 1, 4, 3, 5), "tczyx"),
    ]

    for shape, chunk, axes in cases:
        zyx_ndim = sum(1 for a in axes if a in "xyz")
        zyx_chunk = tuple(c for c, ax in zip(chunk, axes, strict=True) if ax in "xyz")
        zyx_shape = tuple(s for s, ax in zip(shape, axes, strict=True) if ax in "xyz")

        rng = np.random.default_rng(42)
        src = rng.random(shape).astype(np.float32)

        expected_nlevels = _expected_auto_nlevels(zyx_shape, zyx_chunk)

        # Build the reference pyramid
        ref = [src]
        for _ in range(expected_nlevels - 1):
            ref.append(_downsample_reference(ref[-1], zyx_ndim))

        # The top level fits in a single chunk, and the level below it does not
        assert all(
            s <= c for s, c in zip(ref[-1].shape[-zyx_ndim:], zyx_chunk, strict=True)
        ), (
            shape,
            chunk,
            ref[-1].shape,
        )
        if expected_nlevels >= 2:
            assert any(
                s > c for s, c in zip(ref[-2].shape[-zyx_ndim:], zyx_chunk, strict=True)
            ), (
                shape,
                chunk,
                ref[-2].shape,
            )

        for max_memory in [0, 1]:  # trigger simple and advanced alg, respectively
            s = MemoryStore()
            write_ome_zarr_pyramid(
                s,
                "",
                axes,
                PyramidData(src),
                chunk_shape=chunk,
                max_memory=max_memory,
            )

            with pytest.raises(IOError):
                open_zarr(s, f"level{expected_nlevels}")

            for i in range(expected_nlevels):
                arr = open_zarr(s, f"level{i}")
                assert arr.chunk_shape == chunk, (shape, chunk, i)
                got = arr[...].get_now()
                assert got.shape == ref[i].shape, (shape, chunk, i)
                assert np.allclose(got, ref[i]), (shape, chunk, i)


def test_write_multiscale_nlevels():
    shape = (400, 640)
    chunk = (64, 64)

    rng = np.random.default_rng(42)
    a = rng.random(shape).astype(np.float32)

    # The number of levels when nlevels=0 (auto) is picked by the library
    expected_auto_nlevels = _expected_auto_nlevels(shape, chunk)

    for max_memory in [0, 1]:  # trigger simple and advanced alg, respectively
        for nlevels in [0, 1, 2, 3, 4, 5, 8]:
            s = MemoryStore()
            write_ome_zarr_pyramid(
                s,
                "",
                "yx",
                PyramidData(a),
                nlevels=nlevels,
                chunk_shape=chunk,
                max_memory=max_memory,
            )

            # Count the number of levels
            n = 0
            while True:
                try:
                    open_zarr(s, f"level{n}")
                except IOError:
                    break
                n += 1

            if nlevels == 0:
                assert n == expected_auto_nlevels, (n, expected_auto_nlevels)
            else:
                assert n == nlevels, (nlevels, n)

            # Check that the shapes halve from level to level
            expected_shape = shape
            for i in range(n):
                assert open_zarr(s, f"level{i}").shape == expected_shape, (nlevels, i)
                expected_shape = tuple(s // 2 for s in expected_shape)


def test_get_levels_per_step():
    # For (4, 4) float32 chunks and no t/c dims, a step that covers k levels
    # selects 2**k chunks per dimension, and takes 64 * (1 + 4 + .. + 4**k) bytes.
    assert _step_nbytes(1, 1, (4, 4), 4) == 64 * (1 + 4)
    assert _step_nbytes(2, 1, (4, 4), 4) == 64 * (1 + 4 + 16)
    # For (4, 4, 4) float32 chunks, a step that covers k levels takes
    # 256 * (1 + 8 + .. + 8**k) bytes.
    assert _step_nbytes(1, 1, (4, 4, 4), 4) == 256 * (1 + 8)

    # The number of selected chunks is a power of 2 per spatial dimension
    assert _get_levels_per_step(1, 1, (4, 4), 4, 100) == 1  # always at least 1
    assert _get_levels_per_step(320, 1, (4, 4), 4, 100) == 1
    assert _get_levels_per_step(1344, 1, (4, 4), 4, 100) == 2
    assert _get_levels_per_step(5440, 1, (4, 4), 4, 100) == 3
    # ... but never more levels than the pyramid has levels to create
    assert _get_levels_per_step(10**9, 1, (4, 4), 4, 3) == 3
    assert _get_levels_per_step(10**9, 1, (4, 4), 4, 1) == 1
    # t/c dimensions scale the memory usage
    assert _get_levels_per_step(640, 2, (4, 4), 4, 100) == 1
    assert _get_levels_per_step(2688, 2, (4, 4), 4, 100) == 2
    # 3D
    assert _get_levels_per_step(2304, 1, (4, 4, 4), 4, 100) == 1
    assert _get_levels_per_step(18688, 1, (4, 4, 4), 4, 100) == 2


def test_write_multiscale_depthfirst_max_memory():
    # Different memory constraints lead to different bottom-up strategies:
    # fewer or more levels per step, and either keeping the intermediate top
    # in memory (finishing with the simple approach), or loading the chunks
    # from disk again in the next step. The `steps` argument reports
    # the strategy that was taken.
    #
    # For a (64, 96) float32 array with (4, 4) chunks: a chunk is 64 bytes,
    # the level sizes are [24576, 6144, 1536, 384, 96, 24] bytes, and a step
    # that covers k levels takes 64 * (1 + 4 + .. + 4**k) bytes:
    #   k=1 -> 320, k=2 -> 1344, k=3 -> 5440

    shape = (64, 96)
    chunk = (4, 4)
    nlevels = 6

    rng = np.random.default_rng(42)
    src = rng.random(shape).astype(np.float32)
    ref = [src]
    for _ in range(nlevels - 1):
        ref.append(_downsample_reference(ref[-1], 2))

    configs = [
        # (max_memory, expected steps as (bottom level, levels created, kept in memory))
        (0, [(0, 5, True)]),  # the simple algorithm does everything in one step
        (
            1,
            [(0, 1, False), (1, 1, False), (2, 1, False), (3, 1, False), (4, 1, False)],
        ),
        (
            320,
            [(0, 1, False), (1, 1, False), (2, 1, False), (3, 1, False), (4, 1, False)],
        ),
        (1344, [(0, 2, False), (2, 2, False), (4, 1, False)]),
        (1500, [(0, 2, False), (2, 2, True), (4, 1, True)]),
        (2880, [(0, 2, True), (2, 3, True)]),
        (5440, [(0, 3, False), (3, 2, False)]),
        (6000, [(0, 3, True), (3, 2, True)]),
        (10**9, [(0, 5, False)]),
    ]

    for max_memory, expected_steps in configs:
        s = MemoryStore()
        steps = []
        write_ome_zarr_pyramid(
            s,
            "",
            "yx",
            PyramidData(src),
            chunk_shape=chunk,
            max_memory=max_memory,
            steps=steps,
        )
        got_steps = [(d["level"], d["nlevels"], d["kept_in_memory"]) for d in steps]
        assert got_steps == expected_steps, (max_memory, got_steps)
        for i in range(nlevels):
            got = open_zarr(s, f"level{i}")[...].get_now()
            assert np.allclose(got, ref[i]), (max_memory, i)

    # Also a 3D case: a (32, 40, 24) float32 array with (4, 4, 4) chunks.
    # A chunk is 256 bytes, the level sizes are [122880, 15360, 1920, 240, 16],
    # and a step that covers k levels takes 256 * (1 + 8 + .. + 8**k) bytes:
    #   k=1 -> 2304, k=2 -> 18688

    shape = (32, 40, 24)
    chunk = (4, 4, 4)
    nlevels = 5

    src = rng.random(shape).astype(np.float32)
    ref = [src]
    for _ in range(nlevels - 1):
        ref.append(_downsample_reference(ref[-1], 3))

    configs = [
        (2304, [(0, 1, False), (1, 1, False), (2, 1, False), (3, 1, False)]),
        (5000, [(0, 1, False), (1, 1, True), (2, 2, True)]),
    ]

    for max_memory, expected_steps in configs:
        s = MemoryStore()
        steps = []
        write_ome_zarr_pyramid(
            s,
            "",
            "zyx",
            PyramidData(src),
            chunk_shape=chunk,
            max_memory=max_memory,
            steps=steps,
        )
        got_steps = [(d["level"], d["nlevels"], d["kept_in_memory"]) for d in steps]
        assert got_steps == expected_steps, (max_memory, got_steps)
        for i in range(nlevels):
            got = open_zarr(s, f"level{i}")[...].get_now()
            assert np.allclose(got, ref[i]), (max_memory, i)


def _cast_reference(data, dtype):
    """Reference for how write_ome_zarr_pyramid casts to another dtype."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(data), info.min, info.max).astype(dtype)
    return data.astype(dtype)


def test_write_multiscale_dtype():
    # Writing a pyramid with a different dtype than the source. Casting to an
    # integer dtype must round (instead of truncate) and clip to the dtype's
    # range (instead of wrapping around). Each level is the downsample of the
    # previous (stored) level.

    # A float32 source with values that need rounding and clipping
    rng = np.random.default_rng(42)
    src = (rng.random((16, 24)) * 300).astype(np.float32)
    src[0, 0] = 1000.0  # out of range for uint8

    chunk = (4, 4)
    nlevels = _expected_auto_nlevels((16, 24), chunk)

    ref = [_cast_reference(src, "uint8")]
    for _ in range(nlevels - 1):
        ref.append(_cast_reference(_downsample_reference(ref[-1], 2), "uint8"))

    for max_memory in [0, 1]:
        s = MemoryStore()
        write_ome_zarr_pyramid(
            s,
            "",
            "yx",
            PyramidData(src, dtype="uint8"),
            chunk_shape=chunk,
            max_memory=max_memory,
        )
        for i in range(nlevels):
            arr = open_zarr(s, f"level{i}")
            assert arr.dtype == "uint8", (max_memory, i)
            got = arr[...].get_now()
            assert got.dtype == np.uint8, (max_memory, i)
            assert np.array_equal(got, ref[i]), (max_memory, i)

    # An integer source with a float pyramid: means are exact, no rounding
    src = rng.integers(0, 200, (16, 24), dtype=np.uint8)
    ref = [src.astype(np.float32)]
    for _ in range(nlevels - 1):
        ref.append(_downsample_reference(ref[-1], 2).astype(np.float32))

    for max_memory in [0, 1]:
        s = MemoryStore()
        write_ome_zarr_pyramid(
            s,
            "",
            "yx",
            PyramidData(src, dtype="float32"),
            chunk_shape=chunk,
            max_memory=max_memory,
        )
        for i in range(nlevels):
            got = open_zarr(s, f"level{i}")[...].get_now()
            assert np.allclose(got, ref[i]), (max_memory, i)


def test_write_multiscale_int_chunk_shape():
    # An int chunk_shape means that chunk size for all spatial dimensions,
    # with the t/c dimensions covered by a single chunk
    for shape, axes in [((16, 16), "yx"), ((16, 16, 16), "zyx"), ((2, 16, 16), "cyx")]:
        zyx_ndim = sum(1 for a in axes if a in "xyz")
        expected_chunk = (*shape[:-zyx_ndim], *((4,) * zyx_ndim))
        zyx_shape = tuple(s for s, ax in zip(shape, axes, strict=True) if ax in "xyz")

        src = np.random.default_rng(42).random(shape).astype(np.float32)
        nlevels = _expected_auto_nlevels(zyx_shape, (4,) * zyx_ndim)
        ref = [src]
        for _ in range(nlevels - 1):
            ref.append(_downsample_reference(ref[-1], zyx_ndim))

        for max_memory in [0, 1]:
            s = MemoryStore()
            write_ome_zarr_pyramid(
                s, "", axes, PyramidData(src), chunk_shape=4, max_memory=max_memory
            )
            for i in range(nlevels):
                arr = open_zarr(s, f"level{i}")
                assert arr.chunk_shape == expected_chunk, (axes, i)
                assert np.allclose(arr[...].get_now(), ref[i]), (axes, i)


def test_write_multiscale_zarr_source():
    # A ZarrArray in its own store as the source.
    # the pyramid can be written to a nested path in the target store.
    zyx_shape = (40, 56)
    src = np.random.default_rng(42).random(zyx_shape).astype(np.float32)

    s1 = MemoryStore()
    src_array = ZarrArray.create(s1, "source", zyx_shape, "float32", chunk_shape=(8, 8))
    src_array[...].set_now(src)

    nlevels = _expected_auto_nlevels(zyx_shape, (8, 8))
    ref = [src]
    for _ in range(nlevels - 1):
        ref.append(_downsample_reference(ref[-1], 2))

    for max_memory in [0, 1]:
        s = MemoryStore()
        write_ome_zarr_pyramid(
            s,
            "my/pyramid",
            "yx",
            PyramidData(src_array),
            chunk_shape=8,
            max_memory=max_memory,
        )
        for i in range(nlevels):
            arr = open_zarr(s, f"my/pyramid/level{i}")
            assert arr.chunk_shape == (8, 8), (max_memory, i)
            assert np.allclose(arr[...].get_now(), ref[i]), (max_memory, i)


def test_write_multiscale_metadata():
    # The ome metadata of the pyramid: name, axes, and per-level
    # scale and translation coordinate transformations
    shape = (2, 3, 16, 32)
    src = np.random.default_rng(42).random(shape).astype(np.float32)

    s = MemoryStore()
    g = write_ome_zarr_pyramid(
        s,
        "pyr",
        "tcyx",
        PyramidData(src, name="myname"),
        space_unit="micrometer",
        base_scale=2.0,
        chunk_shape=(1, 1, 4, 4),
        nlevels=3,
        max_memory=0,
    )

    ms = g.attributes["ome"]["multiscales"][0]
    assert ms["name"] == "myname"
    assert [d["name"] for d in ms["axes"]] == ["t", "c", "y", "x"]
    assert [d["type"] for d in ms["axes"]] == ["time", "channel", "space", "space"]
    assert [d.get("unit", "") for d in ms["axes"]] == [
        "",
        "",
        "micrometer",
        "micrometer",
    ]

    datasets = ms["datasets"]
    assert [d["path"] for d in datasets] == ["level0", "level1", "level2"]
    for i, d in enumerate(datasets):
        scale, translation = d["coordinateTransformations"]
        assert scale["type"] == "scale"
        assert scale["scale"] == [1.0, 1.0, 2.0 * 2**i, 2.0 * 2**i], i
        assert translation["type"] == "translation"
        assert translation["translation"] == [
            0,
            0,
            2.0 * 2**i / 2 - 1.0,
            2.0 * 2**i / 2 - 1.0,
        ], i

    # The metadata is also there when the group is opened again
    assert open_zarr(s, "pyr").attributes["ome"]["multiscales"][0]["name"] == "myname"

    # base_scale can also be given per dimension
    s = MemoryStore()
    g = write_ome_zarr_pyramid(
        s,
        "",
        "tcyx",
        PyramidData(src),
        base_scale=(1.0, 0.5, 2.0, 4.0),
        chunk_shape=(1, 1, 4, 4),
        nlevels=2,
    )
    ms = g.attributes["ome"]["multiscales"][0]
    d0, d1 = ms["datasets"]
    assert d0["coordinateTransformations"][0]["scale"] == [1.0, 0.5, 2.0, 4.0]
    assert d1["coordinateTransformations"][0]["scale"] == [1.0, 0.5, 4.0, 8.0]


def test_write_multiscale_steps():
    # The steps taken by the algorithm can be obtained via the public API
    src = np.zeros((16, 16), np.float32)

    # A single level: one trivial step that creates no levels
    for max_memory, kept in [(0, True), (1, False)]:
        s = MemoryStore()
        steps = []
        write_ome_zarr_pyramid(
            s,
            "",
            "yx",
            PyramidData(src),
            chunk_shape=(4, 4),
            nlevels=1,
            max_memory=max_memory,
            steps=steps,
        )
        assert steps == [{"level": 0, "nlevels": 0, "kept_in_memory": kept}]


def test_write_multiscale_input_validation():
    src = np.zeros((8, 8), np.float32)

    with pytest.raises(TypeError):
        write_ome_zarr_pyramid(MemoryStore(), "", "not a pyramid", "yx")
    with pytest.raises(ValueError):
        # 1D is not enough
        write_ome_zarr_pyramid(
            MemoryStore(), "", "yx", PyramidData(np.zeros(8, np.float32))
        )
    with pytest.raises(ValueError):
        # axes length does not match the array
        write_ome_zarr_pyramid(MemoryStore(), "", "zyx", PyramidData(src))
    with pytest.raises(ValueError):
        # not a valid combination of axes
        write_ome_zarr_pyramid(MemoryStore(), "", "xyz", PyramidData(src))
    with pytest.raises(TypeError):
        # base_scale must be a float or a tuple of floats
        write_ome_zarr_pyramid(MemoryStore(), "", "yx", PyramidData(src), base_scale=2)


def test_write_multiscale_multiple_pyramids():
    # Multiple pyramids in one ome-zarr file, each with their own data,
    # stored at pyramid{i}/level{j}
    shape = (40, 56)
    chunk = (4, 4)
    nlevels = _expected_auto_nlevels(shape, chunk)

    rng = np.random.default_rng(42)
    srcs = [
        rng.random(shape).astype(np.float32),
        rng.random(shape).astype(np.float32) * 2,
        rng.random(shape).astype(np.float32) * 0.5,
    ]
    refs = []
    for src in srcs:
        ref = [src]
        for _ in range(nlevels - 1):
            ref.append(_downsample_reference(ref[-1], 2))
        refs.append(ref)

    for max_memory in [0, 1]:  # trigger simple and advanced alg, respectively
        s = MemoryStore()
        write_ome_zarr_pyramid(
            s,
            "",
            "yx",
            [PyramidData(src, name=f"pyr{i}") for i, src in enumerate(srcs)],
            chunk_shape=chunk,
            max_memory=max_memory,
        )

        # The single-pyramid paths do not exist
        with pytest.raises(IOError):
            open_zarr(s, f"level{nlevels - 1}")

        for i, ref in enumerate(refs):
            for j in range(nlevels):
                arr = open_zarr(s, f"pyramid{i}/level{j}")
                assert arr.shape == ref[j].shape, (max_memory, i, j)
                assert arr.chunk_shape == chunk, (max_memory, i, j)
                assert np.allclose(arr[...].get_now(), ref[j]), (max_memory, i, j)
            with pytest.raises(IOError):
                open_zarr(s, f"pyramid{i}/level{nlevels}")

        # The pyramids contain different data
        p0 = open_zarr(s, "pyramid0/level0")[...].get_now()
        p1 = open_zarr(s, "pyramid1/level0")[...].get_now()
        assert not np.allclose(p0, p1)


def test_write_multiscale_multiple_pyramids_metadata():
    # The ome metadata has one multiscales entry per pyramid
    shape = (2, 16, 20)  # cyx
    rng = np.random.default_rng(42)
    srcs = [rng.random(shape).astype(np.float32), rng.random(shape).astype(np.float32)]

    s = MemoryStore()
    g = write_ome_zarr_pyramid(
        s,
        "my/pyramids",
        "cyx",
        [PyramidData(srcs[0], name="blobs"), PyramidData(srcs[1], name="rods")],
        chunk_shape=(1, 4, 4),
        nlevels=2,
        space_unit="micrometer",
        base_scale=2.0,
    )

    ms_list = g.attributes["ome"]["multiscales"]
    assert [m["name"] for m in ms_list] == ["blobs", "rods"]
    assert ms_list[0]["axes"] == ms_list[1]["axes"]
    assert [ax["name"] for ax in ms_list[0]["axes"]] == ["c", "y", "x"]
    for i, ms in enumerate(ms_list):
        assert [d["path"] for d in ms["datasets"]] == [
            f"pyramid{i}/level0",
            f"pyramid{i}/level1",
        ]
        for j, d in enumerate(ms["datasets"]):
            scale = d["coordinateTransformations"][0]["scale"]
            assert scale == [1.0, 2.0 * 2**j, 2.0 * 2**j], (i, j)

    # The metadata is also there when the group is opened again
    reopened = open_zarr(s, "my/pyramids").attributes["ome"]["multiscales"]
    assert [m["name"] for m in reopened] == ["blobs", "rods"]

    # The arrays exist at the paths that the metadata points to
    assert open_zarr(s, "my/pyramids/pyramid0/level0").shape == (2, 16, 20)
    assert open_zarr(s, "my/pyramids/pyramid1/level1").shape == (2, 8, 10)


def test_write_multiscale_multiple_pyramids_dtype_codecs():
    # Each pyramid can have its own dtype and codecs, while the chunk shape
    # and the scale are shared
    rng = np.random.default_rng(42)
    src = (rng.random((16, 24)) * 300).astype(np.float32)
    src[0, 0] = 1000.0  # out of range for uint8

    chunk = (4, 4)
    nlevels = _expected_auto_nlevels((16, 24), chunk)

    gzip_codecs = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "gzip", "configuration": {"level": 6}},
    ]
    s = MemoryStore()
    write_ome_zarr_pyramid(
        s,
        "",
        "yx",
        [
            PyramidData(src, name="default"),
            PyramidData(src, name="uint8", dtype="uint8"),
            PyramidData(src, name="uint16", dtype="uint16"),
            PyramidData(src, name="gzip", dtype="uint8", codecs=gzip_codecs),
        ],
        chunk_shape=chunk,
    )

    # The uint8 and gzip pyramids have the same data, the others are cast
    refs = {}
    for dtype in ["uint8", "uint16", "float32"]:
        ref = [_cast_reference(src, dtype)]
        for _ in range(nlevels - 1):
            ref.append(_cast_reference(_downsample_reference(ref[-1], 2), dtype))
        refs[dtype] = ref

    dtypes = ["float32", "uint8", "uint16", "uint8"]
    for i, dtype in enumerate(dtypes):
        for j in range(nlevels):
            arr = open_zarr(s, f"pyramid{i}/level{j}")
            assert arr.dtype == dtype, (i, j)
            got = arr[...].get_now()
            if dtype == "float32":
                assert np.allclose(got, refs[dtype][j]), (i, j)
            else:
                assert np.array_equal(got, refs[dtype][j]), (i, j)
            assert arr.chunk_shape == chunk, (i, j)

    # Only the last pyramid has the gzip codecs
    for i in range(4):
        codecs = open_zarr(s, f"pyramid{i}/level0")._codecs
        assert (codecs == gzip_codecs) == (i == 3), i


def test_write_multiscale_pyramiddata():
    # The PyramidData validates and converts its arguments
    a = np.zeros((8, 8), np.float32)

    # A numpy array is converted to an in-memory ZarrArray, and the
    # dtype / fill_value / codecs are inherited from it
    p = PyramidData(a, name="myname")
    assert p.src_array.shape == (8, 8)
    assert p.dtype == "float32"
    assert p.name == "myname"
    assert p.fill_value == 0.0
    assert isinstance(p.codecs, list)

    with pytest.raises(TypeError):
        PyramidData("not an array")
    with pytest.raises(ValueError):
        # 1D is not enough
        PyramidData(np.zeros(8, np.float32))
    with pytest.raises(ValueError):
        # 6D is too much
        PyramidData(np.zeros((2,) * 6, np.float32))
    with pytest.raises(TypeError):
        PyramidData(a, name=42)


def test_write_multiscale_multiple_pyramids_validation():
    a = np.zeros((8, 8), np.float32)
    b = np.zeros((8, 8), np.float32)

    with pytest.raises(TypeError):
        # not a list
        write_ome_zarr_pyramid(MemoryStore(), "", "yx", "not a list")
    with pytest.raises(TypeError):
        # an empty list
        write_ome_zarr_pyramid(MemoryStore(), "", "yx", [])
    with pytest.raises(TypeError):
        # not PyramidData instances
        write_ome_zarr_pyramid(MemoryStore(), "", "yx", [a, b])
    with pytest.raises(ValueError):
        # different dimensionality
        write_ome_zarr_pyramid(
            MemoryStore(),
            "",
            "yx",
            [PyramidData(a), PyramidData(np.zeros((8, 8, 8), np.float32))],
        )
    with pytest.raises(ValueError):
        # different shapes
        write_ome_zarr_pyramid(
            MemoryStore(),
            "",
            "yx",
            [PyramidData(a), PyramidData(np.zeros((8, 9), np.float32))],
        )


def test_write_multiscale_multiple_pyramids_steps():
    # The steps are reported for each pyramid in the list
    a = np.zeros((16, 16), np.float32)
    b = np.ones((16, 16), np.float32)

    for max_memory, kept in [(0, True), (1, False)]:
        s = MemoryStore()
        steps = []
        write_ome_zarr_pyramid(
            s,
            "",
            "yx",
            [PyramidData(a, name="one"), PyramidData(b, name="two")],
            chunk_shape=(4, 4),
            nlevels=1,
            max_memory=max_memory,
            steps=steps,
        )
        assert steps == [
            {"level": 0, "nlevels": 0, "kept_in_memory": kept},
            {"level": 0, "nlevels": 0, "kept_in_memory": kept},
        ]


def test_write_multiscale_multiple_pyramids_group_nodes():
    # Each pyramid subdirectory is itself a group node, so that the node
    # tree can be traversed as groups. This used to be missing: only the
    # level arrays existed, without a group node for the subdirectory.
    shape = (16, 20)
    srcs = [
        np.zeros(shape, np.float32),
        np.ones(shape, np.float32),
    ]

    s = MemoryStore()
    write_ome_zarr_pyramid(
        s,
        "",
        "yx",
        [PyramidData(src, name=f"pyr{i}") for i, src in enumerate(srcs)],
        chunk_shape=(4, 4),
        nlevels=2,
    )

    # The subdirectories are ome-zarr groups, and the levels are reachable through them
    for i in range(2):
        g = open_zarr(s, f"pyramid{i}")
        assert isinstance(g, ZarrGroup), i
        assert "ome" in g.attributes, i
        assert open_zarr(s, f"pyramid{i}/level0").shape == shape, i
        assert open_zarr(s, f"pyramid{i}/level1").shape == (8, 10), i

    # A single pyramid has no pyramid subdirectories
    s = MemoryStore()
    write_ome_zarr_pyramid(
        s, "", "yx", PyramidData(srcs[0]), chunk_shape=(4, 4), nlevels=2
    )
    with pytest.raises(IOError):
        open_zarr(s, "pyramid0")

    # A custom prefix ends up in the group nodes, the array paths,
    # and the metadata
    s = MemoryStore()
    write_ome_zarr_pyramid(
        s,
        "",
        "yx",
        [PyramidData(src) for src in srcs],
        chunk_shape=(4, 4),
        nlevels=2,
        pyramid_path_prefix="multires",
    )
    g = open_zarr(s, "multires1")
    assert isinstance(g, ZarrGroup)
    assert open_zarr(s, "multires1/level1").shape == (8, 10)
    with pytest.raises(IOError):
        open_zarr(s, "pyramid0")
    ms = open_zarr(s, "").attributes["ome"]["multiscales"]
    assert ms[1]["datasets"][0]["path"] == "multires1/level0"


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
