import numpy as np
import pytest

from simplezarr import MemoryStore, ZarrArray, open_zarr
from simplezarr.utils.write_multiscale import (
    _get_levels_per_step,
    _step_nbytes,
    _write_large_pyramid,
    write_ome_zarr_pyramid,
)


def test_write_multiscale_even():
    for max_memory in [0, 1]:  # trigger simple and advanced alg, respectively
        s = MemoryStore()

        a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, np.float32)
        write_ome_zarr_pyramid(s, "", a, "yx", nlevels=3, max_memory=max_memory)

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
        write_ome_zarr_pyramid(s, "", a, "yx", nlevels=3, max_memory=max_memory)

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
                src,
                axes,
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
                a,
                "yx",
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


def _create_depthfirst_pyramid(src, chunk, nlevels, zyx_ndim):
    """Create a source ZarrArray and the (empty) level arrays, like write_ome_zarr_pyramid does."""
    s = MemoryStore()
    src_array = ZarrArray.create(s, "", src.shape, src.dtype, chunk_shape=chunk)
    src_array[...].set_now(src)

    store = MemoryStore()
    new_arrays = []
    shape = src.shape
    for level in range(nlevels):
        if level > 0:
            shape = (*shape[:-zyx_ndim], *(s // 2 for s in shape[-zyx_ndim:]))
        new_arrays.append(
            ZarrArray.create(
                store, f"level{level}", shape, src.dtype, chunk_shape=chunk
            )
        )
    return src_array, new_arrays, store


def test_write_multiscale_depthfirst_max_memory():
    # Different memory constraints lead to different bottom-up strategies:
    # fewer or more levels per step, and either keeping the intermediate top
    # in memory (finishing with the simple approach), or loading the chunks
    # from disk again in the next step.
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
        # Via the public API
        s = MemoryStore()
        write_ome_zarr_pyramid(
            s,
            "",
            src,
            "yx",
            chunk_shape=chunk,
            max_memory=max_memory,
        )
        for i in range(nlevels):
            got = open_zarr(s, f"level{i}")[...].get_now()
            assert np.allclose(got, ref[i]), (max_memory, i)

        # Directly, so that we can also check the steps that were taken
        src_array, new_arrays, _ = _create_depthfirst_pyramid(src, chunk, nlevels, 2)
        steps = []
        _write_large_pyramid(
            src_array, new_arrays, 2, "local_mean", chunk, max_memory, steps
        )
        got_steps = [(d["level"], d["nlevels"], d["kept_in_memory"]) for d in steps]
        assert got_steps == expected_steps, (max_memory, got_steps)
        for i in range(nlevels):
            got = new_arrays[i][...].get_now()
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
        src_array, new_arrays, _ = _create_depthfirst_pyramid(src, chunk, nlevels, 3)
        steps = []
        _write_large_pyramid(
            src_array, new_arrays, 3, "local_mean", chunk, max_memory, steps
        )
        got_steps = [(d["level"], d["nlevels"], d["kept_in_memory"]) for d in steps]
        assert got_steps == expected_steps, (max_memory, got_steps)
        for i in range(nlevels):
            got = new_arrays[i][...].get_now()
            assert np.allclose(got, ref[i]), (max_memory, i)


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
