import numpy as np
import pytest

from simplezarr import MemoryStore, open_zarr
from simplezarr.utils.write_multiscale import write_ome_zarr_pyramid


def test_write_multiscale_even():

    for alg in "simple", "depthfirst":
        s = MemoryStore()

        a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]] * 8, np.float32)
        write_ome_zarr_pyramid(s, "", a, "yx", nlevels=3, _algorithm=alg)

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
    for alg in "simple", "depthfirst":
        s = MemoryStore()

        a = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] * 11, np.float32)
        write_ome_zarr_pyramid(s, "", a, "yx", nlevels=3, _algorithm=alg)

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

        for alg in "simple", "depthfirst":
            s = MemoryStore()
            write_ome_zarr_pyramid(
                s,
                "",
                src,
                axes,
                chunk_shape=chunk,
                _algorithm=alg,
            )

            with pytest.raises(IOError):
                open_zarr(s, f"level{expected_nlevels}")

            for i in range(expected_nlevels):
                arr = open_zarr(s, f"level{i}")
                assert arr.chunk_shape == chunk, (alg, shape, chunk, i)
                got = arr[...].get_now()
                assert got.shape == ref[i].shape, (alg, shape, chunk, i)
                assert np.allclose(got, ref[i]), (alg, shape, chunk, i)


def test_write_multiscale_nlevels():
    shape = (400, 640)
    chunk = (64, 64)

    rng = np.random.default_rng(42)
    a = rng.random(shape).astype(np.float32)

    # The number of levels when nlevels=0 (auto) is picked by the library
    expected_auto_nlevels = _expected_auto_nlevels(shape, chunk)

    for alg in "simple", "depthfirst":
        for nlevels in [0, 1, 2, 3, 4, 5, 8]:
            s = MemoryStore()
            write_ome_zarr_pyramid(
                s,
                "",
                a,
                "yx",
                nlevels=nlevels,
                chunk_shape=chunk,
                _algorithm=alg,
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
                assert n == expected_auto_nlevels, (alg, n, expected_auto_nlevels)
            else:
                assert n == nlevels, (alg, nlevels, n)

            # Check that the shapes halve from level to level
            expected_shape = shape
            for i in range(n):
                assert open_zarr(s, f"level{i}").shape == expected_shape, (
                    alg,
                    nlevels,
                    i,
                )
                expected_shape = tuple(s // 2 for s in expected_shape)


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
