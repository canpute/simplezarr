# ruff: noqa: N806

from simplezarr import codecs as codecsmod

import pytest
import numpy as np


# %%%%% Helpers


def test_create_ndarray_type():
    create_ndarray_type = codecsmod.create_ndarray_type

    cls1 = create_ndarray_type((2, 3), "uint16")
    cls2 = create_ndarray_type((3, 2), "uint16")

    assert isinstance(cls1, type) and issubclass(cls1, np.ndarray)
    assert isinstance(cls2, type) and issubclass(cls2, np.ndarray)
    assert repr(cls1) == "<class 'simplezarr.codecs.ndarray_2x3_uint16'>"
    assert repr(cls2) == "<class 'simplezarr.codecs.ndarray_3x2_uint16'>"


def test_is_byte_like():
    is_byte_like = codecsmod.is_byte_like

    array_1d_u1 = np.array([1, 2, 3, 4], "u1")
    array_2d_u1 = np.array([1, 2, 3, 4], "u1").reshape(2, 2)
    array_1d_u2 = np.array([1, 2, 3, 4], "u2")

    assert is_byte_like(memoryview(array_1d_u1))
    assert not is_byte_like(array_1d_u1)
    assert not is_byte_like(memoryview(array_2d_u1))
    assert not is_byte_like(memoryview(array_1d_u2))


# %%%%% Resolving the codecs


def test_resolve_codecs_from_dicts():
    resolve_codecs_from_dicts = codecsmod.resolve_codecs_from_dicts
    array_type = codecsmod.create_ndarray_type((10, 10), "uint16")

    # Simplest case
    d = [
        {"name": "bytes", "configuration": {"endian": "little"}},
    ]

    codecs, _types = resolve_codecs_from_dicts(d, array_type)
    assert len(codecs) == 1
    assert isinstance(codecs[0], codecsmod.BytesCodec)

    # Simple case
    d = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "gzip", "configuration": {"level": 6}},
    ]

    codecs, _types = resolve_codecs_from_dicts(d, array_type)
    assert len(codecs) == 2
    assert isinstance(codecs[0], codecsmod.BytesCodec)
    assert isinstance(codecs[1], codecsmod.GzipCodec)

    # More advanced
    d = [
        {"name": "transpose", "configuration": {"order": None}},
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "gzip", "configuration": {"level": 6}},
        {"name": "crc32c", "configuration": {}},
    ]

    codecs, _types = resolve_codecs_from_dicts(d, array_type)
    assert len(codecs) == 4
    assert isinstance(codecs[0], codecsmod.TransposeCodec)
    assert isinstance(codecs[1], codecsmod.BytesCodec)
    assert isinstance(codecs[2], codecsmod.GzipCodec)
    assert isinstance(codecs[3], codecsmod.Crc32cCodec)


def test_resolve_codecs_from_dicts_order_errors():
    resolve_codecs_from_dicts = codecsmod.resolve_codecs_from_dicts
    array_type = codecsmod.create_ndarray_type((10, 10), "uint16")

    # Fail: no codecs
    d = []
    with pytest.raises(codecsmod.CodecError):
        resolve_codecs_from_dicts(d, array_type)

    # Fail: does not end with bytes
    d = [
        {"name": "transpose", "configuration": {"order": None}},
    ]
    with pytest.raises(codecsmod.CodecError):
        resolve_codecs_from_dicts(d, array_type)

    # Fail: does not begin with array
    d = [
        {"name": "gzip", "configuration": {"level": 6}},
    ]
    with pytest.raises(codecsmod.CodecError):
        resolve_codecs_from_dicts(d, array_type)

    # Fail: codecs don't fit v1
    d = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "transpose", "configuration": {"order": None}},
        {"name": "gzip", "configuration": {"level": 6}},
    ]
    with pytest.raises(codecsmod.CodecError):
        resolve_codecs_from_dicts(d, array_type)

    # Fail: codecs don't fit v2
    d = [
        {"name": "gzip", "configuration": {"level": 6}},
        {"name": "bytes", "configuration": {"endian": "little"}},
    ]
    with pytest.raises(codecsmod.CodecError):
        resolve_codecs_from_dicts(d, array_type)

    # Fail: codecs don't fit v3
    d = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "bytes", "configuration": {"endian": "little"}},
    ]
    with pytest.raises(codecsmod.CodecError):
        resolve_codecs_from_dicts(d, array_type)


def test_resolve_codecs_from_dicts_other_errors():
    resolve_codecs_from_dicts = codecsmod.resolve_codecs_from_dicts
    array_type = codecsmod.create_ndarray_type((10, 10), "uint16")

    # Fail: codecs does not exist
    d = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "doesnotexist", "configuration": {}},
        {"name": "gzip", "configuration": {"level": 6}},
    ]
    with pytest.raises(RuntimeError):
        resolve_codecs_from_dicts(d, array_type)


# %%%%% Round trip


def test_encode_decode_round_trip():
    d = [
        {"name": "transpose", "configuration": {"order": None}},
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "gzip", "configuration": {"level": 6}},
        {"name": "crc32c", "configuration": {}},
        {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 6}},
    ]

    image = np.random.uniform(0, 128, (640, 480)).astype(np.uint8) * 2
    array_type = codecsmod.create_ndarray_type(image.shape, image.dtype.name)

    # First some type checking
    with pytest.raises(TypeError):
        codecsmod.encode_array(b"1234", d)
    with pytest.raises(TypeError):
        codecsmod.encode_array(memoryview(b"1234"), d)
    with pytest.raises(TypeError):
        codecsmod.decode_bytes(b"1234", d, array_type)
    with pytest.raises(TypeError):
        codecsmod.decode_bytes(image, d, array_type)

    # Encode

    bb = codecsmod.encode_array(image, d)
    assert memoryview(bb)

    # Decode

    image2 = codecsmod.decode_bytes(bb, d, array_type)
    assert isinstance(image2, np.ndarray)
    assert image2.shape == image.shape
    assert image2.dtype == image.dtype
    assert np.all(image == image2)


def test_custom_codec():
    @codecsmod.register_codec
    class MyCodec1(codecsmod.BaseCodec):
        name = "test_mycodec1"
        _type = "a->a"

        def encode(self, value):
            return value + 1

        def decode(self, value, decoded_representation_type):
            return value - 1

    d = [
        {"name": "test_mycodec1", "configuration": {}},
        {"name": "bytes", "configuration": {"endian": "little"}},
    ]

    image = np.random.uniform(0, 128, (640, 480)).astype(np.uint8) * 2
    array_type = codecsmod.create_ndarray_type(image.shape, image.dtype.name)

    bb = codecsmod.encode_array(image, d)
    image2 = codecsmod.decode_bytes(bb, d, array_type)
    assert isinstance(image2, np.ndarray)
    assert np.all(image == image2)

    # Check resolve error
    d = [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "test_mycodec1", "configuration": {}},
    ]

    with pytest.raises(codecsmod.CodecError):
        codecsmod.resolve_codecs_from_dicts(d, array_type)


# %%%%% Individual codecs


def test_transpose_codec():
    TransposeCodec = codecsmod.TransposeCodec

    arr0 = np.random.uniform(0, 128, (12, 14, 16)).astype(np.uint8) * 2
    array_type = codecsmod.create_ndarray_type(arr0.shape, arr0.dtype.name)

    # Type resolving
    type = TransposeCodec().compute_encoded_representation_type(array_type)
    assert type.shape == (16, 14, 12)
    type = TransposeCodec(order=(0, 2, 1)).compute_encoded_representation_type(
        array_type
    )
    assert type.shape == (12, 16, 14)

    # Encode / decode

    c = TransposeCodec()
    arr1 = c.encode(arr0)
    arr2 = c.decode(arr1, np.ndarray)

    assert arr0.shape == arr2.shape
    assert arr1.shape == (16, 14, 12)
    assert np.all(arr0 == arr2)

    c = TransposeCodec(order=(0, 2, 1))
    arr1 = c.encode(arr0)
    arr2 = c.decode(arr1, np.ndarray)

    assert arr0.shape == arr2.shape
    assert arr1.shape == (12, 16, 14)

    c = TransposeCodec(order=(0, 2))
    with pytest.raises(ValueError):
        c.encode(arr0)
    with pytest.raises(ValueError):
        c.decode(arr0, np.ndarray)


def test_bytes_codec():
    BytesCodec = codecsmod.BytesCodec

    arr0 = np.array([0, 1, 2, 3, 1000, 1001, 1002, 1003]).astype(np.uint16)
    array_type = codecsmod.create_ndarray_type(arr0.shape, "uint16")

    c = BytesCodec()
    mem1 = c.encode(arr0)
    arr2 = c.decode(mem1, array_type)
    assert np.all(arr0 == arr2)

    # Test endian param

    c_little = BytesCodec(endian="little")
    mem1_little = c_little.encode(arr0)
    arr2_little = c_little.decode(mem1_little, array_type)
    assert np.all(arr0 == arr2_little)

    c_big = BytesCodec(endian="big")
    mem1_big = c_big.encode(arr0)
    arr2_big = c_big.decode(mem1_big, array_type)
    assert np.all(arr0 == arr2_big)

    assert mem1_little != mem1_big


def test_crc32c_codec():
    Crc32cCodec = codecsmod.Crc32cCodec

    mem0 = memoryview(b"1234")

    c = Crc32cCodec()
    mem1 = c.encode(mem0)
    mem2 = c.decode(mem1, memoryview)
    assert mem0 == mem2
    assert len(mem1) == len(mem0) + 4

    # Fail because checksum has changed
    mem3 = memoryview(bytearray(mem1))
    mem3[-1] += 1
    with pytest.raises(RuntimeError):
        c.decode(mem3, memoryview)

    # Fail because data has changed
    mem3 = memoryview(bytearray(mem1))
    mem3[0] += 1
    with pytest.raises(RuntimeError):
        c.decode(mem3, memoryview)


def test_gzip_codec():
    GzipCodec = codecsmod.GzipCodec

    arr0 = np.random.uniform(0, 32, (10000)).astype(np.uint8)
    mem0 = memoryview(arr0)

    # Round trips
    for level in (0, 4, 8):
        c = GzipCodec(level=level)
        mem1 = c.encode(mem0)
        mem2 = c.decode(mem1, memoryview)
        assert mem0 == mem2

    # Test compression levels
    mem1 = GzipCodec(level=0).encode(mem0)
    mem2 = GzipCodec(level=1).encode(mem0)
    mem3 = GzipCodec(level=9).encode(mem0)

    assert len(mem0) == 10000
    assert len(mem1) > len(mem0)
    assert len(mem2) < len(mem0)
    assert len(mem3) < len(mem2)


def test_blosc_codec():
    BloscCodec = codecsmod.BloscCodec

    arr0 = np.random.uniform(0, 32, (10000)).astype(np.uint8)
    mem0 = memoryview(arr0)

    # Round trips
    for cname in ("zstd", "lz4"):
        for clevel in (0, 4, 8):
            for shuffle in (None, "noshuffle", "shuffle", "bitshuffle"):
                c = BloscCodec(cname=cname, clevel=clevel, shuffle=shuffle)
                mem1 = c.encode(mem0)
                mem2 = c.decode(mem1, memoryview)
                assert mem0 == mem2

    # Test compression levels
    mem1 = BloscCodec(cname="zstd", clevel=0).encode(mem0)
    mem2 = BloscCodec(cname="zstd", clevel=3).encode(mem0)
    mem3 = BloscCodec(cname="zstd", clevel=9).encode(mem0)

    assert len(mem0) == 10000
    assert len(mem1) > len(mem0)
    assert len(mem2) < len(mem0)
    assert len(mem3) < len(mem0)  # not better than level 3 (or 1) for this data


def test_sharding_codec():
    ShardingCodec = codecsmod.ShardingCodec

    # TODO: implement this once the ShardingCodec is implemented

    with pytest.raises(NotImplementedError):
        c = ShardingCodec()
        c.encode(memoryview(b"1234"))


if __name__ == "__main__":
    for func in list(globals().values()):
        if callable(func) and func.__name__.startswith("test_"):
            print(f"{func.__name__} ... ", end="")
            func()
            print("done")
    print("all done")
