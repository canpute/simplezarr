"""
The logic for Zarr codecs.
"""

# References:
#
# * https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#chunk-encoding
# * https://zarr-specs.readthedocs.io/en/latest/v3/codecs/index.html
#
# Some quotes from the spec:
#
# * In encoding, you start with an array, and end with bytes. There is exactly one array-> bytes codec step.
# * In decoding, you start with bytes and end with an array. There is exactly one bytes-> array codec step.
# * This specification defines a set of codecs (“core codecs”) which all Zarr implementations SHOULD implement.
#
# Notes from simplezarr devs:
#
# * We also implement codecs that are easy to implement because we use numcodecs.
# * Third party code can subclass ``BaseCodec`` and use ``register_codec()`` to implement custom codecs as extensions.
# * In this code, bytes are represented as a 1D memoryview, so that slices can be made without making copies. Arrays are represented with numpy arrays.
# * On error reporting:
#   * Asserts are made only to test the internal integrity of this module.
#   * CodecError is raised when the requested list of codecs is not valid, in terms of input and output types.
#   * Otherwise, the appropriate Python error is raised.

from __future__ import annotations

import sys

import numcodecs
import numpy as np


__all__ = ["create_ndarray_type", "decode_bytes", "encode_array"]

CODEC_CLASS_BY_NAME = {}


def register_codec(cls: type):
    """Register a codec class.

    Can be used as a class decorator. The class must inherit from BaseCodec.
    """
    assert isinstance(cls, type) and issubclass(cls, BaseCodec)
    assert cls.name
    CODEC_CLASS_BY_NAME[cls.name] = cls
    return cls


ndarray = np.ndarray


class CodecError(Exception):
    """An error that is raised when the codec configuration is invalid."""

    pass


class ArrayType(np.ndarray):
    """An array subtype that defines shape and dtype."""

    shape = ()  # type: ignore
    dtype = ""  # type: ignore

    @classmethod
    def match(cls, a):
        """Check whether the given array matches the shape and dtype."""
        return (
            isinstance(a, np.ndarray) and a.shape == cls.shape and a.dtype == cls.dtype
        )


def create_ndarray_type(shape: tuple[int, ...], dtype: str):
    """Create an ``np.ndarray`` subtype with a shape and dtype property.

    This class is needed by the codecs, so that the decoder can turn the bytes into the correct array.
    """
    assert isinstance(dtype, str)
    shape_str = "x".join(str(i) for i in shape)
    name = f"ndarray_{shape_str}_{dtype}"
    return type(
        name,
        (ArrayType,),
        {"shape": tuple(shape), "dtype": dtype},
    )


def is_byte_like(value):
    """Get whether the given value is byte-like in the contex of simplezarr.codecs: a 1D memoryview with format 'B'."""
    return isinstance(value, memoryview) and value.ndim == 1 and value.format == "B"


def encode_array(array: ndarray, codec_dicts: list[dict]) -> memoryview:
    """Encode the given array to bytes, using the codecs as described in the codec_dicts."""

    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"encode_array expects array, but got {array.__class__.__name__}"
        )

    # Get codecs, with their order validated
    array_type = create_ndarray_type(array.shape, array.dtype.name)
    codecs, decoded_representation_types = resolve_codecs_from_dicts(
        codec_dicts, array_type
    )

    # Encode
    value = array
    assert issubclass(decoded_representation_types[0], ArrayType)
    assert decoded_representation_types[0].match(value)
    for i in range(len(codecs)):
        value = codecs[i].encode(value)
        ref_type = decoded_representation_types[i + 1]
        if ref_type is memoryview:
            assert is_byte_like(value)
        else:
            assert ref_type.match(value)

    return value


def decode_bytes(
    encoded_bytes: memoryview, codec_dicts: list[dict], array_type: type
) -> ndarray:
    """Decode the given bytes, using the codecs as described in the codec_dicts."""

    if not is_byte_like(encoded_bytes):
        raise TypeError(
            f"decode_bytes expects bytes as a memoryview, but got {encoded_bytes.__class__.__name__}"
        )

    # Get codecs, with their order validated
    codecs, decoded_representation_types = resolve_codecs_from_dicts(
        codec_dicts, array_type
    )

    # When decoding, the steps are applied in reverse order!
    codecs.reverse()
    decoded_representation_types.reverse()

    # Decode
    value = encoded_bytes
    assert decoded_representation_types[0] is memoryview
    for i in range(len(codecs)):
        value = codecs[i].decode(value, decoded_representation_types[i + 1])
        ref_type = decoded_representation_types[i + 1]
        if ref_type is memoryview:
            assert is_byte_like(value)
        else:
            assert ref_type.match(value)

    return value


def resolve_codecs_from_dicts(
    codec_dicts: list[dict], array_type: type
) -> tuple[list[BaseCodec], list[type]]:
    """Get codec objects from the description in codec_dicts.

    The dicts in ``codec_dicts`` should follow the Zarr spec.

    Returns a tuple with 1) a list of codecs, and 2) a list of representation_types.
    The latter can be used as extra validation.
    """
    # Create codecs
    codecs = []
    for codec_dict in codec_dicts:
        name = codec_dict["name"]
        configuration = codec_dict["configuration"]
        try:
            cls = CODEC_CLASS_BY_NAME[name]
        except KeyError:
            raise TypeError(f"Unknown Zarr codec {name}") from None
        codecs.append(cls(**configuration))

    # Resolve types
    # See https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html#determination-of-encoded-representations
    decoded_representation_types = [array_type]
    for i in range(len(codecs)):
        t = codecs[i].compute_encoded_representation_type(
            decoded_representation_types[i]
        )
        decoded_representation_types.append(t)

    if decoded_representation_types[-1] is not memoryview:
        raise CodecError("Final codec does not return memoryview")

    return codecs, decoded_representation_types


class BaseCodec:
    """The base codec class.

    This defines the methods that a codec must have according to the Zarr spec.
    The handling of ``decoded_representation_type`` may look a bit awkward; it
    is there to validate the codecs, and that the order of codecs is valid.
    """

    name = ""  # Subclasses must set this

    _type = ""  # Subclasses myst set this to either "a->a", "a->b", "b->b"

    def __init__(self, **configuration):
        self._configuration = configuration

    def compute_encoded_representation_type(self, decoded_representation_type: type):
        """Get the type of the value produced by ``encode()``, given the input.

        The returned type is either ``memoryview`` or an ``np.ndarray`` subclass.
        It raises an error when the input type is invalid.

        Subclasses that do "a->" may need to overload this method. For other codecs this implementation does the trick.
        """
        assert isinstance(decoded_representation_type, type)
        if self._type == "a->a":
            if issubclass(decoded_representation_type, ndarray):
                # Assume we return an array with same shape and dtype. If this is not the case,
                # the subclass should overload this method.
                return decoded_representation_type
            else:
                raise CodecError(f"{self.__class__.__name__} only encodes arrays.")
        elif self._type == "a->b":
            if issubclass(decoded_representation_type, ndarray):
                return memoryview
            else:
                raise CodecError(f"{self.__class__.__name__} only encodes arrays.")
        elif self._type == "b->b":
            if issubclass(decoded_representation_type, memoryview):
                return memoryview
            else:
                raise CodecError(f"{self.__class__.__name__} encodes bytes/memoryview.")
        else:
            raise AssertionError(f"Invalid Codec._type: {self._type!r}")

    def encode(self, value: memoryview | ndarray) -> memoryview | ndarray:
        """Encode the given value (memoryview or array)."""
        raise NotImplementedError()

    def decode(
        self, value: memoryview | ndarray, decoded_representation_type: type
    ) -> memoryview | ndarray:
        """Decode the given value (memoryview or array)."""
        raise NotImplementedError()


class BaseChecksumCodec(BaseCodec):
    """Base codec for numcodecs checksums."""

    _type = "b->b"
    _numcodec_class = None

    def encode(self, value: memoryview) -> memoryview:
        return memoryview(self._numcodec_class().encode(value))

    def decode(
        self, value: memoryview, decoded_representation_type: type
    ) -> memoryview:
        assert issubclass(decoded_representation_type, memoryview)
        return memoryview(self._numcodec_class().decode(value))


class BaseCompressionCodec(BaseCodec):
    """Base codec for numcodecs compression."""

    _type = "b->b"
    _numcodec_class = None
    _options = ["level"]

    def encode(self, value: memoryview) -> memoryview:
        options = {}
        for key in self._options:
            if key in self._configuration:
                options[key] = self._configuration[key]
        return memoryview(self._numcodec_class(**options).encode(value))

    def decode(
        self, value: memoryview, decoded_representation_type: type
    ) -> memoryview:
        assert issubclass(decoded_representation_type, memoryview)
        return memoryview(self._numcodec_class().decode(value))


@register_codec
class TransposeCodec(BaseCodec):
    """Implements an ``array -> array`` codec that permutes the dimensions of the chunk array.

    See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/transpose/index.html
    """

    name = "transpose"
    _type = "a->a"

    def compute_encoded_representation_type(self, decoded_representation_type: type):
        assert isinstance(decoded_representation_type, type)
        if not issubclass(decoded_representation_type, ArrayType):
            raise CodecError(f"{self.__class__.__name__} only encodes arrays.")

        shape = decoded_representation_type.shape
        dtype = decoded_representation_type.dtype

        order = self._configuration.get("order", None)
        if order is None:
            shape = tuple(reversed(shape))
        else:
            assert len(order) == len(shape)
            shape = tuple(shape[i] for i in order)

        return create_ndarray_type(shape, dtype)

    def encode(self, value: memoryview) -> memoryview:
        order = self._configuration.get("order", None)
        return value.transpose(order)

    def decode(
        self, value: memoryview, decoded_representation_type: type
    ) -> memoryview:
        assert issubclass(decoded_representation_type, ndarray)
        order = self._configuration.get("order", None)
        if order is None:
            inverse_order = None
        else:
            inverse_order = tuple(int(i) for i in np.argsort(order))
        return value.transpose(inverse_order)


@register_codec
class BytesCodec(BaseCodec):
    """Implements an ``array -> bytes`` codec that encodes arrays of fixed-size numeric
    data types as a sequence of bytes in lexicographical order. For multi-byte
    data types, it encodes the array either in little endian or big endian.

    See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/bytes/index.html
    """

    name = "bytes"
    _type = "a->b"

    def encode(self, value: ndarray) -> memoryview:
        # First flatten, a copy is only made if needed. Keep dtype, or byteswap wont work correctly!
        flat = np.ravel(value, order="C")

        # Swap byteorder if necessary
        data_byteorder = self._configuration.get("endian", "")
        if data_byteorder in ("big", "little") and sys.byteorder != data_byteorder:
            flat = flat.byteswap()  # always copy bc flat may be a view

        # Turn into bytes and return as memoryview
        flat.dtype = np.uint8
        return memoryview(flat)

    def decode(self, value: memoryview, decoded_representation_type: type) -> ndarray:
        assert issubclass(decoded_representation_type, ArrayType)

        arr = np.frombuffer(value, decoded_representation_type.dtype)
        arr = arr.reshape(decoded_representation_type.shape)

        # Make the array match the endianness of the current machine. If the
        # endianness is not given or invalid, the code silently assume that it
        # matches the system, which is probably a good guess.
        data_byteorder = self._configuration.get("endian", "")
        if data_byteorder in ("big", "little") and sys.byteorder != data_byteorder:
            arr = arr.byteswap()

        return arr


@register_codec
class Crc32cCodec(BaseChecksumCodec):
    """Implements a ``bytes -> bytes`` codec that appends a CRC32C checksum of the input bytestream.

    See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/crc32c/index.html
    """

    name = "crc32c"
    _numcodec_class = numcodecs.CRC32C


@register_codec
class GzipCodec(BaseCompressionCodec):
    """Implements a ``bytes -> bytes`` codec that applies gzip compression.

    See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/gzip/index.html
    """

    name = "gzip"
    _numcodec_class = numcodecs.GZip


@register_codec
class BloscCodec(BaseCompressionCodec):
    """Implements a ``bytes -> bytes`` codec that uses the blosc container format.

    See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/blosc/index.html
    """

    name = "blosc"
    _type = "b->b"
    _numcodec_class = numcodecs.Blosc
    _options = ["cname", "clevel", "shuffle", "typesize", "blocksize"]

    def encode(self, value: memoryview) -> memoryview:
        options = {}
        for key in self._options:
            if key in self._configuration:
                options[key] = self._configuration[key]

        # Fix shuffle option
        shuffle_map = {"noshuffle": 0, "shuffle": 1, "bitshuffle": 2}
        shuffle_map.update({0: 0, 1: 1, 2: 2})
        shuffle = options.pop("shuffle", None)
        if shuffle is not None:
            options["shuffle"] = shuffle_map[shuffle]

        c = self._numcodec_class(**options)
        return memoryview(c.encode(value))

    def decode(
        self, value: memoryview, decoded_representation_type: type
    ) -> memoryview:
        assert issubclass(decoded_representation_type, memoryview)
        c = self._numcodec_class()
        return memoryview(c.decode(value))


@register_codec
class ShardingCodec(BaseCodec):
    """Implements a Zarr ``array -> bytes`` codec for sharding.

    Sharding logically splits chunks (“shards”) into sub-chunks (“inner chunks”)
    that can be individually compressed and accessed. This allows to colocate
    multiple chunks within one storage object, bundling them in shards.

    This codec *wraps* other codecs, which are applied for each shard.

    See https://zarr-specs.readthedocs.io/en/latest/v3/codecs/sharding-indexed/index.html
    """

    # AK: Cool, but let's skip for now
    # Probably not too hard to implement, let's peek at https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/codecs/sharding.py

    name = "sharding_indexed"
    _type = "a->b"

    def encode(self, value: memoryview) -> memoryview:
        raise NotImplementedError()

    def decode(
        self, value: memoryview, decoded_representation_type: type
    ) -> memoryview:
        raise NotImplementedError()


# %%%%% Extension codecs (not part of the Zarr spec, but common or easy to implement)


@register_codec
class Crc32Codec(BaseChecksumCodec):
    name = "crc32"  # without the final 'c'
    _numcodec_class = numcodecs.CRC32


@register_codec
class Adler32Codec(BaseChecksumCodec):
    name = "adler32"
    _numcodec_class = numcodecs.Adler32


@register_codec
class Fletcher32Codec(BaseChecksumCodec):
    name = "fletcher32"
    _numcodec_class = numcodecs.Fletcher32


@register_codec
class Jenkinslookup3Codec(BaseChecksumCodec):
    name = "jenkins_lookup3"
    _numcodec_class = numcodecs.JenkinsLookup3


@register_codec
class Lz4Codec(BaseCompressionCodec):
    name = "lz4"
    _numcodec_class = numcodecs.lz4
    _options = ["acceleration"]


@register_codec
class ZstdCodec(BaseCompressionCodec):
    name = "zstd"
    _numcodec_class = numcodecs.Zstd
    _options = ["level", "checksum"]


@register_codec
class ZlibCodec(BaseCompressionCodec):
    name = "zlib"
    _numcodec_class = numcodecs.Zlib


@register_codec
class Bz2Codec(BaseCompressionCodec):
    name = "bz2"
    _numcodec_class = numcodecs.BZ2


@register_codec
class LzmaCodec(BaseCompressionCodec):
    name = "lzma"
    _numcodec_class = numcodecs.LZMA
