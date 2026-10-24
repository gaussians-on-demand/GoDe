"""Zstd stream container shared by every GoDe backbone.

A file holds a list of named byte streams. Each stream is one pickled object
(a numpy array, or the group metadata) compressed with zstd. Streams are
grouped: geometry and appearance are compressed against separate trained
dictionaries because their byte statistics differ.

A backbone only has to produce, and consume, a payload of the form

    {"geom": {"xyz": ndarray, "opacity_scale": 0.01, ...},
     "features": {"sites": ndarray, ...}}

Scalars and other non-array values travel in the group metadata stream.
"""

import io
import os
import pickle
import struct

import numpy as np
import zstandard as zstd

MAGIC = b"GODEZST1"
_PICKLE = pickle.HIGHEST_PROTOCOL


def _compressor(level, zdict=None):
    try:
        params = zstd.ZstdCompressionParameters.from_level(
            level=level, enable_ldm=True, window_log=27,
            ldm_min_match=64, ldm_hash_log=22,
            ldm_bucket_size_log=3, ldm_hash_rate_log=4,
        )
        return zstd.ZstdCompressor(compression_params=params, dict_data=zdict,
                                   write_content_size=False, threads=-1)
    except Exception:
        return zstd.ZstdCompressor(level=level, dict_data=zdict,
                                   write_content_size=False, threads=-1)


def _as_dict(blob):
    return zstd.ZstdCompressionDict(blob) if isinstance(blob, (bytes, bytearray)) else blob


def _split(group):
    """Split a group payload into (metadata, {name: array})."""
    arrays = {k: v for k, v in group.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in group.items() if not isinstance(v, np.ndarray)}
    meta["arrays"] = list(arrays)
    return meta, arrays


def train_dictionaries(payload, dict_size=262_144, block_size=1024,
                       chunk=128, max_samples=80_000, seed=1234, save_path=None):
    """Train one zstd dictionary per group on row-slices of its arrays."""
    rng = np.random.default_rng(seed)
    dicts = {}
    for group, content in payload.items():
        _, arrays = _split(content)
        if not arrays:
            continue
        n = min(a.shape[0] for a in arrays.values())
        samples = []
        for block in rng.permutation(max(1, n // block_size)):
            start, stop = block * block_size, min((block + 1) * block_size, n)
            for offset in range(start, stop, chunk):
                end = min(offset + chunk, stop)
                samples += [pickle.dumps(a[offset:end], protocol=_PICKLE)
                            for a in arrays.values()]
            if len(samples) >= max_samples:
                break
        try:
            dicts[group] = zstd.train_dictionary(dict_size, samples, k=2048, d=8,
                                                 steps=8, split_point=0.75,
                                                 shrink_dict=True, threads=0)
        except TypeError:
            dicts[group] = zstd.train_dictionary(dict_size, samples)

    if save_path is not None:
        with open(save_path, "wb") as f:
            f.write(pickle.dumps({k: d.as_bytes() for k, d in dicts.items()},
                                 protocol=_PICKLE))
    return dicts


def load_dictionaries(path):
    if path is None:
        return {}
    with open(path, "rb") as f:
        blob = pickle.loads(f.read())
    return {k: _as_dict(v) for k, v in blob.items()}


def write(path, payload, dicts=None, level=22):
    """Write a payload; returns the file size in MB."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dicts = dicts or {}

    streams = []
    for group, content in payload.items():
        cctx = _compressor(level, _as_dict(dicts.get(group)))
        meta, arrays = _split(content)
        streams.append((f"{group}/meta", cctx.compress(pickle.dumps(meta, protocol=_PICKLE))))
        for name, array in arrays.items():
            raw = pickle.dumps(np.ascontiguousarray(array), protocol=_PICKLE)
            streams.append((f"{group}/{name}", cctx.compress(raw)))

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(streams)))
        for name, blob in streams:
            name_b = name.encode("utf8")
            f.write(struct.pack("<HQ", len(name_b), len(blob)))
            f.write(name_b)
            f.write(blob)

    size = len(MAGIC) + 4 + sum(10 + len(n.encode("utf8")) + len(b) for n, b in streams)
    return size / (1024 ** 2)


def read(path, dicts=None):
    """Inverse of `write`: returns {group: {name: value}}."""
    dicts = dicts or {}
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(MAGIC):
        raise ValueError(f"{path}: not a {MAGIC.decode()} stream file")

    offset = len(MAGIC)
    count = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    blobs = {}
    for _ in range(count):
        name_len, blob_len = struct.unpack_from("<HQ", data, offset)
        offset += 10
        name = data[offset:offset + name_len].decode("utf8")
        offset += name_len
        blobs[name] = data[offset:offset + blob_len]
        offset += blob_len

    decompressors = {}

    def _decompress(group, blob):
        if group not in decompressors:
            zdict = _as_dict(dicts.get(group))
            decompressors[group] = (zstd.ZstdDecompressor(dict_data=zdict) if zdict
                                    else zstd.ZstdDecompressor())
        with decompressors[group].stream_reader(io.BytesIO(blob)) as reader:
            chunks = iter(lambda: reader.read(131_072), b"")
            return pickle.loads(b"".join(chunks))

    payload = {}
    for name in blobs:
        group, _, key = name.partition("/")
        if key != "meta":
            continue
        content = _decompress(group, blobs[name])
        for array in content.pop("arrays", []):
            content[array] = _decompress(group, blobs[f"{group}/{array}"])
        payload[group] = content
    return payload
