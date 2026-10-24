"""Core tests: no GPU, no backbone, no dataset.

    pytest gode/tests

They cover the two things that silently break when a backbone is ported or
when the core is edited in one repository only: the stream container and the
payload contract a backbone has to honour.
"""

import numpy as np
import torch

from gode import streams
from gode.compression import disjoint_chunks
from gode.hierarchy import geometric_progression, resolve_minimum
from gode.trainer import parse_iterations, sampling_probabilities


# ----------------------------------------------------------------------
# Hierarchy helpers
# ----------------------------------------------------------------------

def test_geometric_progression_spans_the_requested_range():
    steps = geometric_progression(100, 10_000, 5)
    assert steps[0] == 100 and steps[-1] == 10_000
    assert len(steps) == 5
    assert all(a < b for a, b in zip(steps, steps[1:]))


def test_resolve_minimum_accepts_counts_fractions_and_a_default():
    assert resolve_minimum(5_000, 1_000_000) == 5_000
    assert resolve_minimum(0.25, 1_000_000) == 250_000
    assert resolve_minimum(None, 1_000_000) == 100_000


def test_sampling_probabilities():
    uniform = sampling_probabilities(4, 1.0)
    assert torch.allclose(uniform, torch.full((4,), 0.25))

    fine = sampling_probabilities(3, 2.0)
    assert torch.isclose(fine.sum(), torch.tensor(1.0))
    assert fine[0] > fine[1] > fine[2]          # masks[0] is the finest level

    coarse = sampling_probabilities(3, -2.0)
    assert coarse[0] < coarse[1] < coarse[2]


def test_parse_iterations():
    assert parse_iterations("1000, 2000") == [1000, 2000]
    assert parse_iterations("") == []
    assert parse_iterations(None) == []


# ----------------------------------------------------------------------
# Levels
# ----------------------------------------------------------------------

def _nested_masks(sizes, total):
    """masks[0] is the finest; every level contains the next one."""
    return [torch.arange(total) < size for size in sizes]


def test_disjoint_chunks_partition_the_finest_level():
    masks = _nested_masks([100, 40, 10], total=100)
    chunks = disjoint_chunks(masks)

    assert len(chunks) == len(masks)
    assert chunks[0].sum() == 10                      # coarsest level comes first
    assert sum(int(c.sum()) for c in chunks) == 100   # together they rebuild the finest

    stacked = torch.stack(chunks).int().sum(0)
    assert stacked.max() <= 1                         # no primitive is stored twice

    # decoding chunks 0..k must reproduce level k, counted from the coarsest
    for level, mask in enumerate(reversed(masks)):
        decoded = torch.stack(chunks[:level + 1]).any(0)
        assert torch.equal(decoded, mask)


# ----------------------------------------------------------------------
# Stream container
# ----------------------------------------------------------------------

def _payload(n=256):
    rng = np.random.default_rng(0)
    return {
        "geom": {"quantized": True, "scale": 0.125, "count": n,
                 "xyz": rng.standard_normal((n, 3)).astype(np.float16),
                 "opacity": rng.integers(0, 255, size=(n, 1)).astype(np.uint8)},
        "features": {"quantized": True, "degree": 3,
                     "sh": rng.integers(-127, 127, size=(n, 9, 3)).astype(np.int8)},
    }


def test_streams_round_trip(tmp_path):
    payload = _payload()
    path = tmp_path / "lod_0.zst"
    size_mb = streams.write(str(path), payload)

    assert path.exists() and size_mb > 0
    restored = streams.read(str(path))

    assert set(restored) == set(payload)
    for group, content in payload.items():
        assert set(restored[group]) == set(content)
        for key, value in content.items():
            if isinstance(value, np.ndarray):
                assert np.array_equal(restored[group][key], value)
                assert restored[group][key].dtype == value.dtype
            else:
                assert restored[group][key] == value


def test_streams_round_trip_with_a_shared_dictionary(tmp_path):
    payload = _payload(4096)
    dicts = streams.train_dictionaries(payload, dict_size=16_384,
                                       save_path=str(tmp_path / "zstd_dict.pkl"))
    assert set(dicts) == {"geom", "features"}

    path = tmp_path / "lod_0.zst"
    streams.write(str(path), payload, dicts)
    restored = streams.read(str(path), streams.load_dictionaries(str(tmp_path / "zstd_dict.pkl")))
    assert np.array_equal(restored["features"]["sh"], payload["features"]["sh"])


def test_reading_a_foreign_file_fails_loudly(tmp_path):
    path = tmp_path / "not_a_stream.zst"
    path.write_bytes(b"this is not a GoDe bitstream")
    try:
        streams.read(str(path))
    except ValueError:
        return
    raise AssertionError("reading a foreign file should raise")


# ----------------------------------------------------------------------
# Backbone contract
# ----------------------------------------------------------------------

class StubBackbone:
    """The smallest model the core can drive: two primitive attributes."""

    def __init__(self, n=0):
        self._xyz = torch.randn(n, 3)
        self._feature = torch.randn(n, 4)

    @property
    def num_primitives(self):
        return self._xyz.shape[0]

    def primitive_names(self):
        return ["_xyz", "_feature"]

    def clone_empty(self, copy_qa=False):
        return StubBackbone()

    def subset(self, mask, copy_qa=True):
        clone = self.clone_empty()
        for name in self.primitive_names():
            setattr(clone, name, getattr(self, name)[mask].clone())
        return clone

    def append_primitives(self, other):
        for name in self.primitive_names():
            setattr(self, name, torch.cat([getattr(self, name), getattr(other, name)], dim=0))

    def encode_payload(self, quantize=True):
        return {"geom": {"quantized": quantize, "xyz": self._xyz.numpy().astype(np.float16)},
                "features": {"quantized": quantize,
                             "feature": self._feature.numpy().astype(np.float16)}}

    def decode_payload(self, payload):
        self._xyz = torch.from_numpy(payload["geom"]["xyz"]).float()
        self._feature = torch.from_numpy(payload["features"]["feature"]).float()
        return True


def test_a_backbone_payload_survives_encode_decode(tmp_path):
    model = StubBackbone(512)
    masks = _nested_masks([512, 200, 50], total=512)

    for level, chunk in enumerate(disjoint_chunks(masks)):
        part = model.subset(chunk)
        streams.write(str(tmp_path / f"lod_{level}.zst"), part.encode_payload())

    accumulated = None
    for level in range(len(masks)):
        decoded = model.clone_empty()
        decoded.decode_payload(streams.read(str(tmp_path / f"lod_{level}.zst")))
        if accumulated is None:
            accumulated = decoded
        else:
            accumulated.append_primitives(decoded)

    assert accumulated.num_primitives == model.num_primitives
    # the finest level holds every primitive, in coarse-to-fine order
    expected = torch.cat([model._xyz[c] for c in disjoint_chunks(masks)])
    assert torch.allclose(accumulated._xyz, expected.half().float())
