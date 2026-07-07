from collections import OrderedDict

import torch
from torch import nn

from romav2.romav2 import RoMaV2


class CountingDescriptor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.batch_sizes = []

    def forward(self, img):
        self.calls += 1
        self.batch_sizes.append(img.shape[0])
        value = img.flatten(1).mean(dim=1)
        return [
            value[:, None, None, None].expand(-1, 2, 2, 1).clone(),
            (value + 1)[:, None, None, None].expand(-1, 2, 2, 1).clone(),
        ]


def make_model(cache_size=8):
    model = RoMaV2.__new__(RoMaV2)
    nn.Module.__init__(model)
    model.f = CountingDescriptor()
    model._descriptor_cache = OrderedDict()
    model._descriptor_cache_size = cache_size
    return model


def test_descriptor_cache_reuses_overlapping_image_keys():
    model = make_model()
    images = torch.arange(4 * 3 * 4 * 4, dtype=torch.float32).reshape(4, 3, 4, 4)

    out1 = model._descriptor_features(images[:3], ["a", "b", "c"])
    assert model.f.calls == 1
    assert model.f.batch_sizes == [3]

    out2 = model._descriptor_features(images[1:4], ["b", "c", "d"])
    assert model.f.calls == 2
    assert model.f.batch_sizes == [3, 1]

    assert torch.equal(out1[0][1], out2[0][0])
    assert torch.equal(out1[0][2], out2[0][1])


def test_descriptor_cache_deduplicates_repeated_keys_inside_batch():
    model = make_model()
    images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    batch = torch.stack([images[0], images[0], images[1]], dim=0)

    out = model._descriptor_features(batch, ["a", "a", "b"])

    assert model.f.calls == 1
    assert model.f.batch_sizes == [2]
    assert torch.equal(out[0][0], out[0][1])


def test_descriptor_cache_returns_fresh_feature_lists():
    model = make_model()
    image = torch.ones(1, 3, 4, 4)

    out1 = model._descriptor_features(image, ["a"])
    out1[-1] = out1[-1] + 100
    out2 = model._descriptor_features(image, ["a"])

    assert torch.all(out2[-1] < 100)


def test_prefill_descriptor_cache_batches_future_block():
    model = make_model()
    images = torch.arange(4 * 3 * 4 * 4, dtype=torch.float32).reshape(4, 3, 4, 4)

    model.prefill_descriptor_cache(images[:3], ["a", "b", "c"])

    assert model.f.calls == 1
    assert model.f.batch_sizes == [3]

    out = model._descriptor_features(images[1:4], ["b", "c", "d"])
    assert model.f.calls == 2
    assert model.f.batch_sizes == [3, 1]
    assert torch.equal(out[0][0], model._descriptor_cache[("b", 4, 4, "cpu", "torch.float32")][0])


def test_adjacent_descriptor_features_extracts_sequence_once():
    model = make_model()
    sequence = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)

    f_a, f_b = model._adjacent_descriptor_features(sequence)

    assert model.f.calls == 1
    assert model.f.batch_sizes == [5]
    direct = model.f(sequence)
    assert torch.equal(f_a[0], direct[0][:-1])
    assert torch.equal(f_b[0], direct[0][1:])
