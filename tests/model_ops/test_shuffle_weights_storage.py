# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`shuffle_weights` must reshuffle a parameter without moving it.

A captured CUDA graph holds the address of every tensor it reads. Online weight
updates reshuffle the weights in place and do not recapture, so the shuffle has
to write through the parameter's existing storage. Rebinding `.data` to the
tensor aiter hands back keeps the values right and silently leaves the graph
reading the old address -- which is the bug this pins, and it cannot be caught
by comparing values.

The 3D branch already wrote through `copy_`; these tests cover both branches so
they cannot drift apart again.

`shuffle_weight` itself is aiter's and needs a GPU to import, so it is replaced
here by a permutation with the same contract. What is under test is the storage
handling around it, not the shuffle.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest
import torch


def _import_utils():
    """Import `atom.model_ops.utils` against a fake aiter when there is no real
    one, and leave `sys.modules` as it was found.

    Mirrors `tests/aiter_stub.stubbed_aiter`, with the submodules this module
    imports from by name -- a plain module object does not answer
    `from aiter.ops.shuffle import ...`, only attribute access.
    """
    installed = []
    for name in (
        "aiter",
        "aiter.ops",
        "aiter.ops.shuffle",
        "aiter.ops.triton",
        "aiter.ops.triton.quant",
        "aiter.utility",
        "aiter.utility.fp4_utils",
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        module.__getattr__ = lambda _attr: MagicMock()
        sys.modules[name] = module
        installed.append(name)
    try:
        from atom.model_ops import utils

        return utils
    finally:
        for name in installed:
            sys.modules.pop(name, None)


utils = _import_utils()


def _reverse_rows(weight, layout=(16, 16)):
    """A stand-in for aiter's `shuffle_weight`: same shape, same dtype, a fresh
    contiguous tensor, and a permutation that is visible in the values."""
    return weight.flip(0).contiguous()


def test_2d_shuffle_writes_through_the_same_storage(monkeypatch):
    monkeypatch.setattr(utils, "shuffle_weight", _reverse_rows)
    param = torch.nn.Parameter(torch.randn(32, 8), requires_grad=False)
    expected = param.data.flip(0).clone()
    before = param.data_ptr()

    utils.shuffle_weights(param)

    assert param.data_ptr() == before
    assert torch.equal(param.data, expected)
    assert param.is_shuffled


def test_3d_shuffle_writes_through_the_same_storage(monkeypatch):
    monkeypatch.setattr(utils, "shuffle_weight", _reverse_rows)
    param = torch.nn.Parameter(torch.randn(4, 32, 8), requires_grad=False)
    expected = param.data.flip(1).clone()
    before = param.data_ptr()

    utils.shuffle_weights(param)

    assert param.data_ptr() == before
    assert torch.equal(param.data, expected)


def test_a_shuffle_that_changes_shape_falls_back_to_rebinding(monkeypatch):
    """aiter's shuffle returns the same shape and dtype today, so this branch is
    unreachable in practice. It exists so that a layout which does change one of
    them produces correct weights rather than a `copy_` shape error."""
    monkeypatch.setattr(
        utils, "shuffle_weight", lambda w, layout=(16, 16): w.reshape(8, 32)
    )
    param = torch.nn.Parameter(torch.randn(32, 8), requires_grad=False)

    utils.shuffle_weights(param)

    assert tuple(param.data.shape) == (8, 32)


def test_a_rank_it_cannot_shuffle_is_refused(monkeypatch):
    monkeypatch.setattr(utils, "shuffle_weight", _reverse_rows)
    param = torch.nn.Parameter(torch.randn(8), requires_grad=False)

    with pytest.raises(ValueError, match="dim to be 2 or 3"):
        utils.shuffle_weights(param)


def test_a_plain_tensor_is_refused(monkeypatch):
    monkeypatch.setattr(utils, "shuffle_weight", _reverse_rows)

    with pytest.raises(TypeError):
        utils.shuffle_weights(torch.randn(32, 8))
