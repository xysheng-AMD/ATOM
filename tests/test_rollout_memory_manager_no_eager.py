# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sleep keeps everything resident outside eager mode -- and only there.

Decode CUDA graphs capture the addresses of the weights and of the KV pool.
Freeing either across sleep invalidates them, and recapturing on wake faults
under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`. So outside eager mode the
release path returns early and both stay where they are.

That early return reads `enforce_eager` off the host class, which is a
`ModelRunner` in production but not in every caller: `MemoryManagerMixin` is a
mixin, and the suite next door drives it with a `SimpleNamespace` that has no
such attribute. Reading it unguarded turns a missing attribute into an
`AttributeError` part-way through a release, so the guard defaults to the eager
value -- the releasing path, which is what every caller predating this change
got.

`test_rollout_memory_manager.py` covers what a release must drop. These tests
cover when it must not run at all, and which way the default falls.
"""

from types import SimpleNamespace

import torch
from torch import nn

from atom.rollout import memory_manager
from atom.rollout.memory_manager import MemoryManagerMixin


class _CacheOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_cache = torch.empty(1)
        self.v_cache = torch.empty(1)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _CacheOwner()
        self.linear = nn.Linear(2, 2)


def _runner(monkeypatch, **overrides):
    """A host carrying only what the release path reads off `self`."""
    runner = SimpleNamespace(
        label="test",
        model=_Model(),
        kv_cache=torch.empty(1),
        config=SimpleNamespace(num_kvcache_blocks=7),
        attn_metadata_builder=None,
        draft_kv_builder=None,
    )
    runner._get_models_with_kv = lambda: [runner.model]
    for key, value in overrides.items():
        setattr(runner, key, value)
    monkeypatch.setattr(memory_manager, "set_kv_cache_data", lambda _value: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return runner


def test_no_eager_keeps_the_kv_pool(monkeypatch):
    runner = _runner(monkeypatch, enforce_eager=False)

    MemoryManagerMixin._release_kv_cache(runner)

    assert runner.kv_cache is not None
    assert runner.model.attn.k_cache is not None


def test_no_eager_keeps_the_weights_where_the_graphs_expect_them(monkeypatch):
    runner = _runner(monkeypatch, enforce_eager=False)
    before = {n: p.data_ptr() for n, p in runner.model.named_parameters()}

    MemoryManagerMixin._release_weights(runner)

    after = {n: p.data_ptr() for n, p in runner.model.named_parameters()}
    assert before == after
    assert not getattr(runner, "_weights_discarded", False)


def test_no_eager_resume_does_not_reallocate(monkeypatch):
    """The mirror of the release guard: the pool was never freed, so there is
    nothing to allocate -- and allocating would hand the graphs a new base
    pointer, which is the whole thing this policy is avoiding."""
    runner = _runner(monkeypatch, enforce_eager=False)

    def _fail(_n):
        raise AssertionError("allocate_kv_cache called on a pool never released")

    runner.allocate_kv_cache = _fail
    pool = runner.kv_cache

    MemoryManagerMixin._resume_kv_cache(runner)

    assert runner.kv_cache is pool


def test_eager_still_releases(monkeypatch):
    runner = _runner(monkeypatch, enforce_eager=True)

    MemoryManagerMixin._release_kv_cache(runner)

    assert runner.kv_cache is None
    assert runner.model.attn.k_cache is None


def test_a_host_without_enforce_eager_still_releases(monkeypatch):
    """Defaulting the other way would turn sleep into a silent no-op for every
    host that does not define the attribute."""
    runner = _runner(monkeypatch)
    assert not hasattr(runner, "enforce_eager")

    MemoryManagerMixin._release_kv_cache(runner)

    assert runner.kv_cache is None
