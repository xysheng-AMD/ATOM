# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A weight sync has to route routed-expert weights, and re-layout them after.

Experts arrive one tensor per expert (`...experts.3.gate_proj.weight`) and live
in the fused `w13_weight` / `w2_weight` of the layer's FusedMoE. That is neither
the incoming name nor anything `packed_modules_mapping` describes, and the
updater resolved names against only those two. Every routed expert weight
therefore matched nothing and was counted as skipped, at debug level -- the
rollout kept serving the experts it loaded from the checkpoint and no error
said so. The model loader already consults `get_expert_mapping()` for exactly
this; the updater now does too.

Routing alone would trade a silent no-op for silent garbage. FusedMoE's
`process_weights_after_loading` ends in a shuffle that permutes those buffers
into the layout its kernel reads, and a sync writes plain row-major bytes over
it. So the post-load hook is re-run once per sync, after the last bucket.

`update_weights` is driven directly here with fakes: a real FusedMoE needs a
GPU, and what is under test is the routing and the re-layout, not the kernel.
"""

from types import SimpleNamespace

import torch
from torch import nn

from atom.rollout.weight_updater import WeightUpdaterMixin


class _FusedMoE(nn.Module):
    """Records how the updater called it, the way FusedMoE would be called."""

    def __init__(self, num_experts=2):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.zeros(num_experts, 4, 2), requires_grad=False
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(num_experts, 2, 2), requires_grad=False
        )
        self.loaded = []
        self.post_load_calls = 0
        moe = self

        class _QuantMethod:
            def process_weights_after_loading(self, layer):
                assert layer is moe
                moe.post_load_calls += 1

        self.quant_method = _QuantMethod()

    def weight_loader(
        self, param, loaded_weight, weight_name="", shard_id="", expert_id=0
    ):
        self.loaded.append((weight_name, shard_id, expert_id))
        if shard_id == "w2":
            param.data[expert_id].copy_(loaded_weight)
        else:
            half = param.shape[1] // 2
            start = 0 if shard_id == "w1" else half
            param.data[expert_id, start : start + half].copy_(loaded_weight)


class _Model(nn.Module):
    def __init__(self, num_experts=2):
        super().__init__()
        self.experts = _FusedMoE(num_experts)
        self.num_experts = num_experts

    def get_expert_mapping(self):
        return [
            (
                (
                    "experts.w13_"
                    if weight_name in ("gate_proj", "up_proj")
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(self.num_experts)
            for shard_id, weight_name in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            )
        ]


class _Runner(WeightUpdaterMixin):
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cpu")
        self.rank = 0
        self.world_size = 1
        self.label = "test"
        self.kv_cache_cleared = 0

    def clear_kv_cache(self):
        self.kv_cache_cleared += 1


def _expert_tensors(num_experts=2):
    named = []
    for expert_id in range(num_experts):
        named.append((f"experts.{expert_id}.gate_proj.weight", torch.ones(2, 2)))
        named.append((f"experts.{expert_id}.up_proj.weight", torch.full((2, 2), 2.0)))
        named.append((f"experts.{expert_id}.down_proj.weight", torch.full((2, 2), 3.0)))
    return named


def test_every_routed_expert_weight_is_applied():
    runner = _Runner(_Model())

    updated = runner.update_weights(_expert_tensors(), clear_kv_cache=False)

    assert updated == 6
    assert len(runner.model.experts.loaded) == 6
    assert {shard for _, shard, _ in runner.model.experts.loaded} == {"w1", "w2", "w3"}
    assert {eid for _, _, eid in runner.model.experts.loaded} == {0, 1}


def test_the_values_reach_the_fused_buffers():
    model = _Model()
    runner = _Runner(model)

    runner.update_weights(_expert_tensors(), clear_kv_cache=False)

    # w13 is gate then up, so the halves carry the two source values.
    assert torch.equal(model.experts.w13_weight[0, :2], torch.ones(2, 2))
    assert torch.equal(model.experts.w13_weight[0, 2:], torch.full((2, 2), 2.0))
    assert torch.equal(model.experts.w2_weight[0], torch.full((2, 2), 3.0))


def test_the_post_load_layout_hook_runs_once_per_sync():
    model = _Model()
    runner = _Runner(model)

    runner.update_weights(_expert_tensors(), clear_kv_cache=False)

    assert model.experts.post_load_calls == 1


def test_a_model_without_experts_is_untouched():
    """The lookup must cost nothing, and change nothing, on a dense model."""

    class _Dense(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2)

    runner = _Runner(_Dense())

    updated = runner.update_weights(
        [("linear.weight", torch.ones(2, 2))], clear_kv_cache=False
    )

    assert updated == 1
    assert runner._get_expert_params_mapping() == {}


def test_an_unknown_name_is_still_reported_as_skipped():
    runner = _Runner(_Model())

    updated = runner.update_weights(
        [("experts.9.gate_proj.weight", torch.ones(2, 2))], clear_kv_cache=False
    )

    assert updated == 0
    assert runner.model.experts.loaded == []


def test_ipc_style_buckets_defer_the_hook_to_the_last_one():
    """The layout is re-established after the last bucket, not per bucket: a
    permutation applied twice is not the layout the kernel reads."""
    model = _Model()
    runner = _Runner(model)
    runner._ipc_buffer = SimpleNamespace()

    # update_weights is the single-shot path; the bucketed paths gate the same
    # call on is_last, which is asserted here through the shared helper.
    runner._moe_modules_to_post_load.add(model.experts)
    runner._rerun_moe_post_load()
    runner._rerun_moe_post_load()

    assert model.experts.post_load_calls == 1
