# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The padding tail of a padded vocabulary must not be sampled.

Checkpoints round their embedding matrix up to a friendlier width -- Qwen3-8B
carries 151936 rows for 151665 real tokens. The extra rows are neither zero nor
-inf, so a sampler reaches them and returns ids the tokenizer cannot decode. A
training framework masks them on its own side, so a rollout engine that does not
disagrees with the trainer over exactly those positions, and nothing reports it.

`Config.true_vocab_size` is the number of rows that carry a real token. Left at
its default of 0 the mask does not apply, which is correct for every model whose
vocabulary is not padded.

Constructing an RLHFModelRunner needs a GPU, so these drive the override on a
bare instance and patch the base `postprocess` that `super()` resolves to.
"""

from types import SimpleNamespace

import pytest
import torch

try:
    from atom.model_engine.model_runner import ModelRunner
    from atom.rollout.model_runner_ext import RLHFModelRunner
except Exception as exc:  # noqa: BLE001 - any import-time probe failure means no GPU
    # ModelRunner pulls in aiter, which resolves the chip architecture at import
    # time and raises rather than ImportError when there is no device. That puts
    # this module out of reach of the non-GPU runner, so skip instead of erroring
    # the whole collection.
    pytest.skip(
        f"RLHFModelRunner needs a GPU at import time: {exc}", allow_module_level=True
    )


@pytest.fixture
def base_call(monkeypatch):
    """Capture what the override hands to the base implementation."""
    seen = {}

    def _postprocess(self, batch, logits, *args, **kwargs):
        seen["batch"] = batch
        seen["logits"] = logits
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "output"

    monkeypatch.setattr(ModelRunner, "postprocess", _postprocess, raising=False)
    return seen


def _runner(true_vocab_size):
    runner = object.__new__(RLHFModelRunner)
    runner.config = SimpleNamespace(true_vocab_size=true_vocab_size)
    return runner


def _postprocess(runner, logits):
    return runner.postprocess(
        "batch", logits, "temperatures", None, None, False, "hidden_states"
    )


def test_padding_positions_become_minus_inf(base_call):
    logits = torch.zeros(2, 10)
    logits[:, 8:] = 5.0

    _postprocess(_runner(8), logits)

    got = base_call["logits"]
    assert torch.isinf(got[:, 8:]).all() and (got[:, 8:] < 0).all()
    assert (got[:, :8] == 0).all()


def test_a_padded_row_is_never_the_argmax(base_call):
    """The failure this prevents: the tail wins and an undecodable id is
    emitted. The padding is deliberately the largest value present."""
    logits = torch.zeros(1, 6)
    logits[0, 5] = 100.0

    _postprocess(_runner(5), logits)

    assert base_call["logits"].argmax(dim=-1).item() != 5


def test_default_of_zero_masks_nothing(base_call):
    logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    _postprocess(_runner(0), logits)

    assert torch.equal(base_call["logits"], torch.arange(6.0).reshape(1, 6))


def test_an_unpadded_vocabulary_is_untouched(base_call):
    """true_vocab_size equal to the width is the common case for a model that
    does not pad, and must not trip the comparison."""
    logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    _postprocess(_runner(6), logits)

    assert torch.equal(base_call["logits"], torch.arange(6.0).reshape(1, 6))


def test_a_config_without_the_field_masks_nothing(base_call):
    """Configs are constructed in a lot of places; a missing field must not
    raise from inside the sampling path."""
    runner = object.__new__(RLHFModelRunner)
    runner.config = SimpleNamespace()
    logits = torch.arange(6, dtype=torch.float32).reshape(1, 6)

    _postprocess(runner, logits)

    assert torch.equal(base_call["logits"], torch.arange(6.0).reshape(1, 6))


def test_everything_else_is_forwarded_unchanged(base_call):
    runner = _runner(4)
    logits = torch.zeros(1, 6)

    result = runner.postprocess(
        "batch",
        logits,
        "temperatures",
        "top_ks",
        "top_ps",
        True,
        "hidden_states",
        needs_independent_noise=True,
    )

    assert result == "output"
    assert base_call["batch"] == "batch"
    assert base_call["args"] == (
        "temperatures",
        "top_ks",
        "top_ps",
        True,
        "hidden_states",
    )
    assert base_call["kwargs"] == {"needs_independent_noise": True}
