# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""`add_request` has to fan out, because `preprocess` refuses to.

`IOProcessor.preprocess` returns exactly one `Sequence` and raises on
`SamplingParams.n > 1`, telling the caller to use `preprocess_fanout` instead --
its own docstring says so. `LLMEngine.add_request` called `preprocess` anyway,
so every offline `n > 1` request died in the engine before a single token was
generated, and the n == 1 path that CI exercises could not see it.

These tests pin both halves: n > 1 reaches the core manager as n sequences, and
n == 1 still submits exactly one.
"""

import unittest
from types import SimpleNamespace

from atom.model_engine.llm_engine import LLMEngine
from atom.sampling_params import SamplingParams


class _FakeIOProcessor:
    """Stands in for the real one, keeping the contract that matters here:
    `preprocess` is single-sequence only and rejects n > 1."""

    def __init__(self):
        self.fanout_calls = []

    def preprocess(self, prompt_or_tokens, sampling_params, **kwargs):
        if getattr(sampling_params, "n", 1) > 1:
            raise ValueError(
                "preprocess() returns a single Sequence; for SamplingParams.n > 1 "
                "call preprocess_fanout() and manage the returned list."
            )
        return SimpleNamespace(prompt=prompt_or_tokens, sibling=0)

    def preprocess_fanout(self, prompt_or_tokens, sampling_params, **kwargs):
        self.fanout_calls.append((prompt_or_tokens, sampling_params, kwargs))
        n = getattr(sampling_params, "n", 1)
        return [SimpleNamespace(prompt=prompt_or_tokens, sibling=i) for i in range(n)]


class _FakeCoreManager:
    def __init__(self):
        self.submitted = None

    def add_request(self, reqs):
        self.submitted = reqs


def _engine():
    """An LLMEngine with only the two collaborators add_request touches.

    `__init__` loads a tokenizer and starts engine processes, neither of which
    this path needs.
    """
    engine = object.__new__(LLMEngine)
    engine.io_processor = _FakeIOProcessor()
    engine.core_mgr = _FakeCoreManager()
    return engine


class TestAddRequestFanout(unittest.TestCase):
    def test_n_greater_than_one_submits_every_sibling(self):
        engine = _engine()

        engine.add_request(["a prompt"], SamplingParams(n=3))

        self.assertEqual(len(engine.core_mgr.submitted), 3)
        self.assertEqual([s.sibling for s in engine.core_mgr.submitted], [0, 1, 2])

    def test_n_equals_one_still_submits_exactly_one(self):
        engine = _engine()

        engine.add_request(["a prompt"], SamplingParams(n=1))

        self.assertEqual(len(engine.core_mgr.submitted), 1)

    def test_siblings_of_several_prompts_are_all_submitted(self):
        engine = _engine()

        engine.add_request(["first", "second"], SamplingParams(n=2))

        self.assertEqual(len(engine.core_mgr.submitted), 4)
        self.assertEqual(
            [s.prompt for s in engine.core_mgr.submitted],
            ["first", "first", "second", "second"],
        )

    def test_request_id_is_passed_as_the_parent_id(self):
        """The fan-out names the siblings after the caller's id, so it takes the
        id as `parent_request_id`; passing it as `request_id` is a TypeError."""
        engine = _engine()

        engine.add_request(["a prompt"], SamplingParams(n=2), request_ids=["req-7"])

        _, _, kwargs = engine.io_processor.fanout_calls[0]
        self.assertEqual(kwargs["parent_request_id"], "req-7")


if __name__ == "__main__":
    unittest.main()
