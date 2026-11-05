# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A command that zeroes, frees or reallocates the KV pool, or commits new
weights, first preempts what is running and afterwards drops the prefix-cache
index. Refused, it runs nothing and leaves what was running as it was.
"""

import pickle
import queue
import threading
from collections import deque
from types import SimpleNamespace

import pytest
from aiter_stub import stubbed_aiter
from conftest import MockConfig

with stubbed_aiter():
    from atom.model_engine.engine_core import EngineCore
    from atom.model_engine.pp_engine_core import PPEngineCoreProc
from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.model_engine.collective_rpc import (
    DISCARD_WEIGHT_SYNC_CMD,
    RpcResponseRouter,
    RpcResult,
)
from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.engine_utility import (
    RECEIVE_WEIGHTS_RDMA_CMD,
    EngineUtilityHandler,
    kv_change,
)
from atom.model_engine.scheduler import (
    DecodeScheduler,
    PrefillScheduler,
    ScheduledBatchOutput,
    Scheduler,
)
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams


def _running(n, scheduler_cls=Scheduler):
    """A scheduler with *n* sequences past their prefill, decoding."""
    sched = scheduler_cls(MockConfig(num_kvcache_blocks=64))
    seqs = [
        Sequence([10 + i, 11, 12, 13, 14], 4, SamplingParams(max_tokens=40))
        for i in range(n)
    ]
    for seq in seqs:
        sched.add(seq)
    if n:
        batch, active = sched.schedule()
        sched.postprocess(
            list(active.values()),
            ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True),
            batch=batch,
        )
    assert list(sched.running) == seqs
    return sched, seqs


def _engine(scheduler, cls=EngineCore):
    engine = cls.__new__(cls)
    engine.scheduler = scheduler
    engine.kv_transfer_enabled = False
    engine._next_idle_kv_drain = 0.0
    engine._has_pending_utility = True
    engine._is_rl_weights_offloaded = False
    engine._weight_sync_fenced = False
    engine._rl_weights_inconsistent = False
    engine._generation_paused = False
    return engine


def _run(engine, cmd, args=None):
    """Drain one command through process_queue; return (reply, commands run).

    Only the runner side is stood in for: what reaches
    ``_execute_utility_command`` is recorded and answered.
    """
    out = queue.Queue()
    handler = EngineUtilityHandler(None, out, label="test", scheduler=engine.scheduler)
    ran = []

    def execute(cmd, args):
        ran.append(cmd)
        reply = {"cmd": cmd}
        handler.output_queue.put_nowait(("UTILITY_RESPONSE", reply))
        return reply

    handler._execute_utility_command = execute
    pending = queue.Queue()
    pending.put_nowait((cmd, {} if args is None else args))
    engine._has_pending_utility = True
    handler.process_queue(pending, engine)
    kind, reply = out.get_nowait()
    assert kind == "UTILITY_RESPONSE"
    return reply, ran


def _count_index_drops(monkeypatch, sched):
    drops = []
    monkeypatch.setattr(sched.block_manager, "clear_cache", lambda: drops.append(1))
    return drops


# ── what counts as a KV change ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "cmd, args, change",
    [
        ("update_weights", {"sync_id": None}, "weights"),
        ("update_weights_shm", {"sync_id": None, "is_last": True}, "weights"),
        ("update_weights_ipc", {"sync_id": None}, "weights"),
        ("update_weights_shm", {"sync_id": None, "is_last": False}, None),
        ("clear_kv_cache", {}, "kv"),
        ("release_memory", {"tags": ["weights", "kv_cache"]}, "kv"),
        ("release_memory", {}, "kv"),  # no tags: both, as the runner reads it
        ("resume_memory", {"tags": ["kv_cache"]}, "kv"),
        ("release_memory", {"tags": ["weights"]}, None),
        ("start_profile", {}, None),
    ],
)
def test_what_changes_the_kv_pool(cmd, args, change):
    assert kv_change(cmd, args) == change


# ── a KV change preempts first, then drops the index ───────────────────────


@pytest.mark.parametrize(
    "cmd, args",
    [("update_weights", {"sync_id": None}), ("clear_kv_cache", {})],
)
def test_a_kv_change_preempts_what_runs_and_drops_the_index(monkeypatch, cmd, args):
    sched, seqs = _running(3)
    drops = _count_index_drops(monkeypatch, sched)

    reply, ran = _run(_engine(sched), cmd, args)

    assert "error" not in reply and ran == [cmd]
    assert not sched.running
    assert list(sched.waiting)[:3] == seqs, "they resume in the order they ran"
    assert drops == [1]


def test_preempted_sequences_recompute_their_context_when_scheduled_again():
    sched, seqs = _running(2)
    _run(_engine(sched), "update_weights", {"sync_id": None})

    batch, active = sched.schedule()

    assert batch.total_seqs_num_prefill == 2
    assert sorted(active) == sorted(seq.id for seq in seqs)
    for i, seq in enumerate(seqs):
        assert list(seq.token_ids) == [10 + i, 11, 12, 13, 14]


def test_a_non_last_bucket_leaves_what_runs(monkeypatch):
    sched, seqs = _running(2)
    drops = _count_index_drops(monkeypatch, sched)

    _run(_engine(sched), "update_weights_shm", {"sync_id": None, "is_last": False})

    assert list(sched.running) == seqs
    assert drops == []


# ── refused, nothing runs and nothing moves ────────────────────────────────


def _assert_refused(monkeypatch, sched, seqs, engine, match, cmd="update_weights"):
    drops = _count_index_drops(monkeypatch, sched)

    reply, ran = _run(engine, cmd, {"sync_id": None, "timeout": 0.05})

    assert match in reply["error"] and reply["error"].startswith("refused")
    assert ran == [], "a refused change runs nothing"
    assert list(sched.running) == seqs, "and preempts none"
    assert drops == []


def test_refused_while_a_connector_holds_a_running_sequence(monkeypatch):
    sched, seqs = _running(3)
    monkeypatch.setattr(sched, "_is_preemptable", lambda seq: seq is not seqs[1])

    _assert_refused(monkeypatch, sched, seqs, _engine(sched), "a connector holds")


def test_a_refused_direct_update_leaves_the_engine_serving(monkeypatch):
    sched, _ = _running(2)
    monkeypatch.setattr(sched, "_is_preemptable", lambda seq: False)
    engine = _engine(sched)

    reply, ran = _run(engine, "update_weights", {"sync_id": None})

    assert reply["error"].startswith("refused") and ran == []
    assert not engine._weight_sync_fenced, "it wrote nothing, so nothing to fence"


def test_a_refused_last_bucket_fails_its_sync(monkeypatch):
    sched, seqs = _running(2)
    engine = _engine(sched)
    out = queue.Queue()
    handler = EngineUtilityHandler(None, out, label="test", scheduler=sched)
    handler._execute_utility_command = lambda cmd, args: {"cmd": cmd}
    pending = queue.Queue()
    pending.put_nowait(("update_weights_shm", {"sync_id": "s1", "is_last": False}))
    engine._has_pending_utility = True
    handler.process_queue(pending, engine)
    monkeypatch.setattr(sched, "_is_preemptable", lambda seq: False)

    pending.put_nowait(("update_weights_shm", {"sync_id": "s1", "is_last": True}))
    engine._has_pending_utility = True
    handler.process_queue(pending, engine)

    assert "refused" in out.get_nowait()[1]["error"]
    assert handler._weight_sync_failure, "its first bucket was written"
    assert engine._weight_sync_fenced and list(sched.running) == seqs


def test_refused_while_a_token_is_in_the_pipeline(monkeypatch):
    sched, seqs = _running(2)
    sched._pp_inflight_token_block.add(seqs[0].id)

    _assert_refused(monkeypatch, sched, seqs, _engine(sched), "in the pipeline")


def test_refused_while_remote_kv_is_parked(monkeypatch):
    sched, seqs = _running(2)
    sched._num_parked_remote_kv = 1

    _assert_refused(monkeypatch, sched, seqs, _engine(sched), "wait for remote KV")


def test_refused_while_kv_transfers_stay_in_flight(monkeypatch):
    sched, seqs = _running(2)
    engine = _engine(sched)
    polled = []
    engine.has_pending_kv_work = lambda: True
    engine._advance_idle_kv_transfer = lambda dispatch_new=True: polled.append(
        dispatch_new
    )

    _assert_refused(monkeypatch, sched, seqs, engine, "still in flight")
    assert polled and not any(polled), "reported back, nothing new started"


def test_waits_for_kv_transfers_to_report_back(monkeypatch):
    sched, _ = _running(2)
    engine = _engine(sched)
    pending = [True, True, False]
    engine.has_pending_kv_work = lambda: pending.pop(0)
    engine._advance_idle_kv_transfer = lambda dispatch_new=True: None

    _, ran = _run(engine, "clear_kv_cache", {"timeout": 5})

    assert ran == ["clear_kv_cache"] and not sched.running


def _pp_head(sched, completes):
    # A middle prefill chunk in stage 1 is in `_in_flight` though not in
    # `_pp_inflight_token_block`: it still writes KV into its blocks.
    engine = _engine(sched, PPEngineCoreProc)
    engine.is_head = True
    engine._pp_kv_aggregator = None
    engine._in_flight = deque(["middle chunk"])

    def collect(poll_ms):
        if completes and engine._in_flight:
            engine._in_flight.popleft()

    engine._collect_pp_completions = collect
    return engine


def test_the_pp_head_drains_its_pipeline_before_a_kv_change():
    sched, _ = _running(2)
    engine = _pp_head(sched, completes=True)

    _, ran = _run(engine, "clear_kv_cache", {"timeout": 5})

    assert ran == ["clear_kv_cache"] and not engine._in_flight
    assert not sched.running


def test_refused_on_the_pp_head_while_its_pipeline_does_not_empty(monkeypatch):
    sched, seqs = _running(2)
    engine = _pp_head(sched, completes=False)

    _assert_refused(monkeypatch, sched, seqs, engine, "still in the pipeline")


@pytest.mark.parametrize("scheduler_cls", [DecodeScheduler, PrefillScheduler])
def test_a_disaggregated_engine_refuses_with_requests_in_flight(
    monkeypatch, scheduler_cls
):
    # A decode engine's prompts were prefilled by the prefill engine: preempted,
    # a request would wait for a PrefillDone that is not coming again.
    sched = scheduler_cls(MockConfig(num_kvcache_blocks=64))
    sched.add(Sequence([1, 2, 3, 4], 4, SamplingParams(max_tokens=8)))

    reply, ran = _run(_engine(sched), "clear_kv_cache", {"timeout": 0.05})

    assert "cannot recompute" in reply["error"] and ran == []


@pytest.mark.parametrize("scheduler_cls", [DecodeScheduler, PrefillScheduler])
def test_a_disaggregated_engine_with_nothing_in_flight_changes(scheduler_cls):
    sched = scheduler_cls(MockConfig(num_kvcache_blocks=64))

    _, ran = _run(_engine(sched), "clear_kv_cache")

    assert ran == ["clear_kv_cache"]


# ── an offload tier must drop what it keeps ────────────────────────────────


def test_an_offload_tier_cannot_drop_its_store_until_it_says_so():
    assert not KVConnectorSchedulerBase.drop_cached_kv(SimpleNamespace(is_offload=True))
    assert KVConnectorSchedulerBase.drop_cached_kv(SimpleNamespace(is_offload=False))


def test_a_weight_change_is_refused_with_an_offload_tier_a_kv_change_is_not():
    sched = Scheduler(MockConfig(num_kvcache_blocks=64))
    sched.kv_connector = SimpleNamespace(drop_cached_kv=lambda: False)

    reply, ran = _run(_engine(sched), "update_weights", {"sync_id": None})
    assert "cannot drop KV computed with the old weights" in reply["error"]
    assert ran == []

    reply, ran = _run(_engine(sched), "clear_kv_cache")
    assert ran == ["clear_kv_cache"], "the weights stay, so its KV does too"


# ── through CoreManager: a refusal fences only what wrote ──────────────────


class _Runners:
    """The TP ranks of one engine, each applying whatever it is sent, and
    reporting *unavailable* when asked what it supports."""

    def __init__(self, unavailable=None):
        self.unavailable = unavailable or {}
        self.calls = []

    def utility_rpc(self, method, payload, timeout=300.0):
        if method == "get_worker_capabilities":
            value = {"unavailable": self.unavailable}
        else:
            self.calls.append((method, payload))
            value = 1
        return [RpcResult(payload.request_id, 0, value=value)]


def _wired(schedulers, runners=None):
    """A CoreManager whose DP engines are real utility handlers over
    *schedulers*, each control message delivered and answered in order."""
    mgr = object.__new__(CoreManager)
    mgr.label = "test-mgr"
    mgr.control_sockets = [object() for _ in schedulers]
    mgr._rpc_router = RpcResponseRouter()
    mgr._weight_sync_lock = threading.RLock()
    mgr._weight_sync_context = threading.local()
    mgr._weight_sync_dispatch_token = object()
    mgr._rpc_ranks_per_engine = 1
    engines = [_engine(sched) for sched in schedulers]
    outs = [queue.Queue() for _ in schedulers]
    runners = runners or [_Runners() for _ in schedulers]
    handlers = [
        EngineUtilityHandler(runner, out, label=f"dp{i}", scheduler=sched)
        for i, (runner, out, sched) in enumerate(zip(runners, outs, schedulers))
    ]

    def send(dp_rank, payload, copy=False):
        _, body = pickle.loads(payload)
        body = dict(body)
        pending = queue.Queue()
        pending.put_nowait((body.pop("cmd"), body))
        engines[dp_rank]._has_pending_utility = True
        handlers[dp_rank].process_queue(pending, engines[dp_rank])
        while not outs[dp_rank].empty():
            kind, reply = outs[dp_rank].get_nowait()
            if kind == "UTILITY_RESPONSE":
                mgr._route_utility_response(dp_rank, reply)

    mgr._send_control = send
    return mgr, engines


def _update(mgr):
    mgr.broadcast_utility_command_sync("update_weights", named_tensors=[], timeout=5)


def test_a_refused_update_fences_no_engine(monkeypatch):
    """CoreManager answers the refusal with DISCARD on every engine. None of
    them wrote, so all of them keep serving the weights they hold."""
    schedulers = [_running(2)[0], _running(2)[0]]
    for sched in schedulers:
        monkeypatch.setattr(sched, "_is_preemptable", lambda seq: False)
    mgr, engines = _wired(schedulers)

    with pytest.raises(RuntimeError, match="refused"):
        _update(mgr)

    assert not any(e._weight_sync_fenced or e._rl_weights_inconsistent for e in engines)


def test_a_peer_that_wrote_stays_fenced_until_the_next_sync(monkeypatch):
    refusing, writing = _running(2)[0], _running(2)[0]
    monkeypatch.setattr(refusing, "_is_preemptable", lambda seq: False)
    mgr, (refused, wrote) = _wired([refusing, writing])

    with pytest.raises(RuntimeError, match="refused"):
        _update(mgr)

    assert not refused._weight_sync_fenced
    assert wrote._weight_sync_fenced and wrote._rl_weights_inconsistent

    monkeypatch.setattr(refusing, "_is_preemptable", lambda seq: True)
    _update(mgr)

    assert not any(e._weight_sync_fenced for e in (refused, wrote))


def test_a_sync_the_runners_cannot_reload_is_refused_before_it_writes():
    """The engine asks its runners first, so a model they cannot rewrite in
    place is refused with nothing preempted or written, and no engine is
    fenced for it."""
    reason = "the model attends through MLA"
    schedulers = [_running(2)[0], _running(2)[0]]
    runners = [_Runners({"weight_reload": reason}) for _ in schedulers]
    mgr, engines = _wired(schedulers, runners)

    with pytest.raises(RuntimeError, match=reason):
        _update(mgr)

    assert not any(e._weight_sync_fenced or e._rl_weights_inconsistent for e in engines)
    assert all(len(sched.running) == 2 for sched in schedulers)
    assert [cmd for r in runners for cmd, _ in r.calls] == [DISCARD_WEIGHT_SYNC_CMD] * 2


def test_a_weight_stream_the_runners_cannot_take_is_drained_and_fences_nothing():
    """The trainer broadcasts regardless, so the runners still take the
    stream in, told to discard it; the engine keeps serving what it holds."""
    reason = "pipeline_parallel_size=2"
    runner = _Runners({"rdma_weight_receive": reason})
    mgr, (engine,) = _wired([_running(2)[0]], [runner])

    with pytest.raises(RuntimeError, match=reason):
        mgr.broadcast_utility_command_sync(
            RECEIVE_WEIGHTS_RDMA_CMD, timeout=5, group_name="g", version=3
        )

    assert not (engine._weight_sync_fenced or engine._rl_weights_inconsistent)
    assert len(engine.scheduler.running) == 2
    (drain, drained), (discard, _) = runner.calls
    assert (drain, drained.kwargs) == (RECEIVE_WEIGHTS_RDMA_CMD, {"refusal": reason})
    assert discard == DISCARD_WEIGHT_SYNC_CMD
