# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rank-local verdicts every rank of a CPU (gloo) process group takes alike.

Every rank must call these at the same point; ``group`` None is a single rank.
"""

from collections.abc import Sequence

import torch
import torch.distributed as dist


def group_max_each(values: Sequence[int], group) -> tuple[int, ...]:
    """For each position of ``values``, the largest any rank of ``group``
    passed there, in one collective."""
    if group is None:
        return tuple(values)
    flags = torch.tensor(list(values), dtype=torch.int64, device="cpu")
    dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
    return tuple(int(v) for v in flags.tolist())


def group_min_each(values: Sequence[int], group) -> tuple[int, ...]:
    """For each position of ``values``, the smallest any rank of ``group``
    passed there, in one collective."""
    return tuple(-v for v in group_max_each([-v for v in values], group))


def group_max(value: int, group) -> int:
    """The largest ``value`` any rank of ``group`` passed."""
    return group_max_each((value,), group)[0]


def group_min(value: int, group) -> int:
    """The smallest ``value`` any rank of ``group`` passed."""
    return -group_max(-value, group)


def group_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true."""
    return not group_max(int(not ok), group)
