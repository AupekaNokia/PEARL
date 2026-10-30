"""Lean forward hooks on leaf modules (monotonic start/end per firing) and canonical layer indexing."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn


def leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Named submodules without child modules, in :meth:`~torch.nn.Module.named_modules` order."""
    return [(name, mod) for name, mod in model.named_modules() if name and not list(mod.children())]


class LeafHookProfiler:
    """
    Record ``time.monotonic_ns()`` start/end of every leaf-module call.

    A pre-hook on the model itself increments ``step_id`` at each model call, so within one
    ``generate()`` the prefill has ``step_id = 0`` and decode step ``k`` has ``step_id = k``.
    Call :meth:`mark_forward_end` after each ``generate()`` to close one forward block.
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._handles: list[Any] = []
        self._names: list[str] = []
        self._types: list[str] = []
        self._t0: list[int] = []
        self._t1: list[int] = []
        self._idx: list[int] = []
        self._step: list[int] = []
        self._open: list[int] = []
        self._blocks: list[tuple[int, int]] = []
        self._block_start = 0
        self._step_id = -1

    def _advance_step(self, _mod: nn.Module, _inp: Any) -> None:
        self._step_id += 1

    def _pre_hook(self, idx: int) -> Any:
        def hook(_mod: nn.Module, _inp: Any) -> None:
            self._open.append(len(self._t0))
            self._idx.append(idx)
            self._step.append(max(self._step_id, 0))
            self._t1.append(0)
            self._t0.append(time.monotonic_ns())

        return hook

    def _post_hook(self, _mod: nn.Module, _inp: Any, _out: Any) -> None:
        if self._open:
            self._t1[self._open.pop()] = time.monotonic_ns()

    def attach(self) -> None:
        """Register the step pre-hook on the model and pre/post hooks on every leaf."""
        self.detach()
        targets = leaf_modules(self._model)
        self._names = [name for name, _ in targets]
        self._types = [type(mod).__name__ for _, mod in targets]
        self._handles.append(self._model.register_forward_pre_hook(self._advance_step))
        for idx, (_, mod) in enumerate(targets):
            self._handles.append(mod.register_forward_pre_hook(self._pre_hook(idx)))
            self._handles.append(mod.register_forward_hook(self._post_hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def reset(self) -> None:
        for buf in (self._t0, self._t1, self._idx, self._step, self._open, self._blocks):
            buf.clear()
        self._block_start = 0
        self._step_id = -1

    def mark_forward_end(self) -> None:
        end = len(self._t0)
        if end > self._block_start:
            self._blocks.append((self._block_start, end))
        self._block_start = end
        self._step_id = -1

    def collect(self) -> list[dict[str, Any]]:
        """Per-forward blocks ``{"forward_idx", "layers": [{layer_name, layer_type, step_id, start_ns, end_ns}]}``."""
        if self._block_start < len(self._t0):
            self.mark_forward_end()
        return [
            {
                "forward_idx": fwd,
                "layers": [
                    {
                        "layer_name": self._names[self._idx[i]],
                        "layer_type": self._types[self._idx[i]],
                        "step_id": self._step[i],
                        "start_ns": self._t0[i],
                        "end_ns": self._t1[i],
                    }
                    for i in range(start, end)
                ],
            }
            for fwd, (start, end) in enumerate(self._blocks)
        ]


def leaf_firing_order(model: nn.Module, example_input: Any) -> list[tuple[str, str]]:
    """``(layer_name, layer_type)`` of each leaf firing during one plain ``model(example_input)``."""
    records: list[tuple[str, str]] = []
    handles: list[Any] = []

    def register(module: nn.Module, prefix: str) -> None:
        if not list(module.children()):
            handles.append(module.register_forward_hook(
                lambda mod, _i, _o, name=prefix: records.append((name, type(mod).__name__))
            ))
            return
        for child_name, child in module.named_children():
            register(child, f"{prefix}.{child_name}" if prefix else child_name)

    register(model, "")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example_input)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return records


def build_layer_registry(model: nn.Module, example_input: Any) -> list[tuple[int, str, str]]:
    """
    ``(layer_idx, layer_name, layer_type)`` with 1-based ``layer_idx`` in plain-forward firing order.

    Raises
    ------
    ValueError
        If a leaf fires more than once in one plain forward (ambiguous name -> index).
    """
    rows: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for idx, (name, layer_type) in enumerate(leaf_firing_order(model, example_input), start=1):
        if name in seen:
            raise ValueError(f"leaf module {name!r} fired more than once in a single plain forward")
        seen.add(name)
        rows.append((idx, name, layer_type))
    return rows
