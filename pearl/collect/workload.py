"""Hugging Face causal-LM generation workload."""

from __future__ import annotations

import gc
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
PROMPT = "The quick brown fox "


@dataclass(frozen=True)
class WorkloadSpec:
    """One grid point."""

    model_name: str
    batch_size: int
    sequence_length: int
    max_new_tokens: int
    precision: str
    model_revision: str | None = None
    use_cache: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _needs_remote_code(model_name: str, revision: str | None) -> bool:
    """True only when Transformers has no native class for the checkpoint's ``model_type``."""
    from transformers import PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    cfg_dict, _ = PretrainedConfig.get_config_dict(model_name, revision=revision)
    return cfg_dict.get("model_type") not in CONFIG_MAPPING


class GenerationWorkload:
    """
    Greedy generation of ``max_new_tokens`` tokens for ``batch_size`` copies of one padded prompt.

    The prompt is ``"The quick brown fox "`` repeated and padded/truncated to ``sequence_length``.
    Each :meth:`run_forward` is one ``generate()`` call followed by a CUDA synchronize.
    """

    def __init__(self, spec: WorkloadSpec, device: torch.device, seed: int) -> None:
        if spec.precision not in DTYPES:
            raise ValueError(f"precision must be one of {sorted(DTYPES)}, got {spec.precision!r}")
        self.spec = spec
        self.device = device
        self.seed = seed
        self.model: torch.nn.Module | None = None
        self._tokenizer: Any = None
        self._prompt_ids: torch.Tensor | None = None
        self._attention_mask: torch.Tensor | None = None

    def load(self) -> None:
        set_seed(self.seed)
        s = self.spec
        kwargs: dict[str, Any] = {"trust_remote_code": _needs_remote_code(s.model_name, s.model_revision)}
        if s.model_revision:
            kwargs["revision"] = s.model_revision
        self._tokenizer = AutoTokenizer.from_pretrained(s.model_name, **kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(s.model_name, torch_dtype=DTYPES[s.precision], **kwargs)
        self.model.to(self.device)
        self.model.eval()
        enc = self._tokenizer(
            PROMPT * max(1, s.sequence_length // 4),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=s.sequence_length,
        )
        self._prompt_ids = enc["input_ids"].to(self.device)
        self._attention_mask = enc["attention_mask"].to(self.device)

    def _generate(self, batch_size: int, max_new_tokens: int) -> torch.Tensor:
        assert self.model is not None and self._prompt_ids is not None and self._attention_mask is not None
        with torch.inference_mode():
            out = self.model.generate(
                self._prompt_ids.repeat(batch_size, 1),
                attention_mask=self._attention_mask.repeat(batch_size, 1),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
                disable_compile=True,
                use_cache=self.spec.use_cache,
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return out

    def warmup(self, steps: int) -> None:
        """``steps`` batch-1 generations of ``min(8, max_new_tokens)`` tokens (outside the measured window)."""
        for _ in range(max(0, steps)):
            self._generate(1, min(8, self.spec.max_new_tokens))

    def run_forward(self) -> dict[str, Any]:
        """One measured ``generate()``; returns token counts and latency."""
        t0 = time.perf_counter()
        out = self._generate(self.spec.batch_size, self.spec.max_new_tokens)
        latency = time.perf_counter() - t0
        n_new = int(out.shape[0] * (out.shape[1] - self.spec.sequence_length))
        return {"num_inferences": self.spec.batch_size, "num_tokens": n_new, "latency_s": latency}

    def example_input(self) -> torch.Tensor:
        """Prompt ids for one plain forward (used to index leaf layers)."""
        assert self._prompt_ids is not None
        return self._prompt_ids

    def unload(self) -> None:
        self.model = self._tokenizer = self._prompt_ids = self._attention_mask = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
