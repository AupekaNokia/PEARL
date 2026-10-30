"""Per-leaf structural constants of a Hugging Face causal LM from one plain forward on ``meta``/CPU."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

TRACE_SEQ_LEN = 8

_FUSED_QKV = frozenset({"c_attn", "query_key_value", "qkv_proj", "in_proj"})
_SUFFIX_ROLES: tuple[tuple[str, str], ...] = (
    ("q_proj", "q_proj"), ("k_proj", "k_proj"), ("v_proj", "v_proj"),
    ("o_proj", "o_proj"), ("out_proj", "o_proj"),
    ("gate_proj", "gate_proj"), ("up_proj", "up_proj"), ("down_proj", "down_proj"),
)
_EXACT_ROLES = {
    "c_fc": "up_proj", "dense_h_to_4h": "up_proj", "dense_4h_to_h": "down_proj",
    "fc_in": "up_proj", "fc1": "up_proj", "fc_out": "down_proj", "fc2": "down_proj",
}
_ACTIVATION_TYPES = frozenset({
    "GELU", "SiLU", "SiLUActivation", "ReLU", "GELUActivation",
    "NewGELUActivation", "Tanh", "Sigmoid", "Mish",
})
# Model-level leaves invoked functionally (no hook firing in the trace) but measured as layers.
_SUPPLEMENTAL_ROLES = frozenset({"rotary"})


class StructuralExtractionError(RuntimeError):
    """A model config could not be loaded, built or traced."""


@dataclass(frozen=True)
class Leaf:
    """Structural constants of one leaf module."""

    layer_name: str
    role: str
    params: int
    activations: int | None
    in_features: int | None
    out_features: int | None


@dataclass
class ModelStructure:
    """Leaves in plain-forward firing order plus the attention dims of the HF config."""

    model_id: str
    trace_tokens: int
    n_heads: int | None
    n_kv_heads: int | None
    d_head: int | None
    leaves: list[Leaf] = field(default_factory=list)

    @property
    def total_params(self) -> int:
        """Sum of per-leaf direct parameter counts."""
        return sum(leaf.params for leaf in self.leaves)


def _projection_role(name: str, last: str) -> str | None:
    if last in _FUSED_QKV:
        return "qkv_proj"
    if last == "kv_proj":
        return "kv_proj"
    for suffix, role in _SUFFIX_ROLES:
        if last.endswith(suffix):
            return role
    in_attn = ".attn" in name or ".self_attn" in name or ".attention" in name
    in_mlp = ".mlp" in name or ".feed_forward" in name or ".ffn" in name
    if last == "c_proj":
        return "o_proj" if in_attn else "down_proj" if in_mlp else "other"
    if last in _EXACT_ROLES:
        return _EXACT_ROLES[last]
    if last == "dense":
        return "o_proj" if in_attn else "other"
    return None


def _other_role(name: str, last: str, module_type: str) -> str:
    if "lm_head" in name:
        return "lm_head"
    if any(k in last for k in ("wte", "embed_tokens", "word_embeddings")):
        return "embedding"
    if any(k in last for k in ("wpe", "embed_positions", "position_embeddings")):
        return "pos_embedding"
    if "rotary" in name or "rope" in name:
        return "rotary"
    if "norm" in last or last.startswith("ln") or "layernorm" in last:
        return "norm"
    if module_type in _ACTIVATION_TYPES or last in ("act", "act_fn", "activation_fn"):
        return "activation"
    if module_type == "Dropout" or "dropout" in last:
        return "dropout"
    return "embedding" if "embed" in last else "other"


def classify_role(layer_name: str, module_type: str) -> str:
    """Coarse semantic role (``q_proj`` ... ``down_proj``, ``norm``, ``embedding``, ``lm_head``, ...)."""
    name = layer_name.lower()
    last = name.rsplit(".", 1)[-1]
    return _projection_role(name, last) or _other_role(name, last, module_type)


def _matmul_features(module: torch.nn.Module) -> tuple[int | None, int | None]:
    """``(in, out)`` of ``nn.Linear`` and HF ``Conv1D`` (weight stored ``(in, out)``), else ``(None, None)``."""
    import torch

    if isinstance(module, torch.nn.Linear):
        return int(module.in_features), int(module.out_features)
    if hasattr(module, "nf") and hasattr(module, "weight") and module.weight.dim() == 2:
        return int(module.weight.shape[0]), int(module.weight.shape[1])
    return None, None


def _leaf(name: str, module: torch.nn.Module, activations: int | None) -> Leaf:
    in_f, out_f = _matmul_features(module)
    return Leaf(name, classify_role(name, type(module).__name__),
                int(sum(p.numel() for p in module.parameters(recurse=False))), activations, in_f, out_f)


def trace_leaves(model: torch.nn.Module, example_input: torch.Tensor) -> list[Leaf]:
    """One leaf per hook firing of a plain ``model(example_input)``, plus unfired rotary leaves."""
    import torch

    records: list[Leaf] = []
    handles: list[Any] = []

    def register(module: torch.nn.Module, prefix: str) -> None:
        if not list(module.children()):
            handles.append(module.register_forward_hook(
                lambda mod, _i, out, name=prefix: records.append(
                    _leaf(name, mod, int(out.numel()) if isinstance(out, torch.Tensor) else None))
            ))
            return
        for child_name, child in module.named_children():
            register(child, f"{prefix}.{child_name}" if prefix else child_name)

    register(model, "")
    try:
        with torch.no_grad():
            model(example_input)
    finally:
        for h in handles:
            h.remove()
    seen = {leaf.layer_name for leaf in records}
    for name, module in model.named_modules():
        if name and not list(module.children()) and name not in seen:
            leaf = _leaf(name, module, None)
            if leaf.role in _SUPPLEMENTAL_ROLES:
                records.append(leaf)
    return records


def _config_get(config: Any, *names: str) -> Any:
    for name in names:
        val = getattr(config, name, None)
        if val is not None:
            return val
    return None


def _attention_dims(config: Any) -> tuple[int | None, int | None, int | None]:
    """``(n_heads, n_kv_heads, d_head)``; ``n_kv_heads`` defaults to ``n_heads`` (MHA)."""
    hidden = _config_get(config, "hidden_size", "n_embd", "d_model")
    n_heads = _config_get(config, "num_attention_heads", "n_head", "num_heads")
    n_kv = _config_get(config, "num_key_value_heads", "num_kv_heads") or n_heads
    d_head = _config_get(config, "head_dim")
    if d_head is None and hidden is not None and n_heads:
        d_head = int(hidden) // int(n_heads)
    return (int(n_heads) if n_heads else None, int(n_kv) if n_kv else None, int(d_head) if d_head else None)


def _build_and_trace(config: Any, device: str, seq_len: int) -> list[Leaf]:
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.modeling_utils import no_init_weights

    if device == "meta":
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(config)
    else:
        with no_init_weights():
            model = AutoModelForCausalLM.from_config(config).float()
    model.eval()
    return trace_leaves(model, torch.zeros((1, seq_len), dtype=torch.long, device=device))


def extract_structure(model_id: str, revision: str | None = None, seq_len: int = TRACE_SEQ_LEN) -> ModelStructure:
    """
    Structural constants of ``model_id`` without downloading weights.

    Only the config is fetched; the architecture is built on ``meta`` (fallback: CPU
    without weight init) with eager attention in float32, then traced once.

    Raises
    ------
    StructuralExtractionError
        Config unavailable (offline, gated, unknown architecture) or no device can trace it.
    """
    import torch
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_id, revision=revision)
    except Exception as exc:  # noqa: BLE001 - any Hub/config failure is reported as one typed error
        raise StructuralExtractionError(f"cannot load config of {model_id!r}: {type(exc).__name__}: {exc}") from exc
    config._attn_implementation = "eager"  # noqa: SLF001
    config.torch_dtype = torch.float32
    errors: list[str] = []
    for device in ("meta", "cpu"):
        try:
            leaves = _build_and_trace(config, device, seq_len)
        except Exception as exc:  # noqa: BLE001 - some kernels do not run on meta; try CPU
            errors.append(f"{device}: {type(exc).__name__}: {exc}")
            continue
        return ModelStructure(model_id, seq_len, *_attention_dims(config), leaves=leaves)
    raise StructuralExtractionError(f"cannot build/trace {model_id!r}: {'; '.join(errors)}")
