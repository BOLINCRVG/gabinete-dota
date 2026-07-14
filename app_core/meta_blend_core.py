"""Backward-compat re-export shim. Implementation lives in mlcore.meta_blend_core."""
from __future__ import annotations
from mlcore.meta_blend_core import (
    META_BLEND_C_MODEL_FEATURES,
    should_apply_meta_blend_c_model_gate,
    build_meta_blend_c_model_features,
    apply_meta_blend_c_model,
    apply_meta_blend_c_lite,
)

__all__ = [
    "META_BLEND_C_MODEL_FEATURES",
    "apply_meta_blend_c_lite",
    "apply_meta_blend_c_model",
    "build_meta_blend_c_model_features",
    "should_apply_meta_blend_c_model_gate",
]
