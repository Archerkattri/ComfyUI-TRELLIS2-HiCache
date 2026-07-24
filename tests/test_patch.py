"""Standalone unit tests for the TRELLIS-HiCache patch logic.

No ComfyUI, no GPU, no TRELLIS: a dummy DiT and a fake SparseTensor stand in for
the real models, so the scheduling / CFG-routing / run-boundary / copy-on-patch
logic is checked deterministically on CPU.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from trellis_hicache_patch import (
    HiCacheModelPatch, apply_hicache, remove_hicache, validate_config,
)


class DummyDiT(torch.nn.Module):
    """Counts forward calls; returns a deterministic dense velocity."""
    def __init__(self, dim=8):
        super().__init__()
        self.dim = dim
        self.calls = 0
        self.lin = torch.nn.Linear(dim, dim)

    def forward(self, x, t, cond=None, **kw):
        self.calls += 1
        # smooth-in-t signal so the Hermite forecast is accurate
        tv = float(t.reshape(-1)[0]) if torch.is_tensor(t) else float(t)
        return torch.ones(1, self.dim) * tv


class FakeSparse:
    """Minimal stand-in for a TRELLIS SparseTensor: .feats + .replace()."""
    def __init__(self, feats):
        self.feats = feats
    def replace(self, feats):
        return FakeSparse(feats)


def _trellis_t_seq(steps=25):
    # TRELLIS scales t to [0, 1000], decreasing.
    import numpy as np
    return [float(x) for x in np.linspace(1, 0, steps + 1)[:-1] * 1000.0]


def test_validate_config_rejects_bad_params():
    with pytest.raises(ValueError):
        validate_config("nope", 3, 2, 1, 0.5, 5)
    with pytest.raises(ValueError):
        validate_config("hermite", 0, 2, 1, 0.5, 5)
    with pytest.raises(ValueError):
        validate_config("hermite", 3, 2, 1, 1.5, 5)


def test_skips_steps_on_decreasing_trellis_schedule():
    dit = DummyDiT()
    patch = HiCacheModelPatch(dit, method="hermite", interval=3, warmup_steps=2)
    for t in _trellis_t_seq(25):
        patch(torch.zeros(1, 8), t)
    # 25 steps, interval 3: clearly fewer real forwards than steps, some skipped.
    assert patch.skipped_steps > 0
    assert dit.calls == patch.computed_steps
    assert patch.computed_steps + patch.skipped_steps == 25


def test_split_cfg_routes_two_states():
    """Each timestep called twice (cond, uncond) must use two forecast states and
    still skip; the DiT must be called far fewer than 2x steps."""
    dit = DummyDiT()
    patch = HiCacheModelPatch(dit, method="hermite", interval=3, warmup_steps=2)
    for t in _trellis_t_seq(25):
        patch(torch.zeros(1, 8), t)   # cond
        patch(torch.zeros(1, 8), t)   # uncond (repeated t)
    assert patch.skipped_steps > 0
    assert patch.computed_steps + patch.skipped_steps == 50
    assert dit.calls < 50            # genuinely skipped DiT forwards


def test_new_run_resets_on_direction_reversal():
    dit = DummyDiT()
    patch = HiCacheModelPatch(dit, method="hermite", interval=3, warmup_steps=2)
    for t in _trellis_t_seq(10):
        patch(torch.zeros(1, 8), t)
    first = patch.computed_steps + patch.skipped_steps
    # a second run restarts t near 1000 (direction flips up) -> reset
    for t in _trellis_t_seq(10):
        patch(torch.zeros(1, 8), t)
    assert patch.computed_steps + patch.skipped_steps == 10  # counters reset, not 20
    assert first == 10


def test_sparse_output_forecast_rebuilds_sparse():
    """SLaT-style sparse output: forecast must return a FakeSparse, not a tensor."""
    class SparseDiT(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = 0
        def forward(self, x, t, cond=None, **kw):
            self.calls += 1
            tv = float(t.reshape(-1)[0]) if torch.is_tensor(t) else float(t)
            return FakeSparse(torch.ones(4, 8) * tv)
    patch = HiCacheModelPatch(SparseDiT(), method="hermite", interval=2, warmup_steps=1)
    outs = [patch(None, t) for t in _trellis_t_seq(12)]
    assert all(isinstance(o, FakeSparse) for o in outs)
    assert patch.skipped_steps > 0


def test_apply_remove_is_copy_on_patch():
    class Pipe:
        def __init__(self):
            self.models = {"sparse_structure_flow_model": DummyDiT(),
                           "shape_slat_flow_model_512": DummyDiT(), "shape_slat_flow_model_1024": DummyDiT()}
    p = Pipe()
    patched = apply_hicache(p, method="hermite", interval=2, stages="both")
    # original untouched
    assert not getattr(p.models["shape_slat_flow_model_512"], "_hicache_is_patch", False)
    assert getattr(patched.models["shape_slat_flow_model_512"], "_hicache_is_patch", False)
    assert patched.models is not p.models
    # remove restores
    clean = remove_hicache(patched)
    assert not getattr(clean.models["shape_slat_flow_model_512"], "_hicache_is_patch", False)


def test_apply_never_nests():
    class Pipe:
        def __init__(self):
            self.models = {"sparse_structure_flow_model": DummyDiT(),
                           "shape_slat_flow_model_512": DummyDiT(), "shape_slat_flow_model_1024": DummyDiT()}
    p = Pipe()
    a = apply_hicache(p, interval=2, stages="both")
    b = apply_hicache(a, interval=3, stages="both")  # re-patch
    inner = b.models["shape_slat_flow_model_512"].inner
    assert not getattr(inner, "_hicache_is_patch", False)  # unwrapped, not nested
    assert b.models["shape_slat_flow_model_512"].interval == 3


def test_wrap_none_does_not_crash_on_attr_access():
    """Regression for issue #1: GGUF/lazy pipelines leave pipeline.models[key]=None
    at patch time. Wrapping None must not raise, and an unknown-attribute lookup
    must give a clear 'not loaded yet' error, NOT the misleading
    'HiCacheModelPatch object has no attribute inner'."""
    patch = HiCacheModelPatch(None, method="hermite", interval=3, warmup_steps=2)
    assert patch.inner is None                     # readable, no crash
    with pytest.raises(AttributeError) as ei:
        _ = patch.some_config_attr                 # forwarded lookup on a None inner
    msg = str(ei.value)
    assert "not loaded yet" in msg and "some_config_attr" in msg
    # a compute step on an unbound patch fails loudly, not silently
    with pytest.raises(RuntimeError):
        patch(torch.zeros(1, 8), _trellis_t_seq(4)[0])


def test_bind_inner_materializes_lazy_model():
    """After the real DiT loads, bind_inner attaches it and forwards run/skip."""
    patch = HiCacheModelPatch(None, method="hermite", interval=3, warmup_steps=2)
    dit = DummyDiT()
    assert patch.bind_inner(dit) is patch
    assert patch.inner is dit
    assert "inner" in patch._modules          # registered as a submodule
    for t in _trellis_t_seq(25):
        patch(torch.zeros(1, 8), t)
    assert patch.skipped_steps > 0
    assert dit.calls == patch.computed_steps
    assert patch.computed_steps + patch.skipped_steps == 25


def test_lazy_pipeline_autobinds_on_assignment():
    """End-to-end lazy flow: patch a pipeline whose flow model is still None, then
    the sampler assigns the real model into pipeline.models[key]; the patch must
    bind it (not be overwritten) and then skip DiT steps."""
    class LazyPipe:
        def __init__(self):
            self.models = {"sparse_structure_flow_model": None,
                           "shape_slat_flow_model_512": None,
                           "shape_slat_flow_model_1024": None}
    p = LazyPipe()
    patched = apply_hicache(p, method="hermite", interval=3, warmup_steps=2, stages="sparse_structure")
    slot = patched.models["sparse_structure_flow_model"]
    assert getattr(slot, "_hicache_is_patch", False)
    assert slot.inner is None                      # deferred
    # sampler materializes the real DiT (the exact GGUF lazy-load path)
    dit = DummyDiT()
    patched.models["sparse_structure_flow_model"] = dit
    # same patch object, now bound -- not overwritten by the raw DiT
    assert patched.models["sparse_structure_flow_model"] is slot
    assert slot.inner is dit
    for t in _trellis_t_seq(25):
        slot(torch.zeros(1, 8), t)
    assert slot.skipped_steps > 0
    assert dit.calls == slot.computed_steps


def test_eager_inner_still_registered_as_submodule():
    """Eager path unbroken: a real nn.Module inner stays a registered submodule so
    state_dict / parameters / device moves recurse into it."""
    dit = DummyDiT()
    patch = HiCacheModelPatch(dit, method="hermite", interval=3)
    assert patch.inner is dit
    assert "inner" in patch._modules
    # inner's parameters are reachable through the patch (state_dict recurses)
    assert any(k.startswith("inner.") for k in patch.state_dict().keys())
    assert any(p is q for q in dit.parameters() for p in patch.parameters())


def test_stages_selector():
    class Pipe:
        def __init__(self):
            self.models = {"sparse_structure_flow_model": DummyDiT(),
                           "shape_slat_flow_model_512": DummyDiT(), "shape_slat_flow_model_1024": DummyDiT()}
    p = Pipe()
    ss = apply_hicache(p, stages="sparse_structure")
    assert getattr(ss.models["sparse_structure_flow_model"], "_hicache_is_patch", False)
    assert not getattr(ss.models["shape_slat_flow_model_512"], "_hicache_is_patch", False)
