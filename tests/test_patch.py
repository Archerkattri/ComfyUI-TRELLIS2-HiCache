"""Standalone unit tests for the TRELLIS-HiCache patch logic.

No ComfyUI, no GPU, no TRELLIS: a dummy DiT and a fake SparseTensor stand in for
the real models, so the scheduling / CFG-routing / run-boundary / copy-on-patch
logic is checked deterministically on CPU.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest
from trellis_hicache_patch import (
    HiCacheModelPatch, apply_hicache, remove_hicache, validate_config,
)
from nodes import Trellis2HiCacheAccelerate


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
    with pytest.raises(ValueError, match="dmd_history"):
        validate_config("dmd", 3, 2, 1, 0.5, 3)
    with pytest.raises(ValueError, match="dmd_history"):
        validate_config("auto", 3, 2, 1, 0.5, 4)


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


def test_explicit_run_branch_identity_isolates_same_timestep_retry():
    """Explicit job/CFG markers disambiguate retry from split CFG."""
    dit = DummyDiT()
    patch = HiCacheModelPatch(dit, method="hermite", interval=3, warmup_steps=1)
    t0 = torch.tensor([1000.0])
    x = torch.zeros(1, 8)

    patch(x, t0, hicache_run_id="run-a", hicache_branch_id="cond")
    assert patch.run_id == "run-a"
    assert patch.stage_id == "unknown"
    assert patch.branch_id == "cond"
    patch(x, t0, hicache_run_id="run-a", hicache_branch_id="uncond")
    assert patch.branch_id == "uncond"
    assert patch.telemetry["branches"]["cond"]["decisions"]["full"] == 1
    assert patch.telemetry["branches"]["uncond"]["decisions"]["full"] == 1

    patch(x, t0, hicache_run_id="run-b", hicache_branch_id="cond")
    assert patch.run_id == "run-b"
    assert patch.computed_steps == 1 and patch.skipped_steps == 0
    assert dit.calls == 3


def test_telemetry_reports_actual_dmd_and_fallback_per_stage():
    dit = DummyDiT()
    patch = HiCacheModelPatch(dit, method="dmd", interval=3,
                              warmup_steps=2, dmd_history=4,
                              stage_id="tex_slat_flow_model_1024")
    for t in _trellis_t_seq(35):
        patch(torch.zeros(1, 8), t)
    telemetry = patch.telemetry
    assert telemetry["run_id"] == patch.run_id
    assert telemetry["stage_id"] == "tex_slat_flow_model_1024"
    assert telemetry["method_counts"]["hermite"] > 0
    assert telemetry["method_counts"]["dmd"] > 0
    assert sum(telemetry["fallbacks"].values()) > 0
    telemetry["decisions"]["full"] = -1
    assert patch.telemetry["decisions"]["full"] >= 1


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
        def __init__(self):
            super().__init__()
            self.calls = 0
        def forward(self, x, t, cond=None, **kw):
            self.calls += 1
            tv = float(t.reshape(-1)[0]) if torch.is_tensor(t) else float(t)
            return FakeSparse(torch.ones(4, 8) * tv)
    patch = HiCacheModelPatch(SparseDiT(), method="hermite", interval=2, warmup_steps=1)
    outs = [patch(None, t) for t in _trellis_t_seq(12)]
    assert all(isinstance(o, FakeSparse) for o in outs)
    assert patch.skipped_steps > 0


def test_interval_one_is_a_no_cache_full_compute_bypass():
    dit = DummyDiT()
    patch = HiCacheModelPatch(dit, method="hermite", interval=1,
                              warmup_steps=0)
    for t in _trellis_t_seq(12):
        patch(torch.zeros(1, 8), t)
    assert dit.calls == 12
    assert patch.skipped_steps == 0


def test_node_disabled_path_restores_stock_model():
    class Pipe:
        def __init__(self):
            self.models = {"sparse_structure_flow_model": DummyDiT(),
                           "shape_slat_flow_model_512": DummyDiT()}

    p = Pipe()
    enabled = Trellis2HiCacheAccelerate().patch(p, enabled=True, stages="both")[0]
    restored = Trellis2HiCacheAccelerate().patch(
        enabled, enabled=False, stages="both")[0]
    assert restored is not enabled
    assert all(not getattr(m, "_hicache_is_patch", False)
               for m in restored.models.values())


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


def test_lazy_pipeline_guard_then_load_still_triggers():
    """A lazy loader's ``is None`` guard must still run, including after unload."""
    class LazyPipe:
        def __init__(self):
            self.models = {"sparse_structure_flow_model": None,
                           "shape_slat_flow_model_512": None,
                           "shape_slat_flow_model_1024": None}
            self.loads = []

        def load_sparse_structure_model(self):
            if self.models["sparse_structure_flow_model"] is None:
                model = DummyDiT()
                self.loads.append(model)
                self.models["sparse_structure_flow_model"] = model
            return self.models["sparse_structure_flow_model"]

    p = LazyPipe()
    patched = apply_hicache(p, method="hermite", interval=3, warmup_steps=2, stages="sparse_structure")
    assert patched.models["sparse_structure_flow_model"] is None

    slot = patched.load_sparse_structure_model()
    assert getattr(slot, "_hicache_is_patch", False)
    assert slot.inner is patched.loads[0]
    for t in _trellis_t_seq(25):
        slot(torch.zeros(1, 8), t)
    assert slot.skipped_steps > 0
    assert patched.loads[0].calls == slot.computed_steps

    patched.models["sparse_structure_flow_model"] = None
    assert patched.models["sparse_structure_flow_model"] is None
    replacement = patched.load_sparse_structure_model()
    assert len(patched.loads) == 2
    assert replacement is not slot
    assert getattr(replacement, "_hicache_is_patch", False)
    assert replacement.inner is patched.loads[1]


def test_lazy_reapply_updates_config_across_unload_reload():
    """Re-patching a loaded lazy slot must update its future reload config."""
    class LazyPipe:
        def __init__(self):
            self.models = {
                "sparse_structure_flow_model": None,
                "shape_slat_flow_model_512": None,
                "shape_slat_flow_model_1024": None,
            }

    first = apply_hicache(
        LazyPipe(), method="hermite", interval=3, stages="sparse_structure",
    )
    first.models["sparse_structure_flow_model"] = DummyDiT()
    assert first.models["sparse_structure_flow_model"].interval == 3

    second = apply_hicache(
        first, method="dmd", interval=5, stages="sparse_structure",
    )
    loaded = second.models["sparse_structure_flow_model"]
    assert isinstance(loaded, HiCacheModelPatch)
    assert loaded.interval == 5
    assert loaded.method == "dmd"
    assert second.models._pending["sparse_structure_flow_model"]["interval"] == 5
    assert second.models._pending["sparse_structure_flow_model"]["method"] == "dmd"

    second.models["sparse_structure_flow_model"] = None
    second.models["sparse_structure_flow_model"] = DummyDiT()
    reloaded = second.models["sparse_structure_flow_model"]
    assert isinstance(reloaded, HiCacheModelPatch)
    assert reloaded.interval == 5
    assert reloaded.method == "dmd"


def test_lazy_subset_reapply_preserves_unselected_pending_stages():
    """A subset re-patch must not disable deferred wrapping for other stages."""
    class LazyPipe:
        def __init__(self):
            self.models = {
                "sparse_structure_flow_model": None,
                "shape_slat_flow_model_512": None,
                "shape_slat_flow_model_1024": None,
                "tex_slat_flow_model_512": None,
                "tex_slat_flow_model_1024": None,
            }

    all_stages = apply_hicache(
        LazyPipe(), method="hermite", interval=3, stages="all",
    )
    all_stages.models["sparse_structure_flow_model"] = DummyDiT()

    sparse_only = apply_hicache(
        all_stages, method="dmd", interval=5, stages="sparse_structure",
    )
    assert set(sparse_only.models._pending) == set(all_stages.models)
    assert sparse_only.models._pending["sparse_structure_flow_model"]["interval"] == 5
    assert sparse_only.models._pending["shape_slat_flow_model_512"]["interval"] == 3
    assert sparse_only.models._pending["tex_slat_flow_model_1024"]["method"] == "hermite"

    sparse_only.models["shape_slat_flow_model_512"] = DummyDiT()
    shape = sparse_only.models["shape_slat_flow_model_512"]
    assert isinstance(shape, HiCacheModelPatch)
    assert shape.interval == 3
    assert shape.method == "hermite"

    sparse_only.models["tex_slat_flow_model_1024"] = DummyDiT()
    texture = sparse_only.models["tex_slat_flow_model_1024"]
    assert isinstance(texture, HiCacheModelPatch)
    assert texture.interval == 3
    assert texture.method == "hermite"


class GGUFLikeLazyPipe:
    """Mirrors the real Aero-Ex/ComfyUI-Trellis2-GGUF load/unload pair, not just
    a generic lazy pipeline: the loader guards on ``is None``
    (``load_sparse_structure_model`` / ``load_shape_slat_flow_model_512``), and
    the unloader ``del``s the dict entry before resetting it to ``None``
    (``unload_sparse_structure_model`` / ``unload_shape_slat_flow_model_512``)
    rather than reassigning directly. Two independent stages are modeled since
    ``keep_models_loaded=False`` unloads/reloads each slot on its own schedule
    within a single run.
    """

    def __init__(self):
        self.models = {
            "sparse_structure_flow_model": None,
            "shape_slat_flow_model_512": None,
        }
        self.load_counts = {"sparse_structure_flow_model": 0, "shape_slat_flow_model_512": 0}

    def load_sparse_structure_model(self):
        if self.models["sparse_structure_flow_model"] is None:
            self.load_counts["sparse_structure_flow_model"] += 1
            self.models["sparse_structure_flow_model"] = DummyDiT()

    def unload_sparse_structure_model(self):
        if self.models["sparse_structure_flow_model"] is not None:
            del self.models["sparse_structure_flow_model"]
            self.models["sparse_structure_flow_model"] = None

    def load_shape_slat_flow_model_512(self):
        if self.models["shape_slat_flow_model_512"] is None:
            self.load_counts["shape_slat_flow_model_512"] += 1
            self.models["shape_slat_flow_model_512"] = DummyDiT()

    def unload_shape_slat_flow_model_512(self):
        if self.models["shape_slat_flow_model_512"] is not None:
            del self.models["shape_slat_flow_model_512"]
            self.models["shape_slat_flow_model_512"] = None


def test_gguf_del_then_none_unload_still_rewraps_on_reload():
    """Regression for the real GGUF unload cycle: it deletes the dict key
    before resetting it to None, rather than reassigning None directly. `del`
    only removes the dict entry -- it must not disturb `_pending`, or the very
    next `is None`-guarded reload would silently load an unwrapped model.
    Covers two independently-cycled stages, matching the real pipeline's
    per-slot VRAM management."""
    p = GGUFLikeLazyPipe()
    patched = apply_hicache(p, method="hermite", interval=4, stages="both")
    assert patched.models["sparse_structure_flow_model"] is None
    assert patched.models["shape_slat_flow_model_512"] is None

    patched.load_sparse_structure_model()
    patched.load_shape_slat_flow_model_512()
    ss1 = patched.models["sparse_structure_flow_model"]
    shape1 = patched.models["shape_slat_flow_model_512"]
    assert getattr(ss1, "_hicache_is_patch", False)
    assert getattr(shape1, "_hicache_is_patch", False)

    # unload one stage the exact real way (del, then reset to None) while the
    # other stays loaded -- independent lifecycles, as the real pipeline does.
    patched.unload_sparse_structure_model()
    assert patched.models["sparse_structure_flow_model"] is None
    assert patched.models["shape_slat_flow_model_512"] is shape1  # untouched

    # reload: the `is None` guard must still fire, and the fresh model must be wrapped
    patched.load_sparse_structure_model()
    ss2 = patched.models["sparse_structure_flow_model"]
    assert patched.load_counts["sparse_structure_flow_model"] == 2
    assert ss2 is not ss1
    assert getattr(ss2, "_hicache_is_patch", False)
    assert ss2.interval == 4

    # unload+reload the other stage too, confirming both slots independently
    # survive the del/reload cycle
    patched.unload_shape_slat_flow_model_512()
    patched.load_shape_slat_flow_model_512()
    shape2 = patched.models["shape_slat_flow_model_512"]
    assert patched.load_counts["shape_slat_flow_model_512"] == 2
    assert shape2 is not shape1
    assert getattr(shape2, "_hicache_is_patch", False)
    assert shape2.interval == 4


def test_remove_hicache_clears_pending_before_any_load():
    """Disabling a pending patch must not re-enable it on a later assignment."""
    class LazyPipe:
        def __init__(self):
            self.models = {"sparse_structure_flow_model": None,
                           "shape_slat_flow_model_512": None,
                           "shape_slat_flow_model_1024": None}

    patched = apply_hicache(
        LazyPipe(), method="hermite", interval=3, warmup_steps=2,
        stages="sparse_structure",
    )
    clean = remove_hicache(patched)
    assert type(clean.models) is dict
    assert clean.models["sparse_structure_flow_model"] is None

    dit = DummyDiT()
    clean.models["sparse_structure_flow_model"] = dit
    assert clean.models["sparse_structure_flow_model"] is dit


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


def test_all_stage_selector_covers_five_slots_and_texture_only():
    class Pipe:
        def __init__(self):
            self.models = {
                "sparse_structure_flow_model": DummyDiT(),
                "shape_slat_flow_model_512": DummyDiT(),
                "shape_slat_flow_model_1024": DummyDiT(),
                "tex_slat_flow_model_512": DummyDiT(),
                "tex_slat_flow_model_1024": DummyDiT(),
            }

    p = Pipe()
    all_stages = apply_hicache(p, stages="all")
    assert all(
        getattr(all_stages.models[key], "_hicache_is_patch", False)
        for key in p.models
    )
    assert all_stages.models["tex_slat_flow_model_512"].stage_id == (
        "tex_slat_flow_model_512")

    texture = apply_hicache(p, stages="texture")
    assert all(
        getattr(texture.models[key], "_hicache_is_patch", False)
        for key in ("tex_slat_flow_model_512", "tex_slat_flow_model_1024")
    )
    assert not getattr(
        texture.models["sparse_structure_flow_model"], "_hicache_is_patch", False)
