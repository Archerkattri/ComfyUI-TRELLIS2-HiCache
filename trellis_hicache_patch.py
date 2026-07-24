"""HiCache / HiCache++ acceleration for TRELLIS image-to-3D, as a model-level
patch -- pure Python, no ComfyUI imports.

This module holds all of the acceleration logic so it can be unit-tested
standalone (no ComfyUI, no GPU needed for the shape-only tests). ``nodes.py``
only does the ComfyUI plumbing around :func:`apply_hicache` / :func:`remove_hicache`.

How it works
------------
TRELLIS runs two flow-matching stages, each a DiT stored in the pipeline's
``models`` dict and called once per sampling step::

    flow_model = self.models['sparse_structure_flow_model']   # dense latent
    flow_model = self.models['slat_flow_model']                # SparseTensor
    ...
    pred_v = model(x_t, t, cond)                                # one call / step

(see ``TrellisImageTo3DPipeline.sample_sparse_structure`` / ``sample_slat``).
Replacing a ``models[...]`` entry with :class:`HiCacheModelPatch` therefore
intercepts every DiT forward of that stage. On *compute* steps the wrapped DiT
runs normally and its output (the velocity) is cached as a forecast anchor; on
*skipped* steps the DiT is **not called** -- the velocity is forecast from the
cached anchors with ``hicache-pp``:

* ``hermite`` -- HiCache (dual-scaled physicist's Hermite polynomial, arXiv:2508.16984).
* ``dmd``     -- HiCache++ (Dynamic Mode Decomposition / Prony exponential basis;
  lossless at larger skip intervals than the polynomial on the feature-ODE class).
* ``auto``    -- holdout-selected per compute step: serve DMD only when it
  demonstrably beats the polynomial on the cached window.

This is the same model-level patch shipped in ComfyUI-HiCache (Hunyuan3D); the
only TRELLIS-specific part is the SLaT stage, whose DiT returns a TRELLIS
``SparseTensor`` rather than a plain tensor. The sparse layout is fixed during a
SLaT run (the active voxels are decided by the sparse-structure stage), so the
velocity's ``.feats`` matrix has constant shape across the run: we forecast on
``.feats`` and rebuild the SparseTensor from the last computed step via
``template.replace(forecast_feats)``.

Two correctness details specific to TRELLIS's ``FlowEulerSampler``:

* **Timesteps run 1 -> 0** (``t_seq = linspace(1, 0, steps+1)``), the opposite of
  the 0 -> 1 schedule in the Hunyuan pipelines. Run-boundary detection therefore
  cannot key on the sign of the step; it keys on a *large* jump in ``t`` (a new
  run restarts ``t`` near 1, a jump of ~1.0, versus ~1/steps within a run).
* **Classifier-free guidance is two separate forwards per step**, not one batched
  forward: the sampler calls ``model(x, t, cond)`` then ``model(x, t, neg_cond)``
  at the *same* ``t``. A single forecast state would interleave the two
  trajectories and corrupt the forecast, so the patch keeps **two parallel HiCache
  states** (conditional / unconditional) and routes each forward by whether ``t``
  repeated the previous call's value. With CFG disabled (one forward per step) only
  the conditional state is used. This makes the patch correct for both the batched
  (Hunyuan) and the split (TRELLIS) CFG conventions.

Cache safety inside ComfyUI: :func:`apply_hicache` / :func:`remove_hicache` never
mutate the pipeline they are given -- they return a shallow copy whose ``models``
dict is replaced (weights are shared, so this is free). ComfyUI caches node
*outputs* keyed on node *inputs*, so a cached output must own its configuration
forever; copy-on-patch guarantees that.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Dict, List, Optional

import torch

from hicache_pp import (
    hicache_init,
    hicache_decide,
    hicache_update_derivatives,
    hicache_forecast,
    dmd_update_snapshots,
    dmd_forecast_state,
    auto_forecast_state,
)

logger = logging.getLogger("ComfyUI-TRELLIS-HiCache")

METHODS = ("hermite", "dmd", "auto")
STAGES = ("both", "sparse_structure", "shape", "texture", "all")

# TRELLIS.2's flow DiTs, keyed in pipeline.models. v2 splits the SLaT stage into
# resolution variants (512 / 1024) and adds a separate texture flow. Only the
# models actually exercised by a run accumulate forecast state, so patching a
# resolution variant that the run does not use is harmless.
_SS = ["sparse_structure_flow_model"]
_SHAPE = ["shape_slat_flow_model_512", "shape_slat_flow_model_1024"]
_TEX = ["tex_slat_flow_model_512", "tex_slat_flow_model_1024"]
_STAGE_KEYS = {
    "sparse_structure": _SS,
    "shape": _SHAPE,
    "texture": _TEX,
    "both": _SS + _SHAPE,        # shape generation (default): SS + shape SLaT
    "all": _SS + _SHAPE + _TEX,  # also accelerate texture synthesis
}

# Sentinel for "unknown total step count": disables the end-of-run always-compute
# window (any real run is far shorter than this).
_NO_END_WINDOW = 1_000_000


def validate_config(method: str, interval: int, warmup_steps: int,
                    max_order: int, sigma: float, dmd_history: int) -> None:
    """Raise ValueError on bad node parameters (mirrors hicache_pp's checks)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if interval < 1:
        raise ValueError(f"interval must be >= 1, got {interval}")
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
    if max_order < 1:
        raise ValueError(f"max_order must be >= 1, got {max_order}")
    if not (0.0 < sigma < 1.0):
        raise ValueError(f"sigma must be in (0, 1), got {sigma}")
    if dmd_history < 3:
        raise ValueError(f"dmd_history must be >= 3, got {dmd_history}")


def _is_sparse(x: Any) -> bool:
    """Duck-type a TRELLIS SparseTensor (has a .feats tensor and a .replace)."""
    return hasattr(x, "feats") and hasattr(x, "replace") and not torch.is_tensor(x)


class HiCacheModelPatch(torch.nn.Module):
    """Drop-in replacement for a TRELLIS flow DiT that skips forwards.

    Wraps the original DiT; forwards unknown attribute lookups to it so pipeline
    code (``.to(device)``, ``.dtype``, config access) keeps working. Handles both
    dense-tensor (sparse-structure stage) and SparseTensor (SLaT stage) outputs.
    """

    def __init__(self, model: Optional[torch.nn.Module], *, method: str = "hermite",
                 interval: int = 3, warmup_steps: int = 2, max_order: int = 1,
                 sigma: float = 0.5, dmd_history: int = 5) -> None:
        validate_config(method, interval, warmup_steps, max_order, sigma, dmd_history)
        super().__init__()
        # ``model`` may be None here: GGUF / lazy Trellis2 pipelines
        # (e.g. Aero-Ex/ComfyUI-Trellis2-GGUF) leave pipeline.models[key] empty at
        # patch time and materialize the flow DiT later, inside the sampler. Store
        # the wrapped model so it survives both cases -- a real nn.Module is
        # registered as a submodule (so .to()/.state_dict()/.parameters() recurse),
        # while a None/lazy placeholder is kept as a plain attribute. See _set_inner.
        self._set_inner(model)
        self._hicache_is_patch = True
        self.method = method
        self.interval = int(interval)
        self.warmup_steps = int(warmup_steps)
        self.max_order = int(max_order)
        self.sigma = float(sigma)
        self.dmd_history = int(dmd_history)

        # two parallel forecast states: conditional + unconditional CFG branches
        self._state_cond: Optional[Dict[str, Any]] = None
        self._state_uncond: Optional[Dict[str, Any]] = None
        self._tmpl_cond: Any = None     # last computed SparseTensor (cond branch)
        self._tmpl_uncond: Any = None   # last computed SparseTensor (uncond branch)
        self._last_t: Optional[float] = None
        self._run_dir: Optional[int] = None   # sign of t-progression within a run
        self.computed_steps = 0
        self.skipped_steps = 0

    def _set_inner(self, model: Optional[torch.nn.Module]) -> None:
        """Store the wrapped model so ``self.inner`` is always resolvable.

        A real ``nn.Module`` is registered in ``_modules`` (so device moves and
        ``state_dict`` recurse into it, exactly as the eager path relied on). A
        ``None`` placeholder -- or any non-Module -- is kept as a plain attribute
        in ``__dict__``. The bug this avoids: ``nn.Module.__setattr__`` only routes
        real Modules into ``_modules``; a None assigned to ``self.inner`` lands in
        ``__dict__`` instead, and ``nn.Module.__getattr__`` never searches
        ``__dict__`` -- so the old ``__getattr__`` fallback raised the misleading
        ``'HiCacheModelPatch' object has no attribute 'inner'`` for lazy pipelines.
        """
        for d in (self.__dict__, self._parameters, self._buffers, self._modules):
            d.pop("inner", None)
        if isinstance(model, torch.nn.Module):
            self._modules["inner"] = model
        else:
            object.__setattr__(self, "inner", model)

    def _inner(self) -> Any:
        """The wrapped model from wherever it lives (submodule or plain attr)."""
        mod = self.__dict__.get("_modules")
        if mod is not None and "inner" in mod:
            return mod["inner"]
        return self.__dict__.get("inner")

    def bind_inner(self, model: torch.nn.Module) -> "HiCacheModelPatch":
        """Attach the real flow model to a patch created around a lazy (None)
        placeholder, and register it as a submodule. Lets GGUF / lazy pipelines
        that materialize the DiT after patching still route through the cache."""
        self._set_inner(model)
        return self

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        inner = self._inner()
        if inner is None:
            raise AttributeError(
                f"{type(self).__name__!r} has no attribute {name!r}: the wrapped "
                f"flow model is not loaded yet (inner is None). Lazy / GGUF Trellis2 "
                f"pipelines materialize the DiT inside the sampler; the patch binds "
                f"it automatically when it is assigned into pipeline.models, or call "
                f"bind_inner(model) explicitly."
            )
        return getattr(inner, name)

    def _fresh_state(self) -> Dict[str, Any]:
        return hicache_init(
            num_steps=_NO_END_WINDOW,
            interval=self.interval,
            max_order=self.max_order,
            first_enhance=max(1, self.warmup_steps),
            end_enhance=_NO_END_WINDOW,
            sigma=self.sigma,
            backend=self.method,
            history=self.dmd_history,
        )

    def reset(self) -> None:
        if self._state_cond is not None and (self.computed_steps or self.skipped_steps):
            logger.info(
                "[TRELLIS-HiCache] run finished: %d computed + %d skipped DiT steps "
                "(method=%s, interval=%d)",
                self.computed_steps, self.skipped_steps, self.method, self.interval,
            )
        self._state_cond = self._fresh_state()
        self._state_uncond = self._fresh_state()
        self._last_t = None
        self._run_dir = None
        self._tmpl_cond = None
        self._tmpl_uncond = None
        self.computed_steps = 0
        self.skipped_steps = 0

    @staticmethod
    def _timestep_value(timestep: Any) -> float:
        if torch.is_tensor(timestep):
            return float(timestep.reshape(-1)[0].item())
        return float(timestep)

    def _forecast(self, state: Dict[str, Any]) -> torch.Tensor:
        if self.method == "dmd":
            return dmd_forecast_state(state)
        if self.method == "auto":
            return auto_forecast_state(state)
        return hicache_forecast(state)

    def forward(self, latent_model_input: Any, timestep: Any,
                *args: Any, **kwargs: Any) -> Any:
        t_val = self._timestep_value(timestep)
        # Run-boundary + CFG-branch detection, scale-agnostic (works for t in
        # [0,1] or [0,1000], increasing or decreasing). Within a run, distinct
        # timesteps move monotonically; split-CFG repeats each timestep exactly;
        # a new run reverses the direction of travel (t jumps back to its start).
        eps = 1e-6 * (1.0 + abs(t_val))
        if self._last_t is None:
            self.reset()
            is_uncond = False
        elif abs(t_val - self._last_t) <= eps:
            # repeated timestep => the unconditional forward of the same step
            is_uncond = True
        else:
            d = 1 if (t_val - self._last_t) > 0 else -1
            if self._run_dir is not None and d != self._run_dir:
                self.reset()           # direction reversed => new sampling run
            else:
                self._run_dir = d
            is_uncond = False
        self._last_t = t_val

        state = self._state_uncond if is_uncond else self._state_cond

        if hicache_decide(state) == "forecast":
            forecast_feat = self._forecast(state)
            state["step"] += 1
            self.skipped_steps += 1
            tmpl = self._tmpl_uncond if is_uncond else self._tmpl_cond
            if tmpl is not None:
                # SLaT stage: rebuild the SparseTensor from the fixed layout.
                return tmpl.replace(forecast_feat)
            return forecast_feat

        inner = self._inner()
        if inner is None:
            raise RuntimeError(
                "HiCacheModelPatch: a compute step was reached but the wrapped flow "
                "model is not loaded (inner is None). For lazy / GGUF Trellis2 "
                "pipelines, apply HiCache after the model is materialized, or call "
                "bind_inner(model) once it is available."
            )
        out = inner(latent_model_input, timestep, *args, **kwargs)
        if _is_sparse(out):
            anchor = out.feats.detach()
            if is_uncond:
                self._tmpl_uncond = out
            else:
                self._tmpl_cond = out
        else:
            anchor = out.detach()
            if is_uncond:
                self._tmpl_uncond = None
            else:
                self._tmpl_cond = None
        hicache_update_derivatives(state, anchor)
        if self.method in ("dmd", "auto"):
            dmd_update_snapshots(state, anchor, state["history"])
        state["step"] += 1
        self.computed_steps += 1
        return out


class _LazyPatchDict(dict):
    """A ``pipeline.models`` drop-in that keeps patches alive across lazy loads.

    GGUF / lazy Trellis2 pipelines leave a flow-model slot empty (``None``) at
    patch time and assign the real DiT later, from inside the sampler, via
    ``pipeline.models[key] = model``. With a plain dict that assignment would
    overwrite the :class:`HiCacheModelPatch` wrapper and silently disable the
    cache.

    Critically, lazy slots are kept as *real* ``None`` here, never a
    ``HiCacheModelPatch(None)`` placeholder: several lazy loaders (e.g.
    ComfyUI-Trellis2-GGUF's ``load_sparse_structure_model``) gate the actual
    load on ``if self.models[key] is None``, and no wrapper object can ever
    satisfy an ``is None`` identity check. A placeholder there would make the
    loader believe the model is already loaded and skip loading it entirely,
    leaving the DiT never materialized. Pending keys are tracked in
    ``_pending`` instead; the *first* real (non-``None``, non-patch) value
    assigned to a pending key is wrapped fresh in ``HiCacheModelPatch`` here.

    ``_pending`` entries are intentionally never popped, so a key that later
    gets unloaded (reset to ``None``, as the GGUF nodes do for VRAM
    management) and reloaded is wrapped again each time, rather than only
    the first time.

    For a key that's already wrapped (assigned once, still loaded), a later
    real-value assignment instead *binds* into the existing patch, matching
    the eager (non-lazy) pipeline's re-assign behavior.

    It is a ``dict`` subclass, so ``isinstance(models, dict)`` checks still pass.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pending: Dict[str, Dict[str, Any]] = {}

    def _is_real_model(self, value: Any) -> bool:
        return (value is not None
                and not getattr(value, "_hicache_is_patch", False)
                and not isinstance(value, HiCacheModelPatch))

    def __setitem__(self, key, value):
        existing = self.get(key)
        if isinstance(existing, HiCacheModelPatch) and self._is_real_model(value):
            existing.bind_inner(value)   # materialize lazy model into the patch
            return
        if key in self._pending and self._is_real_model(value):
            super().__setitem__(key, HiCacheModelPatch(value, **self._pending[key]))
            return
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):  # route bulk updates through __setitem__
        for k, v in dict(*args, **kwargs).items():
            self[k] = v


def _resolve_keys(pipeline: Any, stages: str) -> List[str]:
    if stages not in STAGES:
        raise ValueError(f"stages must be one of {STAGES}, got {stages!r}")
    if not hasattr(pipeline, "models") or not isinstance(pipeline.models, dict):
        raise TypeError(
            "TRELLIS-HiCache: pipeline has no `.models` dict - expected a "
            f"TrellisImageTo3DPipeline, got {type(pipeline).__name__}"
        )
    keys = [k for k in _STAGE_KEYS[stages] if k in pipeline.models]
    if not keys:
        raise TypeError(
            f"TRELLIS-HiCache: none of {_STAGE_KEYS[stages]} found in pipeline.models "
            f"(have {sorted(pipeline.models)})"
        )
    return keys


def apply_hicache(pipeline: Any, *, method: str = "hermite", interval: int = 3,
                  warmup_steps: int = 2, max_order: int = 1, sigma: float = 0.5,
                  dmd_history: int = 5, stages: str = "both") -> Any:
    """Return a shallow copy of ``pipeline`` whose selected flow DiTs are patched.

    The input pipeline is NOT mutated (copy-on-patch). Weights are shared; only
    the wrapper objects and the ``models`` dict differ. Re-patching an already
    patched pipeline replaces the patch (never nests).
    """
    keys = _resolve_keys(pipeline, stages)
    patched = copy.copy(pipeline)
    # _LazyPatchDict (a dict subclass) so a later pipeline.models[key] = model
    # from a lazy/GGUF sampler binds into the patch instead of overwriting it.
    patched.models = _LazyPatchDict(pipeline.models)  # copy so the original is untouched
    patch_kwargs = dict(method=method, interval=interval, warmup_steps=warmup_steps,
                         max_order=max_order, sigma=sigma, dmd_history=dmd_history)
    lazy_keys = []
    for key in keys:
        inner = patched.models[key]
        if getattr(inner, "_hicache_is_patch", False):
            inner = inner.inner  # replace, never nest
            dict.__setitem__(patched.models, key, inner)
        if inner is None:
            # Leave the slot as real None (not a placeholder patch) so the
            # pipeline's own `is None` lazy-load gate still fires; wrap it in
            # _LazyPatchDict.__setitem__ the first (and every) time the
            # pipeline assigns the real model.
            lazy_keys.append(key)
            patched.models._pending[key] = dict(patch_kwargs)
            continue
        dict.__setitem__(patched.models, key, HiCacheModelPatch(inner, **patch_kwargs))
    if lazy_keys:
        logger.warning(
            "[TRELLIS-HiCache] %s not loaded yet (lazy/GGUF pipeline); will patch "
            "%s on load -- deferred rather than wrapped now, so the pipeline's own "
            "`is None` lazy-load check still fires.",
            type(pipeline).__name__, lazy_keys,
        )
    logger.info(
        "[TRELLIS-HiCache] patched %s on %s: method=%s interval=%d warmup=%d",
        type(pipeline).__name__, keys, method, interval, warmup_steps,
    )
    return patched


def remove_hicache(pipeline: Any) -> Any:
    """Return ``pipeline`` with the original DiTs restored (copy-on-unpatch)."""
    if not hasattr(pipeline, "models") or not isinstance(pipeline.models, dict):
        return pipeline
    has_patched = any(getattr(m, "_hicache_is_patch", False) for m in pipeline.models.values())
    # A lazy key that hasn't loaded yet has no HiCacheModelPatch to find above
    # (its slot is real None, deliberately -- see _LazyPatchDict), but it still
    # has pending patch config that must be dropped, or a load that happens
    # after this "removal" would silently get wrapped again.
    has_pending = bool(getattr(pipeline.models, "_pending", None))
    if not has_patched and not has_pending:
        return pipeline
    clean = copy.copy(pipeline)
    clean.models = dict(pipeline.models)  # plain dict: also drops _pending
    for key, m in list(clean.models.items()):
        if getattr(m, "_hicache_is_patch", False):
            clean.models[key] = m.inner
    logger.info("[TRELLIS-HiCache] removed patch from %s", type(pipeline).__name__)
    return clean
