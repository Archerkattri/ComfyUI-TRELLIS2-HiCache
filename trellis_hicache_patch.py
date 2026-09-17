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
    hicache_telemetry,
    hicache_update_derivatives,
    hicache_forecast,
    dmd_update_snapshots,
    dmd_forecast_state,
    auto_forecast_state,
    CacheBudget,
    CacheBudgetRuntime,
    RunIdentity,
    stable_digest,
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
    min_history = {"hermite": 3, "dmd": 4, "auto": 5}[method]
    if dmd_history < min_history:
        raise ValueError(
            f"dmd_history must be >= {min_history} for {method}, got {dmd_history}"
        )


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
                 sigma: float = 0.5, dmd_history: int = 5,
                 stage_id: str = "unknown",
                 budget: Optional[CacheBudget] = None,
                 max_horizon: Optional[int] = None,
                 max_memory_mb: Optional[float] = None,
                 audit_budget: int = 0) -> None:
        validate_config(method, interval, warmup_steps, max_order, sigma, dmd_history)
        super().__init__()
        # ``model`` may be None here: GGUF / lazy Trellis2 pipelines
        # (e.g. Aero-Ex/ComfyUI-Trellis2-GGUF) leave pipeline.models[key] empty at
        # patch time and materialize the flow DiT later, inside the sampler. Store
        # the wrapped model so it survives both cases -- a real nn.Module is
        # registered as a submodule (so .to()/.state_dict()/.parameters() recurse),
        # while a None/lazy deferred module is kept as a plain attribute. See _set_inner.
        self._set_inner(model)
        self._hicache_is_patch = True
        self.method = method
        self.interval = int(interval)
        self.warmup_steps = int(warmup_steps)
        self.max_order = int(max_order)
        self.sigma = float(sigma)
        self.dmd_history = int(dmd_history)
        self.stage_id = str(stage_id)
        if budget is None:
            budget = CacheBudget(
                backend=method,
                allowed_stages=(self.stage_id,),
                max_horizon=max(1, interval - 1) if max_horizon is None else max_horizon,
                quality_preset="adapter-default",
                max_memory_mb=max_memory_mb,
                audit_budget=audit_budget,
                fallback="full",
            )
        elif not isinstance(budget, CacheBudget):
            raise TypeError("budget must be a CacheBudget")
        self.budget = budget

        # two parallel forecast states: conditional + unconditional CFG branches
        self._state_cond: Optional[Dict[str, Any]] = None
        self._state_uncond: Optional[Dict[str, Any]] = None
        self._tmpl_cond: Any = None     # last computed SparseTensor (cond branch)
        self._tmpl_uncond: Any = None   # last computed SparseTensor (uncond branch)
        self._last_t: Optional[float] = None
        self._run_dir: Optional[int] = None   # sign of t-progression within a run
        self._last_branch_id: Optional[str] = None
        self._last_telemetry: Dict[str, Any] = {}
        self._last_budget_manifest: Dict[str, Any] = {}
        self._budget_runtimes: Dict[str, CacheBudgetRuntime] = {}
        self._last_budget_decision: Dict[str, Any] = {}
        self.last_decision: Optional[str] = None
        self.computed_steps = 0
        self.skipped_steps = 0

    @property
    def run_id(self) -> Optional[str]:
        """Shared identity of the currently active sampling run."""
        return (None if self._state_cond is None
                else self._state_cond.get("run_id"))

    @property
    def branch_id(self) -> Optional[str]:
        """The conditional/unconditional branch used by the latest forward."""
        return self._last_branch_id

    @property
    def cfg_mode(self) -> str:
        """TRELLIS.2 uses split conditional/unconditional model forwards."""
        return "split_or_single"

    @staticmethod
    def _empty_telemetry() -> Dict[str, Any]:
        return {
            "decisions": {"full": 0, "forecast": 0},
            "method_counts": {"hermite": 0, "dmd": 0, "reuse": 0},
            "fallbacks": {},
        }

    def _telemetry_snapshot(self) -> Dict[str, Any]:
        """Aggregate detached central telemetry while retaining branch detail."""
        result = self._empty_telemetry()
        branches: Dict[str, Dict[str, Any]] = {}
        for branch, state in (("cond", self._state_cond),
                              ("uncond", self._state_uncond)):
            if state is None:
                continue
            current = hicache_telemetry(state)
            branches[branch] = current
            for key in ("decisions", "method_counts"):
                for name, count in current[key].items():
                    result[key][name] = result[key].get(name, 0) + int(count)
            for reason, count in current["fallbacks"].items():
                result["fallbacks"][reason] = (
                    result["fallbacks"].get(reason, 0) + int(count))
        result.update({
            "run_id": self.run_id,
            "stage_id": self.stage_id,
            "cfg_mode": self.cfg_mode,
            "branch_id": self.branch_id,
            "branches": branches,
        })
        return result

    @property
    def telemetry(self) -> Dict[str, Any]:
        """Detached actual-method/fallback counters with branch detail."""
        if self._state_cond is not None:
            return self._telemetry_snapshot()
        return copy.deepcopy(self._last_telemetry)

    @property
    def budget_manifest(self) -> Dict[str, Any]:
        """Aggregate branch manifests without exposing tensors or private inputs."""
        if self._budget_runtimes:
            branches = {
                branch: runtime.manifest.as_dict()
                for branch, runtime in self._budget_runtimes.items()
            }
            counts = {"full": 0, "forecast": 0, "fallback": 0}
            for report in branches.values():
                for mode in counts:
                    counts[mode] += int(report["counts"].get(mode, 0))
            return {
                "schema": "hicache-pp.trellis2-budget-manifest.v1",
                "counts": counts,
                "branches": branches,
            }
        return copy.deepcopy(self._last_budget_manifest)

    @property
    def budget_decision(self) -> Dict[str, Any]:
        return copy.deepcopy(self._last_budget_decision)

    def save_budget_manifest(self, destination: str) -> None:
        import json
        from pathlib import Path

        report = self.budget_manifest
        if not report:
            raise RuntimeError("no budget manifest exists; run the patched model first")
        Path(destination).write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    def _set_inner(self, model: Optional[torch.nn.Module]) -> None:
        """Store the wrapped model so ``self.inner`` is always resolvable.

        A real ``nn.Module`` is registered in ``_modules`` (so device moves and
        ``state_dict`` recurse into it, exactly as the eager path relied on). A
        ``None`` deferred value -- or any non-Module -- is kept as a plain attribute
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
        deferred binding, and register it as a submodule. Lets GGUF / lazy pipelines
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

    def _fresh_state(self, branch_id: str) -> Dict[str, Any]:
        state = hicache_init(
            num_steps=_NO_END_WINDOW,
            interval=self.interval,
            max_order=self.max_order,
            first_enhance=max(1, self.warmup_steps),
            end_enhance=_NO_END_WINDOW,
            sigma=self.sigma,
            backend=self.method,
            history=self.dmd_history,
        )
        state["branch_id"] = str(branch_id)
        state["stage_id"] = self.stage_id
        return state

    def reset(self, run_id: Optional[str] = None) -> None:
        if self._state_cond is not None and (self.computed_steps or self.skipped_steps):
            self._last_telemetry = self._telemetry_snapshot()
            logger.info(
                "[TRELLIS-HiCache] run finished: %d computed + %d skipped DiT steps "
                "(method=%s, interval=%d)",
                self.computed_steps, self.skipped_steps, self.method, self.interval,
            )
        if self._budget_runtimes:
            self._last_budget_manifest = self.budget_manifest
        self._budget_runtimes = {}
        self._last_budget_decision = {}
        self._state_cond = self._fresh_state("cond")
        self._state_uncond = self._fresh_state("uncond")
        if run_id is not None:
            self._state_cond["run_id"] = str(run_id)
        self._state_uncond["run_id"] = self._state_cond["run_id"]
        self._last_t = None
        self._run_dir = None
        self._last_branch_id = None
        self._tmpl_cond = None
        self._tmpl_uncond = None
        self.last_decision = None
        self.computed_steps = 0
        self.skipped_steps = 0

    @staticmethod
    def _timestep_value(timestep: Any) -> float:
        if torch.is_tensor(timestep):
            return float(timestep.reshape(-1)[0].item())
        return float(timestep)

    @staticmethod
    def _is_uncond_branch(branch_id: Any) -> bool:
        """Normalize explicit branch markers used by integrations."""
        if isinstance(branch_id, bool):
            return branch_id
        return str(branch_id).strip().lower() in {
            "uncond", "unconditional", "negative", "negative_cond", "uc",
        }

    def _forecast(self, state: Dict[str, Any], method: Optional[str] = None) -> torch.Tensor:
        method = self.method if method is None else method
        if method == "dmd":
            return dmd_forecast_state(state)
        if method == "auto":
            return auto_forecast_state(state)
        return hicache_forecast(state)

    def _ensure_budget_runtime(
        self, state: Dict[str, Any], latent_model_input: Any, branch: str
    ) -> CacheBudgetRuntime:
        features = getattr(latent_model_input, "feats", latent_model_input)
        shape = tuple(int(value) for value in features.shape) if torch.is_tensor(features) else ()
        dtype = str(features.dtype) if torch.is_tensor(features) else "unknown"
        device = str(features.device) if torch.is_tensor(features) else "unknown"
        inner = self._inner()
        identity = RunIdentity(
            model_id=type(inner).__name__ if inner is not None else "unloaded-model",
            run_id=str(state["run_id"]),
            cfg_branch=branch,
            conditioning_id=branch,
            stage=self.stage_id,
            token_layout_digest=stable_digest({"shape": shape, "stage": self.stage_id}),
            dtype=dtype,
            device=device,
            batch_id=stable_digest({"shape": shape, "dtype": dtype, "device": device}),
        )
        runtime = self._budget_runtimes.get(branch)
        if runtime is None or runtime.identity.fingerprint != identity.fingerprint:
            runtime = CacheBudgetRuntime(
                self.budget,
                identity,
                source_digest=stable_digest({"adapter": "ComfyUI-TRELLIS2-HiCache", "contract": "budget-v1"}),
                config_digest=self.budget.digest,
            )
            self._budget_runtimes[branch] = runtime
        return runtime

    @staticmethod
    def _current_memory_mb(value: Any) -> Optional[float]:
        features = getattr(value, "feats", value)
        if not torch.is_tensor(features) or not features.is_cuda:
            return None
        return float(torch.cuda.memory_allocated(features.device)) / (1024.0 * 1024.0)

    def forward(self, latent_model_input: Any, timestep: Any,
                *args: Any, **kwargs: Any) -> Any:
        # Stock TRELLIS.2 does not pass these markers, so the timestep heuristic
        # below remains the compatibility path. Integrations that can identify
        # a retry/job explicitly should pass them; they are consumed here and
        # never forwarded to the wrapped DiT.
        explicit_run_id = kwargs.pop("hicache_run_id", kwargs.pop("run_id", None))
        explicit_branch_id = kwargs.pop(
            "hicache_branch_id", kwargs.pop("branch_id", None))
        t_val = self._timestep_value(timestep)
        # Run-boundary + CFG-branch detection, scale-agnostic (works for t in
        # [0,1] or [0,1000], increasing or decreasing). Within a run, distinct
        # timesteps move monotonically; split-CFG repeats each timestep exactly;
        # a new run reverses the direction of travel (t jumps back to its start).
        eps = 1e-6 * (1.0 + abs(t_val))
        if self._last_t is None:
            self.reset(explicit_run_id)
            is_uncond = self._is_uncond_branch(explicit_branch_id)
        elif explicit_run_id is not None and str(explicit_run_id) != self.run_id:
            self.reset(explicit_run_id)
            is_uncond = self._is_uncond_branch(explicit_branch_id)
        elif abs(t_val - self._last_t) <= eps:
            # repeated timestep => the unconditional forward of the same step
            is_uncond = (self._is_uncond_branch(explicit_branch_id)
                         if explicit_branch_id is not None else True)
        else:
            d = 1 if (t_val - self._last_t) > 0 else -1
            if self._run_dir is not None and d != self._run_dir:
                self.reset()           # direction reversed => new sampling run
            else:
                self._run_dir = d
            is_uncond = self._is_uncond_branch(explicit_branch_id)
        self._last_t = t_val
        self._last_branch_id = "uncond" if is_uncond else "cond"

        state = self._state_uncond if is_uncond else self._state_cond

        decision = hicache_decide(state)
        self.last_decision = decision
        branch = "uncond" if is_uncond else "cond"
        budget_runtime = self._ensure_budget_runtime(state, latent_model_input, branch)
        budget_decision = budget_runtime.decide(
            self.stage_id,
            horizon=int(state.get("counter", 0)) if decision == "forecast" else 0,
            method=self.method,
            supported=self._inner() is not None,
            memory_mb=self._current_memory_mb(latent_model_input),
            controller_selected=self.method == "auto",
        )
        self._last_budget_decision = budget_decision.as_dict()
        if budget_decision.mode == "fallback" and budget_decision.method == "full" and decision == "forecast":
            state["type"] = "full"
            state["counter"] = 0
            state["activated_steps"].append(state["step"])
            telemetry = state["telemetry"]["decisions"]
            telemetry["forecast"] = max(0, int(telemetry.get("forecast", 0)) - 1)
            telemetry["full"] = int(telemetry.get("full", 0)) + 1
            decision = "full"
        if decision == "forecast" and budget_decision.mode in ("forecast", "fallback"):
            forecast_feat = self._forecast(state, budget_decision.method)
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
    """A ``pipeline.models`` drop-in that patches models after lazy loads.

    GGUF / lazy Trellis2 pipelines leave a flow-model slot empty (``None``) at
    patch time and assign the real DiT later, from inside the sampler, via
    ``pipeline.models[key] = model``. The slot must remain the real ``None``
    singleton until then because loaders commonly guard the assignment with
    ``if pipeline.models[key] is None``. Pending patch configuration therefore
    lives out-of-band in ``_pending``; assigning a real model wraps it at that
    boundary. Pending entries survive assignment back to ``None`` so VRAM
    unload/reload cycles receive a fresh patch each time.
    """

    def __init__(self, *args, **kwargs):
        source = args[0] if args else None
        super().__init__(*args, **kwargs)
        inherited = getattr(source, "_pending", {})
        self._pending = {key: dict(config) for key, config in inherited.items()}

    def set_pending(self, key: str, config: Dict[str, Any]) -> None:
        self._pending[key] = dict(config)
        dict.__setitem__(self, key, None)

    def __setitem__(self, key, value):
        if key in self._pending:
            if value is None:
                dict.__setitem__(self, key, None)
                return
            if not getattr(value, "_hicache_is_patch", False):
                value = HiCacheModelPatch(value, **self._pending[key])
            dict.__setitem__(self, key, value)
            return

        existing = self.get(key)
        if (isinstance(existing, HiCacheModelPatch)
                and value is not None
                and not getattr(value, "_hicache_is_patch", False)
                and not isinstance(value, HiCacheModelPatch)):
            existing.bind_inner(value)   # materialize lazy model into the patch
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
                  dmd_history: int = 5, stages: str = "both",
                  budget: Optional[CacheBudget] = None,
                  max_horizon: Optional[int] = None,
                  max_memory_mb: Optional[float] = None,
                  audit_budget: int = 0) -> Any:
    """Return a shallow copy of ``pipeline`` whose selected flow DiTs are patched.

    The input pipeline is NOT mutated (copy-on-patch). Weights are shared; only
    the wrapper objects and the ``models`` dict differ. Re-patching an already
    patched pipeline replaces the patch (never nests).
    """
    keys = _resolve_keys(pipeline, stages)
    patched = copy.copy(pipeline)
    # A dict subclass keeps real None sentinels visible to lazy/GGUF loader
    # guards, then wraps the model at the assignment boundary.
    patched.models = _LazyPatchDict(pipeline.models)  # copy so the original is untouched
    lazy_keys = []
    for key in keys:
        inner = patched.models[key]
        if getattr(inner, "_hicache_is_patch", False):
            inner = inner.inner  # replace, never nest
        config = {
            "method": method,
            "interval": interval,
            "warmup_steps": warmup_steps,
            "max_order": max_order,
            "sigma": sigma,
            "dmd_history": dmd_history,
            "stage_id": key,
            "budget": budget,
            "max_horizon": max_horizon,
            "max_memory_mb": max_memory_mb,
            "audit_budget": audit_budget,
        }
        if inner is None:
            lazy_keys.append(key)
            patched.models.set_pending(key, config)
        else:
            if key in patched.models._pending:
                patched.models._pending[key] = dict(config)
            dict.__setitem__(patched.models, key, HiCacheModelPatch(inner, **config))
    if lazy_keys:
        logger.warning(
            "[TRELLIS-HiCache] %s not loaded yet (lazy/GGUF pipeline); queued a "
            "deferred patch on %s -- assignment wraps the materialized model.",
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
    has_patch = any(
        getattr(m, "_hicache_is_patch", False) for m in pipeline.models.values()
    )
    has_pending = bool(getattr(pipeline.models, "_pending", {}))
    if not has_patch and not has_pending:
        return pipeline
    clean = copy.copy(pipeline)
    clean.models = dict(pipeline.models)
    for key, m in list(clean.models.items()):
        if getattr(m, "_hicache_is_patch", False):
            clean.models[key] = m.inner
    logger.info("[TRELLIS-HiCache] removed patch from %s", type(pipeline).__name__)
    return clean
