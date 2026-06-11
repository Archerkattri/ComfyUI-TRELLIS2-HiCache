"""ComfyUI node: training-free HiCache / HiCache++ acceleration for TRELLIS.2.

One node, no ComfyUI-internal imports: it patches the TRELLIS.2 flow DiTs
(sparse-structure + shape-SLaT, optionally texture-SLaT) so skipped sampling
steps are forecast with hicache-pp instead of running the transformer. Drop it
between the TRELLIS.2 loader (``TRELLIS2PIPELINE``) and the sampler.

Targets the visualbruno/ComfyUI-Trellis2 wrapper, whose ``TRELLIS2PIPELINE`` is a
live ``Trellis2ImageTo3DPipeline`` with the ``.models`` dict this patch needs.

All acceleration logic lives in :mod:`trellis_hicache_patch` (unit-testable with
no ComfyUI / GPU); this file is only the ComfyUI plumbing.
"""
try:
    from .trellis_hicache_patch import METHODS, STAGES, apply_hicache, remove_hicache
except ImportError:  # running as a flat module (tests / standalone)
    from trellis_hicache_patch import METHODS, STAGES, apply_hicache, remove_hicache


class Trellis2HiCacheAccelerate:
    """Patch a TRELLIS.2 pipeline's flow DiTs with a HiCache velocity forecast.

    Connect a ``TRELLIS2PIPELINE`` from the TRELLIS.2 loader to ``model`` and feed
    the returned pipeline to the sampler. ``enabled = Off`` removes the patch and
    restores the stock DiTs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("TRELLIS2PIPELINE",),
                "method": (list(METHODS), {"default": "hermite"}),
                "interval": ("INT", {"default": 2, "min": 1, "max": 10,
                                     "tooltip": "Run the DiT once every `interval` "
                                     "steps; the rest are forecast. 2 is "
                                     "near-lossless (~2x, ~0.01 Chamfer)."}),
                "stages": (list(STAGES), {"default": "both",
                           "tooltip": "both = sparse-structure + shape SLaT "
                           "(shape generation); all also accelerates texture "
                           "synthesis; or pick a single stage."}),
                "warmup_steps": ("INT", {"default": 2, "min": 0, "max": 10,
                                 "tooltip": "Always compute the first N steps of "
                                 "each run before forecasting begins."}),
                "enabled": ("BOOLEAN", {"default": True,
                            "tooltip": "Off removes the patch and restores the "
                            "stock TRELLIS.2 DiTs."}),
            },
            "optional": {
                "sigma": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.05}),
                "dmd_history": ("INT", {"default": 5, "min": 3, "max": 16}),
                "max_order": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
        }

    RETURN_TYPES = ("TRELLIS2PIPELINE",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "TRELLIS2/HiCache"
    DESCRIPTION = ("Training-free acceleration for TRELLIS.2 image-to-3D: forecast "
                   "the flow-matching velocity on skipped DiT steps (HiCache "
                   "Hermite / HiCache++ DMD, via the hicache-pp library).")

    def patch(self, model, method="hermite", interval=2, stages="both",
              warmup_steps=2, enabled=True, sigma=0.5, dmd_history=5, max_order=1):
        if not enabled:
            return (remove_hicache(model),)
        return (apply_hicache(
            model, method=method, interval=interval, warmup_steps=warmup_steps,
            max_order=max_order, sigma=sigma, dmd_history=dmd_history, stages=stages,
        ),)


NODE_CLASS_MAPPINGS = {"Trellis2HiCacheAccelerate": Trellis2HiCacheAccelerate}
NODE_DISPLAY_NAME_MAPPINGS = {"Trellis2HiCacheAccelerate": "TRELLIS.2 HiCache Accelerate"}
