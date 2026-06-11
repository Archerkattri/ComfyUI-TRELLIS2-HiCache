"""GPU validation for the TRELLIS.2 HiCache patch.

Loads a real stock TRELLIS.2 pipeline, applies the model-level forecast patch to
the sparse-structure + shape-SLaT flow DiTs, runs image-to-3D, and confirms it
runs, skips DiT forwards, is faster, and produces near-identical mesh geometry.

Run from a hermit-trellis2 checkout (which vendors the trellis2 package):

    cd third_party/hermit-trellis2
    PYTHONPATH=.:../comfyui-trellis2-hicache python ../comfyui-trellis2-hicache/tests/validate_gpu.py \
        --ckpt /home/krishi/workspace/data/weights/TRELLIS.2-4B --image <some.png>
"""
import argparse, time, sys, types


def _stub_render_deps():
    for m in ("nvdiffrast", "nvdiffrast.torch", "nvdiffrec"):
        sys.modules.setdefault(m, types.ModuleType(m))
    sys.modules["nvdiffrast"].torch = sys.modules["nvdiffrast.torch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--method", default="hermite")
    ap.add_argument("--interval", type=int, default=2)
    ap.add_argument("--stages", default="both")
    ap.add_argument("--pipeline_type", default="shape")
    args = ap.parse_args()

    _stub_render_deps()
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis_hicache_patch import apply_hicache, remove_hicache

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(args.ckpt)
    pipe.to("cuda")
    pipe.enable_faster("base")   # stock samplers; we test the model-level patch
    img = Image.open(args.image).convert("RGBA")
    kw = dict(seed=0, preprocess_image=True, pipeline_type=args.pipeline_type)

    def verts(mesh):
        return mesh.vertices.detach().float()

    torch.cuda.synchronize(); t0 = time.time()
    m_stock = pipe.run(img, **kw)[0]
    torch.cuda.synchronize(); t_stock = time.time() - t0
    v_stock = verts(m_stock)

    patched = apply_hicache(pipe, method=args.method, interval=args.interval, stages=args.stages)
    torch.cuda.synchronize(); t0 = time.time()
    m_fast = patched.run(img, **kw)[0]
    torch.cuda.synchronize(); t_fast = time.time() - t0
    v_fast = verts(m_fast)

    def chamfer(a, b, k=20000):
        if a.shape[0] > k: a = a[torch.randperm(a.shape[0], device=a.device)[:k]]
        if b.shape[0] > k: b = b[torch.randperm(b.shape[0], device=b.device)[:k]]
        return float((torch.cdist(a, b).min(1).values.mean()
                      + torch.cdist(b, a).min(1).values.mean()) / 2)
    cham = chamfer(v_stock, v_fast)

    total_skipped = 0
    print("\n==================== TRELLIS.2-HiCache validation ====================")
    print(f"method={args.method} interval={args.interval} stages={args.stages}")
    for key, m in patched.models.items():
        if getattr(m, "_hicache_is_patch", False):
            c, s = m.computed_steps, m.skipped_steps
            total_skipped += s
            if c or s:
                print(f"  {key}: {c} computed + {s} skipped")
    print(f"stock : {t_stock:.2f}s  ({v_stock.shape[0]} verts)")
    print(f"hicache: {t_fast:.2f}s  ({v_fast.shape[0]} verts)")
    print(f"speedup: {t_stock / max(t_fast, 1e-6):.2f}x")
    print(f"chamfer(stock,hicache) = {cham:.5f}")

    restored = remove_hicache(patched)
    clean = not any(getattr(m, "_hicache_is_patch", False) for m in restored.models.values())
    ok = total_skipped > 0 and t_fast < t_stock and cham < 0.02 and clean
    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
