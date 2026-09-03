#!/usr/bin/env python3
"""
cache.py -- precompute frozen teacher descriptors for the student to imitate.

WHY CACHE
    The student learns by copying the teacher's descriptors. If the teacher runs live
    during student training, every batch pays for a full ViT forward pass -- and the
    teacher is FROZEN, so it produces the identical answer every time. That is the same
    expensive computation, repeated for nothing.

    Compute once, store, read back. Student training then becomes memory-bound rather
    than compute-bound and runs several times faster.

WHY FLOAT16
    Descriptors are compared by direction, not magnitude, so half precision costs
    essentially nothing in accuracy and halves the disk and memory. On this corpus that is
    the difference between fitting in RAM and not.
"""

from __future__ import annotations

import argparse
import os

from training.common.safe_imports import np, require_torch, torch


def cache_split(backbone, dataset, out_path: str, batch: int = 8, device: str = "cuda"):
    """Run the frozen teacher over a split and store its descriptors as fp16."""
    require_torch("cache_split")
    backbone.eval().to(device)
    keys, feats = [], []
    with torch.no_grad():                     # no gradients: this is inference only
        for i in range(0, len(dataset), batch):
            items = [dataset[j] for j in range(i, min(i + batch, len(dataset)))]
            x = torch.from_numpy(np.stack([it["support_image"] for it in items])).to(device)
            g = backbone(x).cpu().numpy().astype(np.float16)
            feats.append(g)
            keys.extend(it["episode_id"] for it in items)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, keys=np.array(keys), feats=np.concatenate(feats))
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", default="tables/v2-staging/episodes/train.parquet")
    ap.add_argument("--manifest", default="tables/manifest.parquet")
    ap.add_argument("--out", default="cache/teacher_train.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from training.common.checkpoint import load
    from training.common.gpu_probe import assert_gpu
    from training.data.dataset import EpisodeDataset
    from training.teacher.dinov3 import DinoV3Backbone

    assert_gpu("teacher cache")
    net = DinoV3Backbone(pretrained=False, frozen=True)
    load(args.checkpoint, model=net, strict_provenance=False)
    ds = EpisodeDataset(args.episodes, args.manifest, augment=False)
    print(f"caching {len(ds):,} episodes -> {cache_split(net, ds, args.out, device=args.device)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
