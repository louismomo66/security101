"""VEREC crime action training pipeline.

Stages:
    1. training.extract_poses     annotated video spans -> skeleton .npz clips
    2. training.train             fine-tune ST-GCN from the NTU60 backbone
    3. training.evaluate          per-class metrics on a held-out split

See training/README.md for the full workflow.

Note: deliberately no `from __future__ import annotations` here. That statement
binds the name `annotations` in this package's namespace, which shadows the
`training.annotations` submodule and makes `from training import annotations`
return a `__future__._Feature` object instead of the module.
"""

__all__ = ["annotations", "dataset", "labels", "metrics"]
