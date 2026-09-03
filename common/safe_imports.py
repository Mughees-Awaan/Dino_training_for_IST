#!/usr/bin/env python3
"""
safe_imports.py -- import heavy libraries in an order that does not crash.

WHY THIS FILE EXISTS
    Measured on this machine (conda env "weed"):

        python -c "import torch,torchvision,timm,onnxruntime,onnx,sklearn"
        Segmentation fault (core dumped)

    A segfault is not a Python exception. There is no traceback, no error message, and
    nothing to catch -- the process simply dies. If that happened three hours into a
    training run it would look like the machine had failed rather than the imports.

    Each library is fine alone. The crash needs the whole chain, and it goes away when
    scikit-learn and scipy are imported BEFORE torch.

USE
    from training.common.safe_imports import torch, np      # instead of `import torch`
"""

from __future__ import annotations

# ---- the fix: these two go first, always -------------------------------------------
# Importing them here means that ANY module which imports from this file gets the safe
# order for free, whatever order the caller happens to use.
import scipy  # noqa: F401
import sklearn  # noqa: F401

import numpy as np  # noqa: F401

try:
    import torch  # noqa: F401
    import torch.nn as nn  # noqa: F401
    import torch.nn.functional as F  # noqa: F401
    HAVE_TORCH = True
except ImportError:                                    # pragma: no cover
    torch = nn = F = None
    HAVE_TORCH = False


def require_torch(what: str = "this module"):
    """Fail with a useful message instead of an AttributeError on `torch.None`."""
    if not HAVE_TORCH:
        raise ImportError(
            f"{what} needs PyTorch, which is not installed in this environment.\n"
            f"The preprocessing stage (training/data, training/runtime) deliberately does "
            f"NOT need it -- that is why the data logic can be tested with no GPU."
        )
    return torch
