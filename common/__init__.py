"""
Shared building blocks for the teacher and student training stages.

IMPORT ORDER WARNING -- READ THIS BEFORE ADDING IMPORTS
    On this machine the chain

        import torch, torchvision, timm, onnxruntime, onnx, sklearn

    SEGFAULTS. Not an exception -- a hard crash with no Python traceback, which makes it
    look like the script "just stopped". Each library imports fine on its own, and most
    pairs are fine; it is the full chain that dies.

    Verified workaround: import scikit-learn (and scipy) BEFORE torch.

        import sklearn, scipy          # first
        import torch, torchvision      # then

    Every module here that needs both follows that order, and common.safe_imports exists
    so callers do not have to remember.
"""
