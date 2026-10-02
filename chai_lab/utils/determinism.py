# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import os

import torch


def make_deterministic():
    """
    Makes results reproducible on GPU. Otherwise the same input can give
    different results from run to run (seen: the same structure scoring from 0.43
    to 0.59 aggregate score; the same fold seed giving structures several Å apart),
    from two sources that seeding does not control:
    - non-deterministic CUDA kernels;
    - the TorchScript executor of Chai's exported components, which switches to
      fused kernels after the first calls, so the first structure scored in a
      process gets different numbers than the next ones.
    Affects the whole process; call it before any CUDA work.
    """
    # Needed by deterministic cuBLAS; only read when cuBLAS is initialized
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch._C._jit_set_profiling_executor(False)
    torch._C._jit_set_profiling_mode(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
