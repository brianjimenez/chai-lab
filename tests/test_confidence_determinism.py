"""
Tests for the deterministic mode of the score command.
"""

import os

import pytest
import torch

from chai_lab.confidence import make_deterministic


@pytest.fixture
def restore_torch_settings(monkeypatch):
    """make_deterministic changes process-wide settings: restore them afterwards."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    tf32 = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    can_fuse_on_gpu = torch._C._jit_can_fuse_on_gpu()
    texpr_fuser = torch._C._jit_texpr_fuser_enabled()
    yield
    # The setters return the previous value
    profiling_executor = torch._C._jit_set_profiling_executor(True)
    profiling_mode = torch._C._jit_set_profiling_mode(True)
    assert not profiling_executor and not profiling_mode
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    torch._C._jit_override_can_fuse_on_gpu(can_fuse_on_gpu)
    torch._C._jit_set_texpr_fuser_enabled(texpr_fuser)


def test_make_deterministic(restore_torch_settings):
    make_deterministic()

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert torch.are_deterministic_algorithms_enabled()
    # Operations without a deterministic implementation warn instead of failing
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert not torch._C._jit_can_fuse_on_gpu()
    assert not torch._C._jit_texpr_fuser_enabled()


def test_make_deterministic_keeps_cublas_config(restore_torch_settings, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    make_deterministic()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
