"""
Tests for keeping the models resident on the GPU in the score command.
No model or GPU needed: the loaders are replaced by counting fakes.
"""

from contextlib import contextmanager

import pytest
import torch

import chai_lab.chai1 as chai1
import chai_lab.data.dataset.embeddings.esm as esm
import chai_lab.data.dataset.inference_dataset as inference_dataset
from chai_lab.confidence import keep_models_resident

DEVICE = torch.device("cpu")


@pytest.fixture
def fakes(monkeypatch):
    """Replaces the loaders by counting fakes; restores everything afterwards."""
    calls = {"components": [], "esm": 0, "conformers": 0}

    def load_exported(comp_key, device):
        calls["components"].append(comp_key)
        return object()

    class FakeConformerGenerator:
        def __init__(self):
            calls["conformers"] += 1

    @contextmanager
    def esm_model(device):
        if not esm._esm_model:
            calls["esm"] += 1
            esm._esm_model.append(object())
        [model] = esm._esm_model
        yield model
        calls["esm_offloaded"] = calls.get("esm_offloaded", 0) + 1

    monkeypatch.setattr(chai1, "load_exported", load_exported)
    monkeypatch.setattr(chai1, "_component_cache", {})
    monkeypatch.setattr(chai1, "_component_moved_to", chai1._component_moved_to)
    monkeypatch.setattr(esm, "_esm_model", [])
    monkeypatch.setattr(esm, "esm_model", esm_model)
    monkeypatch.setattr(
        inference_dataset, "RefConformerGenerator", FakeConformerGenerator
    )
    return calls


def test_components_loaded_once_and_shared(fakes):
    keep_models_resident()

    with chai1._component_moved_to("trunk.pt", DEVICE) as first:
        pass
    with chai1._component_moved_to("trunk.pt", DEVICE) as second:
        pass
    with chai1._component_moved_to("token_embedder.pt", DEVICE) as other:
        pass

    assert first is second
    assert other is not first
    assert fakes["components"] == ["trunk.pt", "token_embedder.pt"]


def test_component_stays_on_device(fakes):
    """Unlike the default, the component is not moved to the CPU on exit."""
    keep_models_resident()

    class Module:
        moves = []

        def to(self, device):
            self.moves.append(device)

    module = Module()
    chai1._component_cache["trunk.pt"] = module
    with chai1._component_moved_to("trunk.pt", DEVICE) as component:
        assert component is module
    assert Module.moves == []


def test_esm_loaded_once_and_not_offloaded(fakes):
    keep_models_resident()

    with esm.esm_model(DEVICE) as first:
        pass
    with esm.esm_model(DEVICE) as second:
        pass

    assert first is second
    assert fakes["esm"] == 1
    assert "esm_offloaded" not in fakes


def test_conformer_generator_built_once(fakes):
    keep_models_resident()

    first = inference_dataset.RefConformerGenerator()
    second = inference_dataset.RefConformerGenerator()

    assert first is second
    assert fakes["conformers"] == 1
