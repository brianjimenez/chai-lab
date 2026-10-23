"""
Tests for scoring FASTA files (no structure) with the score command.
No model or GPU needed: Chai's feature context, trunk and ranking are replaced
by fakes.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import chai_lab.confidence as conf
from chai_lab.confidence import ChaiConfidenceScorer

read_fasta = ChaiConfidenceScorer._read_fasta


def write(tmp_path: Path, text: str, name: str = "x.fasta") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_read_fasta_chai_headers(tmp_path):
    path = write(tmp_path, ">protein|name=A\nEVQL\n>protein|name=B\nHKCD\n")
    chains, content = read_fasta(path)
    assert chains == [("A", []), ("B", [])]
    assert content == ">protein|name=A\nEVQL\n>protein|name=B\nHKCD\n"


def test_read_fasta_plain_headers_and_multiline(tmp_path):
    path = write(tmp_path, ">H heavy chain\nevq\nlvE\n\n>L\nHKCD\n")
    chains, content = read_fasta(path)
    assert [c for c, _ in chains] == ["H", "L"]
    assert content == ">protein|name=H\nEVQLVE\n>protein|name=L\nHKCD\n"


def test_read_fasta_output_matches_pdb_derived_fasta(tmp_path):
    # Same format as _to_fasta, so MSA lookup by sequence hash is unchanged
    path = write(tmp_path, ">protein|name=A\nEVQL\n")
    assert read_fasta(path)[1] == ">protein|name=A\nEVQL\n"


@pytest.mark.parametrize(
    "text, match",
    [
        ("", "No sequences"),
        ("EVQL\n>protein|name=A\nEVQL\n", "before first header"),
        (">protein|name=A\n>protein|name=B\nEVQL\n", "empty sequence"),
        (">protein|name=A\nEVQL\n>ligand|name=L\nCCO\n", "only protein"),
        (">rna|name=R\nACGU\n", "only protein"),
    ],
)
def test_read_fasta_errors(tmp_path, text, match):
    with pytest.raises(ValueError, match=match):
        read_fasta(write(tmp_path, text))


def test_score_fasta_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChaiConfidenceScorer(device="cpu").score_fasta(str(tmp_path / "nope.fasta"))


@pytest.fixture
def fake_chai(monkeypatch):
    """Fakes the Chai calls of _score; records the atom_coords it is given."""
    calls = {}
    num_atoms = 7

    def make_context(fasta_file, **kwargs):
        seqs = fasta_file.read_text().splitlines()[1::2]
        return SimpleNamespace(
            chains=[SimpleNamespace(entity_data=SimpleNamespace(sequence=s)) for s in seqs],
            structure_context=SimpleNamespace(num_atoms=num_atoms),
        )

    def run_folding(context, atom_coords=None, **kwargs):
        calls["atom_coords"] = atom_coords
        calls["kwargs"] = kwargs
        return SimpleNamespace(ranking_data=[None])

    def get_scores(_):
        return {
            "aggregate_score": np.array([-99.0]),
            "ptm": np.array([0.6]),
            "iptm": np.array([0.2]),
            "has_inter_chain_clashes": np.array([True]),
            "per_chain_ptm": np.array([0.9, 0.8]),
            "per_chain_pair_iptm": np.array([[0.9, 0.2], [0.1, 0.8]]),
            "chain_chain_clashes": np.array([[0, 5], [5, 0]]),
        }

    monkeypatch.setattr(conf, "make_all_atom_feature_context", make_context)
    monkeypatch.setattr(conf, "run_folding_on_context", run_folding)
    monkeypatch.setattr(conf, "get_scores", get_scores)
    calls["num_atoms"] = num_atoms
    return calls


def test_score_fasta_uses_origin_coordinates_and_drops_clash_scores(tmp_path, fake_chai):
    path = write(tmp_path, ">protein|name=A\nEVQL\n>protein|name=B\nHKCD\n")
    scores = ChaiConfidenceScorer(device="cpu", num_trunk_recycles=6).score_fasta(
        str(path)
    )

    # No structure: all atoms at the origin, a single sample, no diffusion input
    coords = fake_chai["atom_coords"]
    assert coords.shape == (fake_chai["num_atoms"], 3)
    assert torch.count_nonzero(coords) == 0
    assert fake_chai["kwargs"]["num_diffn_samples"] == 1
    assert fake_chai["kwargs"]["num_trunk_recycles"] == 6

    assert scores == {
        "ptm": pytest.approx(0.6),
        "iptm": pytest.approx(0.2),
        "ptm_A": pytest.approx(0.9),
        "ptm_B": pytest.approx(0.8),
        "iptm_A_B": pytest.approx(0.2),
        "iptm_B_A": pytest.approx(0.1),
    }


def test_score_fasta_sequence_mismatch_raises(tmp_path, fake_chai, monkeypatch):
    path = write(tmp_path, ">protein|name=A\nEVQL\n")
    original = conf.make_all_atom_feature_context

    def altered(fasta_file, **kwargs):
        ctx = original(fasta_file, **kwargs)
        ctx.chains[0].entity_data.sequence = "XXXX"
        return ctx

    monkeypatch.setattr(conf, "make_all_atom_feature_context", altered)
    with pytest.raises(AssertionError, match="Sequence mismatch"):
        ChaiConfidenceScorer(device="cpu").score_fasta(str(path))


class DispatchScorer:
    calls: list[tuple[str, str]] = []

    def __init__(self, msa_directory=None, num_trunk_recycles=6):
        pass

    def score_pdb(self, path):
        self.calls.append(("pdb", Path(path).name))
        return {"iptm": 0.1}

    def score_fasta(self, path):
        self.calls.append(("fasta", Path(path).name))
        return {"iptm": 0.2}


def test_score_dispatches_on_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(conf, "ChaiConfidenceScorer", DispatchScorer)
    monkeypatch.setattr(conf, "make_deterministic", lambda: None)
    monkeypatch.setattr(DispatchScorer, "calls", [])
    files = [tmp_path / "a.fasta", tmp_path / "b.pdb", tmp_path / "c.FA", tmp_path / "d.PDB"]
    out = tmp_path / "scores.csv"

    conf.score(files, out, keep_models_on_gpu=False)

    assert DispatchScorer.calls == [
        ("fasta", "a.fasta"),
        ("pdb", "b.pdb"),
        ("fasta", "c.FA"),
        ("pdb", "d.PDB"),
    ]
    assert out.read_text().splitlines()[1:] == [
        "a.fasta,0.2",
        "b.pdb,0.1",
        "c.FA,0.2",
        "d.PDB,0.1",
    ]


def test_default_trunk_recycles_is_6():
    import inspect

    for fn in (conf.score, ChaiConfidenceScorer.__init__):
        assert inspect.signature(fn).parameters["num_trunk_recycles"].default == 6
