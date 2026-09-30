"""
Tests for the partial scores saved by the score command, and resuming from them.
No model or GPU needed: the scorer is replaced by a fake.
"""

from pathlib import Path

import pytest

import chai_lab.confidence as conf


class FakeScorer:
    scored: list[str] = []
    fail_on: str | None = None

    def __init__(self, msa_directory=None, num_trunk_recycles=3):
        pass

    def score_pdb(self, pdb_file_path):
        name = Path(pdb_file_path).name
        if name == self.fail_on:
            raise RuntimeError("interrupted")
        self.scored.append(name)
        return {"aggregate_score": 0.1 * len(name), "ptm": 0.5, "iptm": 1 / 3}


@pytest.fixture(autouse=True)
def fake_scorer(monkeypatch):
    monkeypatch.setattr(conf, "ChaiConfidenceScorer", FakeScorer)
    monkeypatch.setattr(conf, "make_deterministic", lambda: None)
    monkeypatch.setattr(conf, "keep_models_resident", lambda: None)
    monkeypatch.setattr(FakeScorer, "scored", [])
    monkeypatch.setattr(FakeScorer, "fail_on", None)


@pytest.fixture
def pdbs(tmp_path):
    return [tmp_path / f"s{i}.pdb" for i in range(4)]


def run(pdbs, out, **kwargs):
    conf.score(pdbs, out, keep_models_on_gpu=False, **kwargs)


def test_partial_file_is_written_per_structure_and_removed(tmp_path, pdbs, monkeypatch):
    out = tmp_path / "scores.csv"
    partial = conf._partial_path(out)
    lengths = []

    def score_pdb(self, pdb_file_path):
        lengths.append(len(partial.read_text().splitlines()))
        return {"aggregate_score": 0.5}

    monkeypatch.setattr(FakeScorer, "score_pdb", score_pdb)
    run(pdbs, out)

    # Settings line, plus one line per structure already scored
    assert lengths == [1, 2, 3, 4]
    assert out.is_file()
    assert not partial.exists()


def test_resume_skips_scored_structures_and_gives_same_csv(tmp_path, pdbs):
    reference = tmp_path / "reference.csv"
    run(pdbs, reference)

    out = tmp_path / "scores.csv"
    FakeScorer.scored.clear()
    FakeScorer.fail_on = "s2.pdb"
    with pytest.raises(RuntimeError):
        run(pdbs, out)
    assert not out.exists()
    assert conf._partial_path(out).is_file()

    FakeScorer.scored.clear()
    FakeScorer.fail_on = None
    run(pdbs, out)

    assert FakeScorer.scored == ["s2.pdb", "s3.pdb"]
    assert out.read_text() == reference.read_text()
    assert not conf._partial_path(out).exists()


def test_truncated_last_line_is_rescored(tmp_path, pdbs):
    out = tmp_path / "scores.csv"
    FakeScorer.fail_on = "s2.pdb"
    with pytest.raises(RuntimeError):
        run(pdbs, out)
    partial = conf._partial_path(out)
    with open(partial, "a") as f:
        f.write('{"structure": "s2.pdb", "aggreg')  # crash while writing

    FakeScorer.scored.clear()
    FakeScorer.fail_on = None
    run(pdbs, out)

    assert FakeScorer.scored == ["s2.pdb", "s3.pdb"]
    assert len(out.read_text().splitlines()) == 5


def test_different_settings_raise(tmp_path, pdbs):
    out = tmp_path / "scores.csv"
    FakeScorer.fail_on = "s1.pdb"
    with pytest.raises(RuntimeError):
        run(pdbs, out, num_trunk_recycles=3)

    FakeScorer.fail_on = None
    with pytest.raises(ValueError, match="different settings"):
        run(pdbs, out, num_trunk_recycles=1)
    # The saved scores are still there
    assert conf._partial_path(out).is_file()


def test_no_resume_starts_over(tmp_path, pdbs):
    out = tmp_path / "scores.csv"
    FakeScorer.fail_on = "s2.pdb"
    with pytest.raises(RuntimeError):
        run(pdbs, out)

    FakeScorer.scored.clear()
    FakeScorer.fail_on = None
    run(pdbs, out, resume=False)

    assert FakeScorer.scored == [p.name for p in pdbs]


def test_no_resume_when_not_deterministic(tmp_path, pdbs):
    out = tmp_path / "scores.csv"
    FakeScorer.fail_on = "s2.pdb"
    with pytest.raises(RuntimeError):
        run(pdbs, out, deterministic=False)

    FakeScorer.scored.clear()
    FakeScorer.fail_on = None
    run(pdbs, out, deterministic=False)

    assert FakeScorer.scored == [p.name for p in pdbs]


def test_partial_file_without_settings_is_ignored(tmp_path, pdbs):
    out = tmp_path / "scores.csv"
    conf._partial_path(out).write_text('{"num_tr')

    run(pdbs, out)

    assert FakeScorer.scored == [p.name for p in pdbs]


def test_saved_scores_of_structures_no_longer_listed_are_dropped(tmp_path, pdbs):
    out = tmp_path / "scores.csv"
    FakeScorer.fail_on = "s3.pdb"
    with pytest.raises(RuntimeError):
        run(pdbs, out)

    FakeScorer.fail_on = None
    FakeScorer.scored.clear()
    run(pdbs[1:3], out)

    assert FakeScorer.scored == []
    assert [line.split(",")[0] for line in out.read_text().splitlines()] == [
        "structure",
        "s1.pdb",
        "s2.pdb",
    ]
