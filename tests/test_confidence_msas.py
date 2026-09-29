"""
Tests for the MSA lookup of the score command: exact matches, and MSAs built for
a longer sequence that are trimmed of the extra terminal residues.
"""

from pathlib import Path

import pandas as pd
import pytest

from chai_lab.confidence import ChaiConfidenceScorer
from chai_lab.data.parsing.msas.aligned_pqt import (
    expected_basename,
    parse_aligned_pqt_to_msa_context,
)

SEQ = "ACDEFGHIKLMNPQRT"


def write_msa(msa_dir: Path, rows: list[str], name: str | None = None) -> Path:
    """Writes an .aligned.pqt file whose first row is the query."""
    path = msa_dir / (name or expected_basename(rows[0]))
    pd.DataFrame(
        {
            "sequence": rows,
            "source_database": ["query"] + ["uniref90"] * (len(rows) - 1),
            "pairing_key": [""] + [str(i) for i in range(1, len(rows))],
            "comment": [""] * len(rows),
        }
    ).to_parquet(path, index=False)
    return path


def to_fasta(chains: list[tuple[str, str]]) -> str:
    return "".join(f">protein|name=chain_{c}\n{seq}\n" for c, seq in chains)


def prepare(
    scorer: ChaiConfidenceScorer,
    chains: list[tuple[str, str]],
    out: Path,
    structure_name: str = "structure",
):
    out.mkdir()
    return scorer._prepare_msas(
        [(c, []) for c, _ in chains], to_fasta(chains), structure_name, out
    )


def find_trimmable(scorer: ChaiConfidenceScorer, seq: str):
    return scorer._find_trimmable_msa(seq, scorer._msa_queries())


@pytest.fixture
def msa_dir(tmp_path: Path) -> Path:
    path = tmp_path / "msas"
    path.mkdir()
    return path


@pytest.mark.parametrize(
    "row,start,end,expected",
    [
        ("ABCDE", 1, 4, "BCD"),
        ("A-C-E", 0, 3, "A-C"),
        # Insertions between kept columns are kept
        ("ABcdCE", 1, 3, "BcdC"),
        # Insertions before the first or after the last kept column are dropped
        ("AbBCdE", 1, 3, "BC"),
        ("xABCy", 0, 3, "ABC"),
        # Insertion right after the last kept column
        ("AB-cdCEfF", 1, 3, "B-"),
        ("AB-cdCEfF", 1, 4, "B-cdC"),
    ],
)
def test_trim_a3m_row(row: str, start: int, end: int, expected: str):
    assert ChaiConfidenceScorer._trim_a3m_row(row, start, end) == expected


def test_find_trimmable_msa(msa_dir: Path):
    write_msa(msa_dir, ["M" + SEQ + "SS"])
    best = write_msa(msa_dir, [SEQ + "S"])
    write_msa(msa_dir, ["QWERTY"])
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)

    # The MSA with the fewest extra residues is used
    assert find_trimmable(scorer, SEQ) == (best, 0, 1)
    assert find_trimmable(scorer, "FGHIK") is not None
    assert find_trimmable(scorer, "NOTPRESENT") is None


def test_find_trimmable_msa_n_terminal(msa_dir: Path):
    path = write_msa(msa_dir, ["GSM" + SEQ])
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)
    assert find_trimmable(scorer, SEQ) == (path, 3, 0)


def test_find_trimmable_msa_ambiguous(msa_dir: Path):
    # The sequence occurs twice in the query: the offset is ambiguous
    write_msa(msa_dir, [SEQ + "G" + SEQ])
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)
    assert find_trimmable(scorer, SEQ) is None


def test_write_trimmed_msa(msa_dir: Path, tmp_path: Path):
    query = "M" + SEQ + "SS"
    path = write_msa(
        msa_dir,
        [
            query,
            "D" + SEQ[:5] + "ab" + SEQ[5:] + "TT",  # insertion inside the kept columns
            "-" + "-" * len(SEQ) + "SS",  # only aligned to trimmed columns
            "x-" + "-" * len(SEQ) + "SS",  # insertion only, before the kept columns
        ],
    )
    out = tmp_path / expected_basename(SEQ)
    ChaiConfidenceScorer(msa_directory=msa_dir)._write_trimmed_msa(path, 1, 2, out)

    table = pd.read_parquet(out)
    assert table["sequence"].tolist() == [SEQ, SEQ[:5] + "ab" + SEQ[5:]]
    assert table["source_database"].tolist() == ["query", "uniref90"]
    assert table["pairing_key"].tolist() == ["", "1"]

    # Chai parses it, with one column per residue of the trimmed sequence
    msa = parse_aligned_pqt_to_msa_context(out)
    assert msa.tokens.shape == (2, len(SEQ))
    assert msa.deletion_matrix[1].tolist() == [0] * 5 + [2] + [0] * (len(SEQ) - 6)


def test_prepare_msas(msa_dir: Path, tmp_path: Path):
    exact = write_msa(msa_dir, ["QWERTYIPAS"])
    write_msa(msa_dir, [SEQ + "S", SEQ + "T"])
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)

    out = tmp_path / "run"
    chains = [("A", SEQ), ("B", "QWERTYIPAS"), ("C", "NOMSAHERE")]
    found, trimmed = prepare(scorer, chains, out)

    assert found == 2
    assert trimmed == ["A(N0,C1)"]
    assert (out / exact.name).resolve() == exact.resolve()
    assert pd.read_parquet(out / expected_basename(SEQ))["sequence"].tolist() == [
        SEQ,
        SEQ,
    ]
    assert not (out / expected_basename("NOMSAHERE")).exists()


def test_prepare_msas_exact_skips_index(msa_dir: Path, tmp_path: Path):
    write_msa(msa_dir, [SEQ])
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)
    assert prepare(scorer, [("A", SEQ)], tmp_path / "run") == (1, [])
    # The MSA directory is only indexed when a chain has no exact match
    assert scorer._msa_query_index is None


def test_prepare_msas_repeated_chains(msa_dir: Path, tmp_path: Path):
    write_msa(msa_dir, [SEQ + "S"])
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)

    # Chains with the same sequence share the trimmed MSA
    found, trimmed = prepare(scorer, [("A", SEQ), ("B", SEQ)], tmp_path / "run")
    assert found == 2
    assert trimmed == ["A(N0,C1)"]


def test_prepare_msas_in_subfolders(msa_dir: Path, tmp_path: Path):
    # One subfolder per structure, as produced by per-complex MSA searches
    (msa_dir / "complex_1").mkdir()
    (msa_dir / "complex_2" / "nested").mkdir(parents=True)
    exact = write_msa(msa_dir / "complex_1", ["QWERTYIPAS"])
    write_msa(msa_dir / "complex_2" / "nested", [SEQ + "S"])
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)

    out = tmp_path / "run"
    found, trimmed = prepare(scorer, [("A", SEQ), ("B", "QWERTYIPAS")], out)
    assert found == 2
    assert trimmed == ["A(N0,C1)"]
    assert (out / exact.name).resolve() == exact.resolve()
    assert (out / expected_basename(SEQ)).is_file()


def test_msa_files_prefers_copy_closest_to_root(msa_dir: Path):
    (msa_dir / "a" / "b").mkdir(parents=True)
    (msa_dir / "z").mkdir()
    write_msa(msa_dir / "a" / "b", [SEQ])
    shallow = write_msa(msa_dir / "z", [SEQ])
    assert ChaiConfidenceScorer._index_msa_files(msa_dir) == {
        expected_basename(SEQ): shallow
    }

    root = write_msa(msa_dir, [SEQ])
    assert ChaiConfidenceScorer._index_msa_files(msa_dir) == {
        expected_basename(SEQ): root
    }


def test_prepare_msas_prefers_structure_folder(msa_dir: Path, tmp_path: Path, capsys):
    # Each complex has its own copy of the partner MSA, whose pairing keys only
    # match the other MSAs of the complex
    for name in ["complex_1", "complex_2"]:
        (msa_dir / name).mkdir()
        write_msa(msa_dir / name, ["QWERTYIPAS", "QWERTYIPAS"])
        write_msa(msa_dir / name, [SEQ + "S"])
    own = msa_dir / "complex_2" / expected_basename("QWERTYIPAS")
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)

    out = tmp_path / "run"
    chains = [("A", SEQ), ("B", "QWERTYIPAS")]
    assert prepare(scorer, chains, out, "complex_2") == (2, ["A(N0,C1)"])
    assert (out / own.name).resolve() == own.resolve()
    assert "complex_2/" in capsys.readouterr().out
    # The rest of the directory is not searched when the structure's folder has all MSAs
    assert scorer._msa_query_index is None


def test_prepare_msas_falls_back_to_other_folders(
    msa_dir: Path, tmp_path: Path, capsys
):
    (msa_dir / "complex_1").mkdir()
    (msa_dir / "complex_2").mkdir()
    write_msa(msa_dir / "complex_1", [SEQ + "S"])
    other = write_msa(msa_dir / "complex_2", ["QWERTYIPAS"])
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)

    out = tmp_path / "run"
    chains = [("A", SEQ), ("B", "QWERTYIPAS")]
    assert prepare(scorer, chains, out, "complex_1") == (2, ["A(N0,C1)"])
    assert (out / other.name).resolve() == other.resolve()
    # Using another structure's MSA is flagged, as its pairing may not match
    assert "no MSA for chain B in complex_1/" in capsys.readouterr().out


def test_prepare_msas_exact_content_under_other_name(msa_dir: Path, tmp_path: Path):
    # An MSA whose query equals the chain sequence, but not named by its hash
    path = write_msa(msa_dir, [SEQ], name="custom.aligned.pqt")
    scorer = ChaiConfidenceScorer(msa_directory=msa_dir)

    out = tmp_path / "run"
    assert prepare(scorer, [("A", SEQ)], out) == (1, [])
    assert (out / expected_basename(SEQ)).resolve() == path.resolve()
