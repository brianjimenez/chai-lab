import csv
import torch
import numpy as np
import tempfile
from pathlib import Path
from Bio.PDB import PDBParser
from Bio.PDB.Residue import Residue

# Import chai_lab core modules
try:
    from chai_lab.chai1 import make_all_atom_feature_context, run_folding_on_context
    from chai_lab.data.dataset.structure.all_atom_structure_context import AllAtomStructureContext
    from chai_lab.data.parsing.msas.aligned_pqt import expected_basename
    from chai_lab.ranking.rank import get_scores
except ImportError:
    raise ImportError("chai_lab is not installed. Please install it via: pip install -e .")

# Standard amino acid mapping to avoid Biopython versioning issues
THREE_TO_ONE = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def _superimpose(mobile: np.ndarray, target: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Finds the rigid transform (Kabsch) that best superimposes mobile onto target
    and applies it to points.
    """
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    h = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (points - mobile_center) @ rotation.T + target_center


class ChaiConfidenceScorer:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        msa_directory: Path | None = None,
        num_trunk_recycles: int = 3,
    ):
        self.device = device
        if num_trunk_recycles < 1:
            raise ValueError(f"num_trunk_recycles must be >= 1, got {num_trunk_recycles}")
        self.num_trunk_recycles = num_trunk_recycles
        if msa_directory is not None and not Path(msa_directory).is_dir():
            raise NotADirectoryError(f"MSA directory not found: {msa_directory}")
        self.msa_directory = Path(msa_directory) if msa_directory is not None else None

    def _check_msa_coverage(self, fasta_content: str) -> tuple[int, int]:
        """
        Chai looks up MSAs by the hash of each chain sequence and silently falls
        back to single-sequence mode when no .aligned.pqt file matches. As the
        sequences here are derived from the PDB (which may miss residues), report
        the chains without a matching MSA file.
        """
        lines = fasta_content.splitlines()
        found = 0
        for header, seq in zip(lines[0::2], lines[1::2]):
            msa_file = self.msa_directory / expected_basename(seq)
            if msa_file.is_file():
                found += 1
            else:
                print(f">> Warning: no MSA found for {header[1:]} (expected {msa_file.name}). "
                      "This chain will be scored in single-sequence mode.")
        return found, len(lines) // 2

    def _extract_chains(self, pdb_path: str) -> list[tuple[str, list[Residue]]]:
        """
        Parses the PDB and returns, for each protein chain of the first model,
        its id and its standard amino acid residues.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("input", pdb_path)
        model = next(iter(structure))  # Only process the first model in the PDB

        chains = []
        for chain in model:
            residues = [r for r in chain if r.get_resname().upper() in THREE_TO_ONE]
            if residues:
                chains.append((chain.id, residues))
        return chains

    @staticmethod
    def _to_fasta(chains: list[tuple[str, list[Residue]]]) -> str:
        fasta_str = ""
        for chain_id, residues in chains:
            chain_seq = "".join(THREE_TO_ONE[r.get_resname().upper()] for r in residues)
            fasta_str += f">protein|name=chain_{chain_id}\n{chain_seq}\n"
        return fasta_str

    @staticmethod
    def _map_atom_coords(
        structure_context: AllAtomStructureContext,
        chains: list[tuple[str, list[Residue]]],
    ) -> tuple[torch.Tensor, int]:
        """
        Places the PDB atoms into Chai's atom layout, matching them by chain,
        residue position and atom name. Heavy atoms missing in the PDB are rebuilt
        by superimposing the residue's reference conformer onto the atoms present.
        Returns the coordinates (one row per Chai atom) and the number of rebuilt atoms.
        """
        atom_token_index = structure_context.atom_token_index
        atom_chain_index = (structure_context.token_asym_id[atom_token_index] - 1).tolist()
        atom_residue_index = structure_context.token_residue_index[atom_token_index].tolist()
        atom_names = structure_context.atom_ref_name

        num_atoms = structure_context.num_atoms
        coords = np.zeros((num_atoms, 3), dtype=np.float32)
        found = np.zeros(num_atoms, dtype=bool)
        for i in range(num_atoms):
            _, residues = chains[atom_chain_index[i]]
            residue = residues[atom_residue_index[i]]
            if atom_names[i] in residue:
                coords[i] = residue[atom_names[i]].get_coord()
                found[i] = True

        ref_pos = structure_context.atom_ref_pos.numpy()
        atom_token_index = atom_token_index.numpy()
        for token in np.unique(atom_token_index[~found]):
            token_atoms = atom_token_index == token
            present = token_atoms & found
            missing = token_atoms & ~found
            if present.sum() < 3:
                chain_id, residues = chains[atom_chain_index[missing.argmax()]]
                residue = residues[atom_residue_index[missing.argmax()]]
                raise ValueError(
                    f"Chain {chain_id} residue {residue.get_resname()}{residue.id[1]} has "
                    f"only {present.sum()} heavy atoms; at least 3 are needed to rebuild the missing ones."
                )
            coords[missing] = _superimpose(ref_pos[present], coords[present], ref_pos[missing])

        return torch.from_numpy(coords), int((~found).sum())

    @staticmethod
    def _flatten_scores(raw_scores: dict[str, np.ndarray], chain_labels: list[str]) -> dict:
        """
        Turns Chai's score arrays into flat columns named by chain:
        - ptm_<X>: pTM of chain X
        - iptm_<X>_<Y>: ipTM of chain Y from the alignment on chain X (not symmetric)
        - clashes_<X>_<Y>: number of clashing atom pairs between chains X and Y
        The within-chain entries are left out: the pair ipTM diagonal repeats the
        per-chain pTM, and intra-chain clashes are not used by Chai's ranking.
        """
        n = len(chain_labels)
        per_chain_ptm = raw_scores['per_chain_ptm'].reshape(n)
        pair_iptm = raw_scores['per_chain_pair_iptm'].reshape(n, n)
        clashes = raw_scores['chain_chain_clashes'].reshape(n, n)

        scores = {
            'aggregate_score': float(raw_scores['aggregate_score'].item()),
            'ptm': float(raw_scores['ptm'].item()),
            'iptm': float(raw_scores['iptm'].item()),
            'has_inter_chain_clashes': bool(raw_scores['has_inter_chain_clashes'].item()),
        }
        for i, x in enumerate(chain_labels):
            scores[f'ptm_{x}'] = float(per_chain_ptm[i])
        for i, x in enumerate(chain_labels):
            for j, y in enumerate(chain_labels):
                if i != j:
                    scores[f'iptm_{x}_{y}'] = float(pair_iptm[i, j])
        for i, x in enumerate(chain_labels):
            for j, y in enumerate(chain_labels[i + 1:], start=i + 1):
                scores[f'clashes_{x}_{y}'] = int(clashes[i, j])
        return scores

    def score_pdb(self, pdb_file_path: str) -> dict:
        """
        Runs the sequence through the Chai trunk and scores the PDB coordinates
        with the confidence head to calculate the different scores.
        """
        path = Path(pdb_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")

        print(f"Parsing {path.name}...")
        chains = self._extract_chains(pdb_file_path)
        if not chains:
            raise ValueError("No standard amino acid residues found in PDB.")
        fasta_content = self._to_fasta(chains)

        msa_coverage = None
        if self.msa_directory is not None:
            found, total = self._check_msa_coverage(fasta_content)
            msa_coverage = f"{found}/{total}"

        device = torch.device(self.device)
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = Path(tmpdir)

            tmp_fasta = base_path / "input.fasta"
            tmp_fasta.write_text(fasta_content)

            feature_context = make_all_atom_feature_context(
                fasta_file=tmp_fasta,
                output_dir=base_path / "features",
                use_esm_embeddings=True,
                msa_directory=self.msa_directory,
                esm_device=device,
            )
            parsed_seqs = fasta_content.splitlines()[1::2]
            chai_seqs = [chain.entity_data.sequence for chain in feature_context.chains]
            assert chai_seqs == parsed_seqs, f"Sequence mismatch: {chai_seqs} != {parsed_seqs}"

            atom_coords, num_rebuilt = self._map_atom_coords(
                feature_context.structure_context, chains
            )
            if num_rebuilt > 0:
                print(f">> Warning: rebuilt {num_rebuilt} heavy atoms missing in {path.name}.")

            print("Running Chai-1 trunk and scoring the PDB coordinates (skipping diffusion)...")
            candidates = run_folding_on_context(
                feature_context,
                output_dir=base_path / "chai_outputs",
                num_trunk_recycles=self.num_trunk_recycles,
                num_diffn_samples=1,
                device=device,
                low_memory=True,
                atom_coords=atom_coords,
            )

        chain_labels = [chain_id.strip() or f"chain{i + 1}" for i, (chain_id, _) in enumerate(chains)]
        scores = self._flatten_scores(get_scores(candidates.ranking_data[0]), chain_labels)

        scores['rebuilt_atoms'] = num_rebuilt
        if msa_coverage is not None:
            scores['msa_chains'] = msa_coverage

        return scores


def score(
    pdb_files: list[Path],
    csv_output_path: Path,
    msa_directory: Path | None = None,
    num_trunk_recycles: int = 3,
):
    """
    Scores a list of PDB structures through the confidence head and
    write the result in a CSV output file.

    If msa_directory is given, precomputed MSAs (.aligned.pqt files named by
    sequence hash, e.g. from `chai-lab a3m-to-pqt`) are used by the trunk.
    num_trunk_recycles defaults to 3, as in `chai-lab fold`, so that scores
    are comparable to regular Chai-1 predictions.
    """
    print("Loading model...")
    scorer = ChaiConfidenceScorer(
        msa_directory=msa_directory, num_trunk_recycles=num_trunk_recycles
    )
    if msa_directory is not None:
        print(f"Using MSAs from {msa_directory}")
    print("Done.")
    scores = []
    i = 1
    total = len(pdb_files)
    print(f"Scoring {total} structures:")
    for pdb_file in pdb_files:
        pdb_scores = {'structure': pdb_file.name, **scorer.score_pdb(pdb_file)}
        scores.append(pdb_scores)
        print(f"  > Run {i} / {total}: {pdb_scores}")
        i += 1

    # Per-chain columns depend on each structure's chains: write the union of all
    # columns, leaving the ones a structure does not have empty
    headers = list(dict.fromkeys(key for pdb_scores in scores for key in pdb_scores))
    with open(csv_output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers, restval="")
        writer.writeheader()
        writer.writerows(scores)
