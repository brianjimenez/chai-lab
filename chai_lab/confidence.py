import csv
import torch
import numpy as np
import tempfile
from pathlib import Path
from Bio.PDB import PDBParser

# Import chai_lab core modules
try:
    import chai_lab.chai1
    from chai_lab.chai1 import run_inference
except ImportError:
    raise ImportError("chai_lab is not installed. Please install it via: pip install -e .")

class ChaiConfidenceScorer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        # Save the original loader so we can restore it later
        self.original_load_exported = chai_lab.chai1.load_exported

    def _extract_fasta_and_coords(self, pdb_path: str):
        """
        Parses the PDB to extract the amino acid sequence (FASTA) 
        and the Carbon-Alpha (CA) coordinates.
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("input", pdb_path)
        
        # Standard amino acid mapping to avoid Biopython versioning issues
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
            'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        
        fasta_str = ""
        ca_coords = []
        
        for model in structure:
            for chain in model:
                chain_seq = ""
                for residue in chain:
                    res_name = residue.get_resname().upper()
                    
                    # Check if it's a standard amino acid
                    if res_name in three_to_one:
                        chain_seq += three_to_one[res_name]
                        
                        # Extract CA coordinate (fallback to 0 if missing)
                        if 'CA' in residue:
                            ca_coords.append(residue['CA'].get_coord())
                        else:
                            ca_coords.append([0.0, 0.0, 0.0])
                            
                if chain_seq:
                    fasta_str += f">protein|name=chain_{chain.id}\n{chain_seq}\n"
            break # Only process the first model in the PDB
            
        return fasta_str, np.array(ca_coords)

    def score_pdb(self, pdb_file_path: str) -> float:
        """
        Runs the sequence through the Chai trunk and injects the PDB 
        coordinates into the confidence head to calculate the different scores.
        """
        path = Path(pdb_file_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")

        print(f"Parsing {path.name}...")
        fasta_content, pdb_coords = self._extract_fasta_and_coords(pdb_file_path)
        
        if len(pdb_coords) == 0:
            raise ValueError("No valid amino acid CA coordinates found in PDB.")

        # Proxy class to intercept the confidence head
        class ConfidenceHeadProxy:
            def __init__(self, original_module, pdb_coords):
                self.original_module = original_module
                self.pdb_coords = pdb_coords

            def __call__(self, *args, **kwargs):
                print(">> Intercepting confidence head! Injecting custom PDB coordinates...")
                
                # Identify where the pred_coords are
                is_kwarg = 'pred_coords' in kwargs
                
                if is_kwarg:
                    original_coords = kwargs['pred_coords']
                elif len(args) > 2:
                    original_coords = args[2]
                else:
                    print(">> Warning: Could not locate pred_coords in arguments. Running original.")
                    return self.original_module(*args, **kwargs)

                modified_coords = original_coords.clone()
                
                # Convert our PDB coords to match the tensor properties
                pdb_tensor = torch.tensor(
                    self.pdb_coords, 
                    dtype=original_coords.dtype, 
                    device=original_coords.device
                )
                
                L = min(pdb_tensor.shape[0], original_coords.shape[1])
                
                if len(original_coords.shape) == 3:
                    modified_coords[0, :L, :] = pdb_tensor[:L, :]
                elif len(original_coords.shape) == 4:
                    modified_coords[0, :L, 1, :] = pdb_tensor[:L, :]
                    
                # Package the modified coordinates back into the arguments
                if is_kwarg:
                    kwargs['pred_coords'] = modified_coords
                    new_args = args
                else:
                    new_args = list(args)
                    new_args[2] = modified_coords
                    
                # Run the actual model with the injected coordinates
                # The pipeline will calculate the pTM downstream from these outputs
                return self.original_module(*new_args, **kwargs)

            def __getattr__(self, name):
                return getattr(self.original_module, name)

        def patched_load_exported(name, device):
            module = self.original_load_exported(name, device)
            
            # Wrap the confidence head with our proxy
            if "confidence_head" in name:
                return ConfidenceHeadProxy(module, pdb_coords)
                
            return module

        # Apply patch and run the inference pipeline
        chai_lab.chai1.load_exported = patched_load_exported
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                base_path = Path(tmpdir)
                
                tmp_fasta = base_path / "input.fasta"
                tmp_fasta.write_text(fasta_content)
                
                chai_output_dir = base_path / "chai_outputs"
                chai_output_dir.mkdir(exist_ok=True)
                
                print("Running Chai-1 trunk to generate embeddings (skipping diffusion)...")
                
                run_inference(
                    fasta_file=tmp_fasta,
                    output_dir=chai_output_dir, 
                    num_trunk_recycles=1,       
                    num_diffn_timesteps=1,      
                    device=self.device
                )
                
                # Dynamically locate the output .npz file
                npz_files = list(chai_output_dir.glob("*.npz"))
                
                if not npz_files:
                    print(f"Contents of output directory: {list(chai_output_dir.iterdir())}")
                    raise RuntimeError("ERROR: Inference failed to produce any .npz score files.")
                
                # Read the first generated .npz file
                target_file = npz_files[0]
                data = np.load(target_file)
                
                # Scores: 'aggregate_score', 'ptm', 'iptm', 'per_chain_ptm', 'per_chain_pair_iptm', 'has_inter_chain_clashes', 'chain_chain_clashes'
                scores = {}
                for score in data.keys():
                    scores[score] = float(data[score].flatten()[0])
                
                return scores
                    
        finally:
            # Restore the original loader
            chai_lab.chai1.load_exported = self.original_load_exported


def score(pdb_files: list[Path], csv_output_path: Path):
    """
    Scores a list of PDB structures through the confidence head and 
    write the result in a CSV output file. 
    """
    print("Loading model...")
    scorer = ChaiConfidenceScorer()
    print("Done.")
    scores = []
    i = 1
    total = len(pdb_files)
    print(f"Scoring {total} structures:")
    for pdb_file in pdb_files:
        pdb_scores = scorer.score_pdb(pdb_file)
        pdb_scores['structure'] = pdb_file.name
        scores.append(pdb_scores)
        print(f"  > Run {i} / {total}: {pdb_scores}")
        i += 1

    headers = scores[0].keys()
    with open(csv_output_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(scores)
