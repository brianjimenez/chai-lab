# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Guidance of the diffusion denoised estimate towards satisfying contact restraints."""

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from chai_lab.data import residue_constants as rc
from chai_lab.data.dataset.structure.all_atom_structure_context import (
    AllAtomStructureContext,
)
from chai_lab.data.parsing.restraints import (
    PairwiseInteraction,
    PairwiseInteractionType,
)
from chai_lab.model.utils import get_asym_id_from_subchain_id
from chai_lab.utils.tensor_utils import tensorcode_to_string

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContactGuidanceConfig:
    # gradient: fraction of the linearised contact penalty removed per call;
    # rigid: fraction of each violation corrected per call
    scale: float = 1.0
    sigma_min: float = 1.0
    sigma_max: float = 160.0  # higher values can cause clashes with rigid mode
    # "gradient": step on the gradient of the contact penalty through the diffusion
    # module (needs autograd and memory); "rigid": move whole chains, no gradients
    mode: str = "gradient"
    max_step: float = 10.0  # gradient mode: largest per-atom step in Å per call

    def __post_init__(self):
        assert self.mode in ("gradient", "rigid"), self.mode

    def weight(self, sigma: float) -> float:
        return self.scale if self.sigma_min <= sigma <= self.sigma_max else 0.0


class ContactGuidance:
    """Contacts resolved against an AllAtomStructureContext.

    Each side of a contact is a set of candidate atoms: all the heavy atoms of the
    residue, or the single atom if the restraint names one. The contact distance is
    the closest pair between the two sets. In rigid mode (`apply`) whole chains are
    moved so that violated inter-chain contacts get closer to being satisfied; in
    gradient mode (`loss` and `gradient_step`) the caller backpropagates the penalty
    through the diffusion module.
    """

    def __init__(
        self,
        atoms_a: Tensor,
        atoms_b: Tensor,
        group_a: Tensor,
        group_b: Tensor,
        min_dist: Tensor,
        max_dist: Tensor,
        conf: Tensor,
        config: ContactGuidanceConfig,
    ):
        # (P, K) atom indices, padded with -1
        self.atoms_a, self.atoms_b = atoms_a.long(), atoms_b.long()
        self.group_a, self.group_b = group_a.long(), group_b.long()
        self.min_dist, self.max_dist = min_dist.float(), max_dist.float()
        self.conf = conf.float()
        self.config = config

    @property
    def num_contacts(self) -> int:
        return self.atoms_a.shape[0]

    def to(self, device: torch.device) -> "ContactGuidance":
        for name in (
            "atoms_a",
            "atoms_b",
            "group_a",
            "group_b",
            "min_dist",
            "max_dist",
            "conf",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @classmethod
    def from_interactions(
        cls,
        interactions: list[PairwiseInteraction],
        ctx: AllAtomStructureContext,
        config: ContactGuidanceConfig,
    ) -> "ContactGuidance | None":
        rows = []
        for it in interactions:
            if it.connection_type != PairwiseInteractionType.CONTACT:
                continue
            tok_a, atoms_a = _resolve(it.chainA, it.res_idxA, it.atom_nameA, ctx)
            tok_b, atoms_b = _resolve(it.chainB, it.res_idxB, it.atom_nameB, ctx)
            group_a = int(ctx.token_asym_id[tok_a])
            group_b = int(ctx.token_asym_id[tok_b])
            if group_a == group_b:
                logger.warning(
                    f"Contact guidance skips same-chain contact {it.chainA}:"
                    f"{it.res_idxA} - {it.chainB}:{it.res_idxB}"
                )
                continue
            rows.append(
                (
                    atoms_a,
                    atoms_b,
                    group_a,
                    group_b,
                    it.min_dist_angstrom,
                    it.max_dist_angstrom,
                    it.confidence,
                )
            )
        if not rows:
            logger.warning("No usable contact restraints for contact guidance")
            return None
        a, b, ga, gb, lo, hi, conf = zip(*rows)
        return cls(
            atoms_a=_pad(a),
            atoms_b=_pad(b),
            group_a=torch.tensor(ga),
            group_b=torch.tensor(gb),
            min_dist=torch.tensor(lo),
            max_dist=torch.tensor(hi),
            conf=torch.tensor(conf)[None, :, None],
            config=config,
        )

    def _closest_pairs(self, coords: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Closest atom pair per contact: position of a, vector a->b, distance.

        Shapes (B, P, 3), (B, P, 3), (B, P, 1).
        """
        ma, mb = self.atoms_a >= 0, self.atoms_b >= 0
        xa = coords[:, self.atoms_a.clamp_min(0)]  # (B, P, Ka, 3)
        xb = coords[:, self.atoms_b.clamp_min(0)]  # (B, P, Kb, 3)
        d = torch.linalg.norm(xa[:, :, :, None] - xb[:, :, None, :], dim=-1)
        d = d.masked_fill(~(ma[None, :, :, None] & mb[None, :, None, :]), torch.inf)
        flat = d.flatten(2).argmin(-1)  # (B, P)
        ia, ib = flat // d.shape[-1], flat % d.shape[-1]
        pos_a = _pick(xa, ia)
        vec = _pick(xb, ib) - pos_a
        dist = torch.take_along_dim(d.flatten(2), flat[..., None], dim=2)
        return pos_a, vec, dist

    def violations(self, coords: Tensor) -> Tensor:
        """Signed excess in Å, (B, P): >0 too far, <0 too close, 0 satisfied."""
        _, _, dist = self._closest_pairs(coords)
        dist = dist[..., 0]
        return (dist - self.max_dist).clamp_min(0) - (self.min_dist - dist).clamp_min(0)

    def apply(
        self,
        denoised: Tensor,
        sigma: Tensor | float,
        *,
        atom_group: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        """Rigid-body move of each chain towards satisfying its violated contacts.

        denoised: (B, A, 3); atom_group, atom_mask: (1, A). Per chain, the violated
        contacts define target displacements of their closest atoms (each side
        moves half the violation, scaled by weight and confidence). The chain gets
        the mean displacement as translation, plus the rotation (about the centroid
        of those atoms) that best fits the remaining differences, so several
        contacts between the same chains are satisfied jointly. Satisfied contacts
        do not take part, so they never dilute the correction.
        """
        w = self.config.weight(float(sigma))
        if w == 0.0:
            return denoised
        pos_a, vec, dist = self._closest_pairs(denoised)  # (B, P, 3) x2, (B, P, 1)
        viol = self.violations(denoised)[..., None]  # (B, P, 1)
        shift = 0.5 * w * self.conf * viol * vec / dist.clamp_min(1e-6)  # a -> b
        active = (viol != 0).to(denoised.dtype)

        # one entry per contact side: (B, 2P, ...)
        groups = torch.cat([self.group_a, self.group_b])
        pts = torch.cat([pos_a, pos_a + vec], dim=1)
        disp = torch.cat([shift, -shift], dim=1)
        m = torch.cat([active, active], dim=1)

        n_groups = int(atom_group.max()) + 1
        B = denoised.shape[0]

        def group_sum(x: Tensor) -> Tensor:
            out = x.new_zeros(B, n_groups, *x.shape[2:])
            return out.index_add_(1, groups, x)

        n = group_sum(m).clamp_min(1.0)  # (B, G, 1)
        pivot = group_sum(m * pts) / n
        trans = group_sum(m * disp) / n
        r = pts - pivot[:, groups]  # (B, 2P, 3)
        resid = disp - trans[:, groups]
        torque = group_sum(m * torch.cross(r, resid, dim=-1))  # (B, G, 3)
        eye = torch.eye(3, dtype=r.dtype, device=r.device)
        inertia = group_sum(
            m[..., None]
            * (
                (r * r).sum(-1)[..., None, None] * eye
                - r[..., :, None] * r[..., None, :]
            )
        )  # (B, G, 3, 3)
        ridge = 1e-2 * inertia.diagonal(dim1=-2, dim2=-1).mean(-1) + 1e-6
        omega = torch.linalg.solve(
            inertia + ridge[..., None, None] * eye, torque[..., None]
        )
        omega = omega[..., 0]  # (B, G, 3), small-angle rotation vector
        angle = omega.norm(dim=-1, keepdim=True)
        omega = omega * (angle.clamp_max(_MAX_ROTATION) / angle.clamp_min(1e-9))

        # exact rotation (Rodrigues) about the pivot, then translation
        g = atom_group[0]
        rel = denoised - pivot[:, g]
        theta = omega.norm(dim=-1, keepdim=True)[:, g]
        k = omega[:, g] / theta.clamp_min(1e-9)
        rotated = (
            rel * theta.cos()
            + torch.cross(k, rel, dim=-1) * theta.sin()
            + k * (k * rel).sum(-1, keepdim=True) * (1 - theta.cos())
        )
        moved = pivot[:, g] + rotated + trans[:, g]
        return torch.where(atom_mask[0, :, None].bool(), moved, denoised)

    def loss(self, coords: Tensor) -> Tensor:
        """Flat-bottom contact penalty, ½·conf·excess², summed per sample: (B,)."""
        v = self.violations(coords)
        return 0.5 * (self.conf[..., 0] * v**2).sum(-1)

    def gradient_step(
        self, grad: Tensor, loss: Tensor, sigma: float, atom_mask: Tensor
    ) -> Tensor:
        """Displacement for the denoiser input, from d loss / d input.

        The gradient through the network is small and spread over many atoms, so a
        fixed step size does little. Instead the step length is chosen to cut the
        linearised loss by the fraction `scale`: delta = -scale * loss * g / |g|^2
        (per sample), then clipped per atom to max_step.
        grad: (B, A, 3); loss: (B,).
        """
        w = self.config.weight(sigma)
        g = grad * atom_mask[..., None]
        gsq = (g * g).sum(dim=(-2, -1)).clamp_min(1e-12)  # (B,)
        step = -w * (loss / gsq)[:, None, None] * g
        norm = step.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        return step * (norm.clamp_max(self.config.max_step) / norm)


# largest rotation (radians) a chain gets per denoising call
_MAX_ROTATION = 0.1


def _pick(x: Tensor, i: Tensor) -> Tensor:
    return torch.take_along_dim(x, i[..., None, None], dim=2)[:, :, 0]


def _pad(atom_lists) -> Tensor:
    k = max(len(a) for a in atom_lists)
    return torch.tensor([list(a) + [-1] * (k - len(a)) for a in atom_lists])


def _resolve(
    chain: str, res_idx: str, atom_name: str, ctx: AllAtomStructureContext
) -> tuple[int, list[int]]:
    """Returns (token index, candidate atom indices) for one side of a contact."""
    asym = get_asym_id_from_subchain_id(
        subchain_id=chain,
        source_pdb_chain_id=ctx.subchain_id,
        token_asym_id=ctx.token_asym_id,
    )
    pos = int(res_idx[1:]) if res_idx[1:] else 1
    toks = torch.where(
        (ctx.token_asym_id == asym) & (ctx.token_residue_index == pos - 1)
    )[0]
    assert toks.numel() > 0, f"No residue found for {chain}:{res_idx}"
    if res_idx:
        expected = rc.restype_1to3_with_x[res_idx[0]]
        for t in toks.tolist():
            got = tensorcode_to_string(ctx.token_residue_name[t])
            assert got == expected, f"{chain}:{res_idx}: expected {expected}, got {got}"
    # NOTE glycans have several tokens per residue
    atoms = torch.where(torch.isin(ctx.atom_token_index, toks) & ctx.atom_exists_mask)[
        0
    ].tolist()
    if atom_name:
        atoms = [i for i in atoms if ctx.atom_ref_name[i] == atom_name]
        assert len(atoms) == 1, f"{chain}:{res_idx}@{atom_name}: {len(atoms)} atoms"
    assert atoms, f"{chain}:{res_idx} has no atoms"
    return int(toks[0]), atoms
