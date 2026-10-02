# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import pytest
import torch

from chai_lab.data.dataset.inference_dataset import Input, load_chains_from_raw
from chai_lab.data.dataset.structure.all_atom_structure_context import (
    AllAtomStructureContext,
)
from chai_lab.data.parsing.restraints import (
    PairwiseInteraction,
    PairwiseInteractionType,
)
from chai_lab.data.parsing.structure.entity_type import EntityType
from chai_lab.model.contact_guidance import ContactGuidance, ContactGuidanceConfig

_EXACT = "donot_use_mm_for_euclid_dist"


def _contact(a="A", ia="G2", b="B", ib="H3", max_dist=6.0):
    return PairwiseInteraction(
        chainA=a,
        res_idxA=ia,
        atom_nameA="",
        chainB=b,
        res_idxB=ib,
        atom_nameB="",
        connection_type=PairwiseInteractionType.CONTACT,
        max_dist_angstrom=max_dist,
    )


@pytest.fixture(scope="module")
def ctx() -> AllAtomStructureContext:
    chains = load_chains_from_raw(
        [
            Input("GGGGGG", entity_type=EntityType.PROTEIN.value, entity_name="G"),
            Input("HHHHHH", entity_type=EntityType.PROTEIN.value, entity_name="H"),
        ],
        entity_name_as_subchain=False,
    )
    return AllAtomStructureContext.merge([c.structure_context for c in chains])


def test_resolution(ctx):
    cfg = ContactGuidanceConfig(mode="rigid")
    g = ContactGuidance.from_interactions([_contact()], ctx, cfg)
    assert g is not None and g.num_contacts == 1
    atoms_a, atoms_b = g.atoms_a[0][g.atoms_a[0] >= 0], g.atoms_b[0][g.atoms_b[0] >= 0]
    # all heavy atoms of the residue (G2, H3)
    assert (ctx.token_residue_index[ctx.atom_token_index[atoms_a]] == 1).all()
    assert (ctx.token_residue_index[ctx.atom_token_index[atoms_b]] == 2).all()
    assert len(atoms_a) > 1 and g.group_a != g.group_b
    # same-chain contacts are skipped
    assert (
        ContactGuidance.from_interactions([_contact(b="A", ib="G4")], ctx, cfg) is None
    )
    # wrong residue name is rejected
    with pytest.raises(AssertionError):
        ContactGuidance.from_interactions([_contact(ia="H2")], ctx, cfg)


def test_apply(ctx):
    torch.manual_seed(1)
    cfg = ContactGuidanceConfig(scale=0.5, mode="rigid")
    g = ContactGuidance.from_interactions([_contact()], ctx, cfg)
    n = ctx.num_atoms
    atom_group = ctx.token_asym_id[ctx.atom_token_index][None]
    mask = torch.ones(1, n)
    x = torch.randn(1, n, 3)
    x[:, atom_group[0] == g.group_b[0]] += torch.tensor([30.0, 0, 0])
    start = x.clone()
    assert g.violations(x).item() > 20

    # outside the sigma window: no-op
    assert torch.equal(g.apply(x, 500.0, atom_group=atom_group, atom_mask=mask), x)

    for _ in range(20):
        x = g.apply(x, 10.0, atom_group=atom_group, atom_mask=mask)
    assert g.violations(x).abs().item() < 1e-2
    _, _, dist = g._closest_pairs(x)
    assert dist.item() <= 6.0 + 1e-2

    # rigid shifts: intra-chain distances unchanged
    for grp in atom_group[0].unique():
        sel = atom_group[0] == grp
        assert torch.allclose(
            torch.cdist(x[0, sel], x[0, sel], compute_mode=_EXACT),
            torch.cdist(start[0, sel], start[0, sel], compute_mode=_EXACT),
            atol=1e-3,
        )


def _rotation(axis, angle):
    axis = torch.tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()
    K = torch.zeros(3, 3)
    K[0, 1], K[0, 2], K[1, 2] = -axis[2], axis[1], -axis[0]
    K = K - K.T
    return torch.eye(3) + angle.sin() * K + (1 - angle.cos()) * K @ K


def test_multiple_contacts_need_rotation(ctx):
    """Several contacts between two chains, with a rotated + shifted partner."""
    torch.manual_seed(0)
    n = ctx.num_atoms
    atom_group = ctx.token_asym_id[ctx.atom_token_index][None]
    mask = torch.ones(1, n)
    in_b = atom_group[0] == atom_group[0, -1]
    # "native": both chains compact and interpenetrating, so every contact holds
    native = torch.randn(1, n, 3) * 4.0
    pairs = [("G1", "H1"), ("G2", "H3"), ("G4", "H4"), ("G6", "H6")]
    interactions = [_contact(ia=a, ib=b, max_dist=6.0) for a, b in pairs]
    g = ContactGuidance.from_interactions(
        interactions, ctx, ContactGuidanceConfig(scale=1.0, mode="rigid")
    )
    assert g.num_contacts == 4
    # make the contacts satisfiable: native contacts are close
    _, _, d0 = g._closest_pairs(native)
    g.max_dist = torch.full_like(g.max_dist, float(d0.max()) + 0.5)
    assert g.violations(native).abs().max() == 0

    R = _rotation([1.0, 2.0, -1.0], torch.tensor(0.7))
    x = native.clone()
    x[:, in_b] = native[:, in_b] @ R.T + torch.tensor([18.0, -9.0, 6.0])
    start = x.clone()
    assert g.violations(x).max() > 5

    for _ in range(100):
        x = g.apply(x, 10.0, atom_group=atom_group, atom_mask=mask)
    assert g.violations(x).abs().max() < 0.5
    for sel in (in_b, ~in_b):  # rigid
        assert torch.allclose(
            torch.cdist(x[0, sel], x[0, sel], compute_mode=_EXACT),
            torch.cdist(start[0, sel], start[0, sel], compute_mode=_EXACT),
            atol=1e-2,
        )


def test_satisfied_contacts_do_not_dilute(ctx):
    torch.manual_seed(2)
    """One violated contact among many satisfied ones is still corrected quickly."""
    n = ctx.num_atoms
    atom_group = ctx.token_asym_id[ctx.atom_token_index][None]
    mask = torch.ones(1, n)
    in_b = atom_group[0] == atom_group[0, -1]
    x = torch.randn(1, n, 3)
    pairs = [("G1", "H1"), ("G2", "H3"), ("G4", "H4"), ("G6", "H6")]
    g = ContactGuidance.from_interactions(
        [_contact(ia=a, ib=b, max_dist=100.0) for a, b in pairs],
        ctx,
        ContactGuidanceConfig(scale=1.0, mode="rigid"),
    )
    g.max_dist[0] = 2.0  # only the first contact is tight, and it is violated
    x[:, in_b] += torch.tensor([20.0, 0, 0])
    before = g.violations(x)[0, 0]
    x = g.apply(x, 10.0, atom_group=atom_group, atom_mask=mask)
    assert g.violations(x)[0, 0] <= 0.5 * before  # half the violation closed in a step


def _scene(ctx, **cfg):
    g = ContactGuidance.from_interactions(
        [_contact()], ctx, ContactGuidanceConfig(scale=1.0, **cfg)
    )
    atom_group = ctx.token_asym_id[ctx.atom_token_index][None]
    atom_res = ctx.token_residue_index[ctx.atom_token_index][None]
    x = torch.randn(1, ctx.num_atoms, 3)
    x[:, atom_group[0] == g.group_b[0]] += torch.tensor([30.0, 0, 0])
    return g, x, atom_group, atom_res, torch.ones(1, ctx.num_atoms)


def test_gradient_step_pulls_contact_atoms_together(ctx):
    torch.manual_seed(4)
    g, x, grp, res, mask = _scene(ctx, mode="gradient", max_step=2.0)
    x.requires_grad_(True)
    loss = g.loss(x)
    assert loss.item() > 100
    (grad,) = torch.autograd.grad(loss.sum(), x)
    step = g.gradient_step(grad, loss.detach(), 10.0, mask)
    assert step.norm(dim=-1).max() <= 2.0 + 1e-5  # clipped
    assert step.abs().sum() > 0
    assert g.loss((x + step).detach()).item() < loss.item()
    # outside the window: no step
    assert g.gradient_step(grad, loss.detach(), 500.0, mask).abs().sum() == 0
    # unclipped, the step cuts the linearised loss by `scale` (here to ~zero)
    g.config = ContactGuidanceConfig(scale=1.0, mode="gradient", max_step=1e6)
    full = g.gradient_step(grad, loss.detach(), 10.0, mask)
    assert abs((grad * full).sum().item() + loss.item()) < 1e-3 * loss.item()


def test_config_defaults_and_validation():
    cfg = ContactGuidanceConfig()
    assert cfg.mode == "gradient" and cfg.scale == 1.0 and cfg.max_step == 10.0
    assert (cfg.sigma_min, cfg.sigma_max) == (1.0, 160.0)
    with pytest.raises(AssertionError):
        ContactGuidanceConfig(mode="local")


def test_guidance_requires_constraint_path(tmp_path):
    from chai_lab.chai1 import run_inference

    with pytest.raises(AssertionError, match="constraint_path"):
        run_inference(
            tmp_path / "x.fasta",
            output_dir=tmp_path / "out",
            contact_guidance_scale=0.5,
        )
