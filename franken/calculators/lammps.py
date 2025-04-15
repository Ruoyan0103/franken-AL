import os
import argparse
from typing import Optional
import torch
from e3nn.util import jit

from franken.data.base import Configuration
from franken.rf.model import FrankenPotential


@jit.compile_mode("script")
class LammpsFrankenCalculator(torch.nn.Module):
    def __init__(
        self,
        franken_model: FrankenPotential,
    ):
        super().__init__()

        self.model = franken_model
        self.register_buffer("atomic_numbers", self.model.gnn.base_model.atomic_numbers)
        self.register_buffer("r_max", self.model.gnn.base_model.r_max)
        self.register_buffer(
            "num_interactions", self.model.gnn.base_model.num_interactions
        )
        # this attribute is used for dtype detection in LAMMPS-MACE.
        # See: https://github.com/ACEsuit/lammps/blob/mace/src/ML-MACE/pair_mace.cpp#314
        self.model.node_embedding = self.model.gnn.base_model.node_embedding

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: dict[str, torch.Tensor],
        local_or_ghost: torch.Tensor,
        compute_virials: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        # node_attrs is a one-hot representation of the atom types
        atom_nums = torch.nonzero(data["node_attrs"])[:, 1] + 1
        franken_data = Configuration(
            atom_pos=data["positions"],
            atomic_numbers=atom_nums,
            natoms=torch.tensor(
                len(atom_nums), dtype=torch.int32, device=atom_nums.device
            ).view(1),
            node_attrs=data["node_attrs"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
            unit_shifts=data["unit_shifts"],
        )
        energy, forces = self.model(franken_data)  # type: ignore
        # Kokkos doesn't like total_energy_local and only looks at node_energy.
        # We hack around this:
        node_energy = energy.repeat(len(atom_nums)).div(len(atom_nums))
        virials: Optional[torch.Tensor] = None
        if compute_virials:
            virials = torch.zeros((1, 3, 3), dtype=forces.dtype, device=forces.device)
        return {
            "total_energy_local": energy,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }


def create_lammps_model(model_path: str, rf_weight_id: int | None):
    franken_model = FrankenPotential.load(
        model_path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        rf_weight_id=rf_weight_id,
    )
    # NOTE:
    # Kokkos is hardcoded to double and will silently corrupt data if the model
    # does not use dtype double.
    franken_model = franken_model.double().to("cpu")
    lammps_model = LammpsFrankenCalculator(franken_model)
    lammps_model_compiled = jit.compile(lammps_model)

    save_path = f"{os.path.splitext(model_path)[0]}-lammps.pt"
    print(f"Saving compiled model to '{save_path}'")
    lammps_model_compiled.save(save_path)
    return save_path


def create_lammps_model_cli():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model to be converted to LAMMPS",
    )
    parser.add_argument(
        "--rf_weight_id",
        type=int,
        help="Head of the model to be converted to LAMMPS",
        default=None,
    )
    args = parser.parse_args()
    create_lammps_model(args.model_path, args.rf_weight_id)


if __name__ == "__main__":
    create_lammps_model_cli()
