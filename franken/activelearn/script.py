import os 
import torch
from franken.config import MaceBackboneConfig
from franken.backbones.utils import get_checkpoint_path
from mace.calculators import MACECalculator
from mace.calculators import mace_mp
from ase.io import read, write
from ase.optimize import BFGS
import numpy as np
from tqdm import tqdm
from franken.dft import VaspCalculator
from copy import deepcopy
from franken.autotune.cli import build_parser, parse_cli
import sys

def init_dataset(random_struct_file: str, 
                 relaxed_struct_file: str,
                 calculate_struct_file: str,
                 num_struct: int) -> list[object]:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gnn_config = MaceBackboneConfig(
        path_or_id="MACE-L0",
        interaction_block=2,
    )
    small_macemp0_path = get_checkpoint_path(gnn_config.path_or_id)
    small_macemp0_calc = mace_mp(small_macemp0_path, device=device, default_dtype="float32")
    os.system('rm -rfv '+relaxed_struct_file)
    structures = read(random_struct_file, index=':')
    for struct in structures[:num_struct]:
        struct.calc = small_macemp0_calc
        dyn = BFGS(struct)
        dyn.run(fmax=0.05)
        write(relaxed_struct_file, struct, append=True)
    # struct_list = read(relaxed_struct_file, index=':')
    # return struct_list

    gnn_config = MaceBackboneConfig(
        path_or_id="MACE-L1",
        interaction_block=2,
    )
    medium_macemp0_path = get_checkpoint_path(gnn_config.path_or_id)

    gnn_config = MaceBackboneConfig(
        path_or_id="MACE-L2",
        interaction_block=2,
    )
    large_macemp0_path = get_checkpoint_path(gnn_config.path_or_id)
    model_paths = [small_macemp0_path, medium_macemp0_path, large_macemp0_path]
    mace_calcs = MACECalculator(model_paths=model_paths, device=device, default_dtype="float32")

    traj = read(relaxed_struct_file, index=":")
    variances = []
    for at in tqdm(traj):
        at.calc = mace_calcs
        engs = at.get_potential_energies()
        # at.info['energy_mace_1'] = engs[0]
        # at.info['energy_mace_2'] = engs[1]
        # at.info['energy_mace_3'] = engs[2]
        at.info['variance'] = np.std(engs)
        variances.append(at.info['variance']/len(at))
    avg_var = np.mean(variances)
    struct_list = []
    for idx, var in enumerate(variances):
        if var > avg_var:
            write(calculate_struct_file, structures[idx], append=True)
            struct_list.append(structures[idx])
    print(f"Selected {len(struct_list)} structures for DFT.")
    return struct_list

def label_init_dataset(
        random_struct_file: str, 
        relaxed_struct_file: str,
        calculated_struct_file: str,
        calculation_dir: str,
        labeled_struct_file: str,
        num_struct: int):
    vasp_obj = VaspCalculator()
    os.makedirs(calculation_dir, exist_ok=True)
    calc = vasp_obj.calculator(calculation_dir=calculation_dir)
    relaxed_struct_list = init_dataset(random_struct_file, relaxed_struct_file, calculated_struct_file, num_struct)
    for struct in relaxed_struct_list[:num_struct]: 
        struct.calc = calc
        energy = struct.get_potential_energy(force_consistent=True) # free energy
        forces = struct.get_forces()
        stress = struct.get_stress(voigt=False)
        v = struct.get_volume()
        struct.info['virial'] = v * -stress
        del struct.calc.results['dipole']
        del struct.calc.results['magmom']
        del struct.calc.results['magmoms']
        del struct.calc.results['energy']
        del struct.calc.results['free_energy']
        del struct.calc.results['forces']
        struct.info['energy_dft'] = energy
        struct.arrays['forces_dft'] = forces

        struct_add = deepcopy(struct)
        write(labeled_struct_file, struct_add, append=True)
    print("Calculation finished and dataset labeled.")    

def cli_entry_point():
    args = parse_cli(sys.argv[1:])
    label_init_dataset(args)

if __name__ == "__main__":
    cli_entry_point()

        

    



