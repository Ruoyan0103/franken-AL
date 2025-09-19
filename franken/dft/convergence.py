import os
import numpy as np
from ase import Atoms
from ase.io import write

AVOGADRO = 6.02214076e23
ANGSTROM3_TO_CM3 = 1e-24

file_dir = os.path.dirname(__file__) 

class RandomStructures:
    def __init__(
        self, 
        element: str = "Ge",
        mass: float = 72.64, # atomic mass
        supercell_size: int = 1, # supercell size
        unit_box_range: tuple[float, float] = (3, 6), # in Angstrom
        density_range: tuple[float, float] = (5, 6), # in g/cm3
        r_min: float = 2.11, # val der Waals radius
        seed = None):
        self.element = element
        self.mass = mass
        self.box_range = tuple(x * supercell_size for x in unit_box_range)
        self.density_range = density_range
        self.r_min = r_min
        self.rng = np.random.default_rng(seed)

    def _generate_one_structure(self) -> object:
        box = self.rng.uniform(self.box_range[0], self.box_range[1], 3)
        density = self.rng.uniform(self.density_range[0], self.density_range[1])
        volume = np.prod(box) * ANGSTROM3_TO_CM3
        n_atoms = int(round(density * volume *  AVOGADRO / self.mass))
        print(f"box: {box}, volume: {volume:.3e} cm3, density: {density:.2f} g/cm3, n_atoms: {n_atoms}")
        positions = []
        while len(positions) < n_atoms:
            candidate = self.rng.uniform(0, 1, 3) * box
            if positions:
                distances = np.linalg.norm(np.array(positions) - candidate, axis=1)
                if np.any(distances < self.r_min):
                    continue
            positions.append(candidate)
        structure = Atoms(f'{self.element}{n_atoms}', positions=positions, cell=box, pbc=True)
        return structure
        
    def generate_multiple_structures(
        self, 
        n_structures: int,
        structure_folder: str) -> list[object]:
        if not os.path.exists(structure_folder):
            os.makedirs(structure_folder)
        structures = []
        n = 0
        while n < n_structures:
            structure = self._generate_one_structure()
            if structure is not None:
                structures.append(structure)
                n += 1
                file = os.path.join(structure_folder, f"rs-{n}.data")
                write(file, structure, format="vasp")
        return structures



import os 
import shutil
import numpy as np
from matplotlib import pyplot as plt
from vasp_calculator import VaspCalculator
from ase import Atoms
from ase.build import bulk

class TestConvergence:
    def __init__(self):
        pass 
    
    def _get_time(self, outcar_file: str):
        with open(outcar_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if 'Elapsed time (sec):' in line:
                time = float(line.split()[-1])
                return time
        return None
        
    def test_encut_vs_energy(
        self,
        structure: object, 
        encut_values: list,
        kx: int=3,
        ky: int=3,
        kz: int=3,
        calculation_dir='encut_test'):
        if not os.path.exists(calculation_dir):
            os.makedirs(calculation_dir)
        structure = structure 
        encut_values = encut_values
        energy_diffs = []
        times = []
        pre_energy = 0 
        for encut in encut_values:
            vasp_obj = VaspCalculator()
            calc = vasp_obj.calculator(encut=encut, calculation_dir=calculation_dir, kx=kx, ky=ky, kz=kz)
            structure.calc = calc
            energy = structure.get_potential_energy() / len(structure)
            energy_diff = energy - pre_energy
            pre_energy = energy
            energy_diffs.append(abs(energy_diff)*1e3)
            # shutil.copy(os.path.join(calculation_dir, 'OUTCAR'), \
            #             os.path.join(calculation_dir, f'OUTCAR_encut_{encut}'))
            time = self._get_time(os.path.join(calculation_dir, 'OUTCAR'))
            times.append(time)
            print(f'ENCUT: {encut} finished.')
        
        fig, ax1 = plt.subplots(figsize=(6,4))
        
        ax1.plot(encut_values[1:], energy_diffs[1:], marker='o', color='b', label='Energy diff')
        ax1.axhline(y=1, color='r', linestyle='--', label='Convergence threshold (1 meV/atom)')
        ax1.set_xlabel('ENCUT (eV)')
        ax1.set_ylabel('Energy diff (meV/atom)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        
        ax2 = ax1.twinx()
        ax2.plot(encut_values[1:], times[1:], marker='x', color='g', label='Calculation time')
        ax2.set_ylabel('Time (s)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(calculation_dir, 'encut_vs_energy_time.png'), dpi=300)

    def test_kpoints_vs_energy(
        self,
        structure: object, 
        kx: int=3,
        ky: int=3,
        kz: int=3,
        calculation_dir='kpoints_test'):
        if not os.path.exists(calculation_dir):
            os.makedirs(calculation_dir)
        structure = structure
        energy_diffs = []
        times = []
        pre_energy = 0
        kpoints = list(range(8, kx+1))
        for k in kpoints:
            vasp_obj = VaspCalculator()
            calc = vasp_obj.calculator(kx=k, ky=k, kz=k, calculation_dir=calculation_dir)
            structure.calc = calc
            energy = structure.get_potential_energy() / len(structure)
            energy_diff = energy - pre_energy
            pre_energy = energy
            energy_diffs.append(abs(energy_diff)*1e3)
            time = self._get_time(os.path.join(calculation_dir, 'OUTCAR'))
            times.append(time)
            print(f'KPOINTS: {k}x{k}x{k} finished.')

        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(kpoints[1:], energy_diffs[1:], marker='o', color='b', label='Energy diff')
        ax1.axhline(y=1, color='r', linestyle='--', label='Convergence threshold (1 meV/atom)')
        ax1.set_xlabel('KPOINTS (kx=kx=k)')
        ax1.set_ylabel('Energy diff (meV/atom)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(kpoints[1:], times[1:], marker='x', color='g', label='Calculation time')
        ax2.set_ylabel('Time (s)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(calculation_dir, 'kpoints_vs_energy_time.png'), dpi=300)

    def test_sigma_vs_energy(
            self,
            structure: object,
            sigma_values: list,
            calculation_dir='sigma_test'
            ):
        if not os.path.exists(calculation_dir):
            os.makedirs(calculation_dir)
        structure = structure
        energy_diffs = []
        times = []
        pre_energy = 0
        for sigma in sigma_values:
            vasp_obj = VaspCalculator()
            calc = vasp_obj.calculator(sigma=sigma, calculation_dir=calculation_dir)
            structure.calc = calc
            energy = structure.get_potential_energy() / len(structure)
            energy_diff = energy - pre_energy
            pre_energy = energy
            energy_diffs.append(abs(energy_diff)*1e3)
            time = self._get_time(os.path.join(calculation_dir, 'OUTCAR'))
            times.append(time)
            print(f'SIGMA: {sigma} finished.')
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(sigma_values[1:], energy_diffs[1:], marker='o', color='b', label='Energy diff')
        ax1.axhline(y=1, color='r', linestyle='--', label='Convergence threshold (1 meV/atom)')
        ax1.set_xlabel('SIGMA (eV)')
        ax1.set_ylabel('Energy diff (meV/atom)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(sigma_values[1:], times[1:], marker='x', color='g', label='Calculation time')
        ax2.set_ylabel('Time (s)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(calculation_dir, 'sigma_vs_energy_time.png'), dpi=300)

    def test_kspacing_vs_energy(
        self,
        structure: object,
        kspacing_values: list,
        calculation_dir='kspacing_test'
        ):
        if not os.path.exists(calculation_dir):
            os.makedirs(calculation_dir)
        structure = structure   
        energy_diffs = []
        times = []
        pre_energy = 0
        for kspacing in kspacing_values:
            vasp_obj = VaspCalculator()
            calc = vasp_obj.calculator(kspacing=kspacing, calculation_dir=calculation_dir)
            structure.calc = calc
            energy = structure.get_potential_energy() / len(structure)
            energy_diff = energy - pre_energy
            pre_energy = energy
            energy_diffs.append(abs(energy_diff)*1e3)
            time = self._get_time(os.path.join(calculation_dir, 'OUTCAR'))
            times.append(time)
            print(f'KSPACING: {kspacing} finished.')
        fig, ax1 = plt.subplots(figsize=(6,4))
        ax1.plot(kspacing_values[1:], energy_diffs[1:], marker='o', color='b', label='Energy diff')
        ax1.axhline(y=1, color='r', linestyle='--', label='Convergence threshold (1 meV/atom)')
        ax1.set_xlabel('KSPACING (1/Angstrom)')
        ax1.set_ylabel('Energy diff (meV/atom)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True)
        ax2 = ax1.twinx()
        ax2.plot(kspacing_values[1:], times[1:], marker='x', color='g', label='Calculation time')
        ax2.set_ylabel('Time (s)', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(calculation_dir, 'kspacing_vs_energy_time.png'), dpi=300)
    
if __name__ == "__main__":
    # supercell_size = 1
    # num_struct = 1
    # seed = 42
    # structure_folder = os.path.join(file_dir, 'structures')
    
    # rs_generator = RandomStructures(supercell_size=supercell_size, seed=seed)
    # structure = rs_generator._generate_one_structure()
    structure = bulk('Ge', 'diamond', a=5.76, cubic=True) 
    # encut_values = np.arange(300, 900, 50)
    tester = TestConvergence()
    # tester.test_encut_vs_energy(structure, encut_values)
    # tester.test_kpoints_vs_energy(structure, kx=10, ky=10, kz=10)
    # kspacing_values = np.arange(0.1, 0.5, 0.05)
    # tester.test_kspacing_vs_energy(structure, kspacing_values)
    sigma_values = np.arange(0.01, 0.35, 0.04)
    tester.test_sigma_vs_energy(structure, sigma_values)

