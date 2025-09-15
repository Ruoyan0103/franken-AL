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

if __name__ == "__main__":
    supercell_size = 1
    num_struct = 10
    seed = 42
    structure_folder = os.path.join(file_dir, 'structures')
    
    rs_generator = RandomStructures(supercell_size=supercell_size, seed=seed)
    rs_generator.generate_multiple_structures(n_structures=num_struct, structure_folder=structure_folder)
    




     
        

   