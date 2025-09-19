from ase.calculators.vasp import Vasp

class VaspCalculator:
    def __init__(self):
        pass
    
    def calculator(
        self, 
        calculation_dir: str,
        encut: int=500,
        kx: int=8,
        ky: int=8,
        kz: int=8,
        ismear: int=0, 
        sigma: int=0.05,
        # kspacing: float=0.2
        ):
        calc = Vasp(               
            command='srun vasp_std',
            istart = 0,         # (Default = 0; from scratch) 
            xc     = 'PBE',     # (Exchange-correlation functional) 
            setups = {'Ge':'_d'},

            # Ionic:
            ibrion = -1,        # (Static calculation, default for nsw=0)
            nsw    = 0,         # (Max ionic steps) 

            # Electronic:
            ediff  = 1E-06,     # (SCF energy convergence; eV) --------------DIFFERENT-------------
            ismear = ismear,    # (Electronic temperature, Gaussian smearing)        *
            sigma  = sigma,     # (Smearing value in eV)                             *

            # Plane wave basis set:
            encut  = encut,       # (Default = largest ENMAX in the POTCAR file)     *
            prec   = 'Accurate',  # (Accurate forces are required)

            # Output features:
            lwave  = False,     # (WAVECAR is NOT written out)
            lcharg = False,     # (CHGCAR is NOT written out)

            # Parallelization:  
            # ncore  = 4,         # (Number of cores per k-point)

            directory = calculation_dir,
            kpts = (kx, ky, kz)   # ------------------ DIFFERENT-----------------
            # kspacing = kspacing,
            # gamma = True

        )
        return calc
        