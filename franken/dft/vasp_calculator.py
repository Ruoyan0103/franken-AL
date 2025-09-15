from ase.calculators.vasp import Vasp

class VaspCalculator:
    def __init__(self):
        pass
    
    def calculator(self):
        calc = Vasp(               
            command='srun vasp_std',
            istart = 0,         # (Default = 0; from scratch) 
            setups={'Ge':''}

            # Ionic:
            ibrion = -1,        # (Static calculation, default for nsw=0)
            nsw    = 0,         # (Max ionic steps) 

            # Electronic:
            ediff  = 1E-07,     # (SCF energy convergence; eV) 
            ismear = 0,         # (Electronic temperature, Gaussian smearing)
            sigma  = 0.05,      # (Smearing value in eV)
            nelm   = 100,       # (Max SCF steps)   
            gga    = 'PE',      # (Exchange-correlation functional)

            # Plane wave basis set:
            encut  = 500,         # (Default = largest ENMAX in the POTCAR file)
            prec   = 'Accurate',  # (Accurate forces are required)
            lasph  = True,        # (Non-spherical elements included)

            # Reciprocal space
            kspacing = 0.15,    # (Smallest spacing between k points in 1/A)
            kgamma   = True,    # (Default = Gamma centered)

            # Parallelisation: 
            ncore  = 4,         # ()

            # Output features:
            lwave  = False,     # (WAVECAR is NOT written out)
            lcharg = False      # (CHGCAR is NOT written out)

        )
        return calc