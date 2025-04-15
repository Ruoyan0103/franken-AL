import logging
from unittest import mock

import ase.calculators.lj
import numpy as np
from omegaconf import DictConfig
import pytest
from ase.build import molecule


from franken.mdgen.asemd import run_md_simulation



@pytest.fixture()
def atoms():
    return molecule("H2O")


class NaNCalculator(ase.calculators.lj.LennardJones):
    def calculate(self, atoms=None, properties=None, system_changes=...):
        super().calculate(atoms, properties, system_changes)
        self.results["forces"] = np.full_like(self.results["forces"], np.nan)


def test_nan_catcher(atoms, tmp_path, caplog):
    calc = NaNCalculator()
    atoms.calc = calc

    with caplog.at_level(logging.ERROR):
        with mock.patch.object(calc, "calculate", wraps=calc.calculate) as calc_mock:
            run_md_simulation(
                output_dir=tmp_path,
                integrator_cfg=DictConfig({
                    "_target_": "ase.md.langevin.Langevin",
                    "friction": 0.1,
                    "timestep": 0.01,
                    "temperature_K": 300,
                }),
                initial_velocity_cfg=DictConfig({
                    "_target_": "ase.md.velocitydistribution.MaxwellBoltzmannDistribution",
                    "temperature_K": 300
                }),
                initial_configuration=atoms,
                num_steps=20,
                save_every=1,
                log_every=1,
                seed=4,
            )
            calc_mock.assert_called_once()
    assert "Simulation failed after" in caplog.text
