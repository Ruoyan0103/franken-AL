# Running molecular dynamics with learned potentials through ASE

The scripts to run MD simulations with learned potentials are located in the `mdgen` subfolder,
and can be accessed using the `franken.mdgen` command.

We briefly detail the different 'calculators' which can be used as potential functions with our
script, and then go deeper into the configuration of a MD run.

## Configuration

Configuration is through a set of [hydra](https://hydra.cc) configs, located at `mdgen/configs`.
The configuration is hierarchical, with the `master.yaml` file at the top level.
Typically you will want to start from one of the predefined 'experiment' configurations
(subdirectory `exp`) and modify it for your needs, either changing the file directly or
overriding it from the command line.

The `master.yaml` file does not contain relevant parameters that you might want to change, it
is only used to configure output paths or other run-time details.

The experiment configuration (which you **must** select at runtime by passing for
example `exp=franken-exp` on the commandline) allows you to configure all the details about
the MD run:
 1. The integrator
 2. The observables: quantities which are monitored throughout the run
 3. The calculator itself and its relevant parameters (very important for example
    is the path to the model which should be loaded)
 4. Other simulation parameters such as temperature, integration time-step, etc.

Some examples on how to override existing experiments from the command line

1. This overrides the default `bond_stability` observable with a different one
```bash
python franken/mdgen \
    exp=franken-exp \
    calculator.model_path=~/franken/experiments/TM23-Cr/SevenNet0/2048_rfs/gaussian/run_241121_141006_a34b29aa/best_ckpt.pt \
    initial_configuration=~/franken/datasets/TM23/Cr_cold_nequip_test.xyz \
    temperature_K=255 \
    num_replicas=1 \
    md_length_ns=0.1 \
    save_every=1000 \
    log_every=100 \
    'observables=[rdf_mae]' \
    observables.rdf_mae.observe_every=1000 \
    exp_name=tm23-cr-franken-sevennet
```