# Fitting a Franken ML Potential

To fit a `franken` potential we created a script which

1. Performs automatic hyperparameter selection 
2. Automatically parallelizes the training over multi-GPUs 
3. Can be fully configured with [hydra](https://hydra.cc/). 

```{danger}
The training script only supports CUDA devices. CPU training has not been tested.
```

## Configuring the training

The script, which can be executed via the `franken.autotune` command, has the following default configuration options

```yaml
seed: 1337 # Random Number Generator Seed for RF heads initialization

console_logging_level: WARNING # Logs to show in the console. The full logs are always saved to file in the run directory.

dataset:
  dataset_name: null  # Used to create the run directory
  train_path: null    # Required. Should be a .xyz file with a list of ASE atoms.
  test_path: null     # Optional
  val_path: null      # Optional
  train_subsample:
    num: null         # If not null, takes a random subsample of the training data
    rng_seed: null    # Random number generator seed to subsample the training data

trainer:
  _target_: franken.trainers.RandomFeaturesTrainer
  random_features_normalization: "leading_eig"
  save_every_model: false # If true saves a checkpoint for every trial, otherwise it saves only the best model.

franken:
  gnn_backbone_id: null     # See the "model registry" section in the docs.
  interaction_block: null   # GNN layer out of which the features are extracted.
  kernel_type: null         # Must be "poly" or "gaussian"
  jac_chunk_size: "auto"

hyperparameters: 
# Hyperparameters are split into groups. 
# Each hyperparameter should be an iterable, or must be insantiable as an iterable. 
# This should happen even if don't want to sweep over a specific parameter, see e.g. num_random_features.

  random_features:
    num_random_features: 256

  solver:
    L2_penalty:
      _target_: numpy.logspace
      start: -11
      stop: -6
      num: 10
    loss_lerp_weight: # loss = loss_lerp_weight * loss_energy + (1 - loss_lerp_weight) * loss_forces
      _target_: numpy.linspace
      start: 0.1
      stop: 1.0
      num: 10
```
```{hint}
If the `dataset.dataset_name` matches an ID in our internal dataset registry, the `{split}_path` folders belows can be left unspecified.
```

As `franken.autotune` configurations are handled by [hydra](https://hydra.cc/), each one of these parameters can be overridden from the CLI. For example, if we want to use the second layer of a `MACE-L0` backbone together with a `gaussian` head we can run
```bash
franken.autotune \
    dataset.dataset_name=mydataset \
    dataset.train_path=/my/dataset/file.xyz \
    franken.gnn_backbone_id=MACE-L0 \
    franken.kernel_type=gaussian \
    franken.interaction_block=2 \
    hyperparameters.random_features.num_random_features=1024 \
    +'hyperparameters.random_features.length_scale=[0.5, 1., 2.]'
```
where we fitted a model with 1024 random features and we sweeped the Gaussian kernel length scale on the values `[0.5, 1., 2.]`


Gaussian and Polynomial kernel accepts different hyperparameters, which should be defined within the `hyperparameters.random_features` configuration. The complete list, and their default values are reported in {ref}`the API reference <model_reference>`. 

Several configuration presets [are available](#presets). Using presets, the example above (with automatic length-scale selection) can be alternatively runned as:
```bash
franken.autotune \
    dataset.dataset_name=mydataset \
    dataset.train_path=/my/dataset/file.xyz \
    preset=gaussian-MACE-L0 \
    hyperparameters.random_features.num_random_features=1024
```

## Multi-GPU training
The `franken.autotune` script can parallelize the workload over multiple GPUs, if available. In this case, the script should be runned with `torchrun` as 

### Multi GPU
```bash
torchrun --standalone --nproc_per_node=<num_gpus> -m franken.autotune \
    dataset.dataset_name=mydataset \
    dataset.train_path=/my/dataset/file.xyz \
    preset=gaussian-MACE-L0 \
    hyperparameters.random_features.num_random_features=1024
```

(presets)=
## Presets

### `gaussian-MACE-L0`
```yaml
defaults:
  - default

franken:
  gnn_backbone_id: "MACE-L0"
  interaction_block: 2
  kernel_type: "gaussian"
  jac_chunk_size: "auto"

hyperparameters:
  random_features:
    num_random_features: 1024
    length_scale:
      _target_: franken.utils.misc.pdist_quantile
      quantiles: [0.1, 0.5, 0.7, 0.9, 0.99, 0.999]
```
### `gaussian-SchNet-All`

```yaml
# @package _global_

defaults:
  - default

franken:
  gnn_backbone_id: "SchNet-S2EF-OC20-All"
  interaction_block: 3
  kernel_type: "gaussian"
  jac_chunk_size: "auto"

hyperparameters:
  random_features:
    num_random_features: 1024
    length_scale:
      _target_: franken.utils.misc.pdist_quantile
      quantiles: [0.1, 0.5, 0.7, 0.9, 0.99, 0.999]
```

### `poly-MACE-L0`
```yaml
defaults:
  - default

franken:
  gnn_backbone_id: "MACE-L0"
  interaction_block: 2
  kernel_type: "poly"
  jac_chunk_size: "auto"

hyperparameters:
  random_features:
    num_random_features: 1024
    bias: [0.5, 1.0, 1.5]
    degree: [2, 3, 4, 5]
    length_scale: [0.1, 0.5, 1.0, 2.0, 10.]
```
### `poly-SchNet-All`

```yaml
# @package _global_
defaults:
  - default

franken:
  gnn_backbone_id: "SchNet-S2EF-OC20-All"
  interaction_block: 3
  kernel_type: "poly"
  jac_chunk_size: "auto"

hyperparameters:
  random_features:
    num_random_features: 1024
    bias: [0.5, 1.0, 1.5]
    degree: [2, 3, 4, 5]
    length_scale: [0.1, 0.5, 1.0, 2.0, 10.]
```
