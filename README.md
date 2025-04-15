# Franken

## Live documentation
[The docs are live at this address](http://35.223.78.91/index.html). They'll be updated every 6 hours.

## Building the documentation
It is very easy to build and view locally.

1. **Preliminary step to do only once:** install docs dependencies
  ```bash
  cd docs
  pip install -r requirements.txt
  ```
2. Make the docs
  ```make html```

3. Visualize the docs; open the file `_build/html/index.html` with any browser. If on VSCode you can use the `live-server` extension to visualize it.

## Installation
To install `franken` you need to setup the `conda` environment first and then install the `franken` package cloned from source. The installation comes barebone, and any pre-trained ML potential has its own installation instructions.

### Installation steps

1. Clone the `franken` repository locally
    ```bash
    git clone git@github.com:CSML-IIT-UCL/mlpot_transfer.git franken
    cd franken
    ```
2. Create an environment in which to install `franken`
    The default environment will use CUDA 12.4, if you need a different version you will have to edit `pytorch-cuda` version accordingly.
    ```bash
    conda env create -f env.yml
    ```
    or download the development environment if you wish to install optional development dependencies as well
    ```bash
    conda env create -f env-dev.yml
    ```
3. Activate the environment
    ```bash
    conda activate franken
    ```
4. `pip`-install `franken` locally
    ```bash
    pip install .
    ```
    or use the `-e` (editable) flag for development
    ```bash
    pip install -e .
    ```
## Supported pre-trained models
### MACE MP0
We support [`MACE-MP0`](https://arxiv.org/abs/2401.00096) by Batatia et al. Additional informations on the pre-training of `MACE-MP0` are available on its [HuggingFace model card](https://huggingface.co/cyrusyc/mace-universal). To use `MACE-MP0` as a backbone for `franken` just `pip`-install `mace-torch` in `franken`'s environment
```bash
pip install mace-torch
```


### SchNet OC20 (fairchem, formerly OCP)
We support the [SchNet model](https://arxiv.org/abs/1706.08566) by Schütt et al. as implemented in the [`fairchem`](https://fair-chem.github.io/) library by Meta's FAIR. The pre-training was done on the [Open Catalyst dataset](https://fair-chem.github.io/core/datasets/oc20.html). To use it as a backbone for `franken`, install the `fairchem` library and the `torch_geometric` dependencies
```bash
pip install torch_geometric torch_scatter torch_sparse torch_cluster fairchem-core
```
> [!NOTE]
> The `mace-torch` package requires an old version of `e3nn` which **may** conflict with `fairchem-core`, see [this relevant issue](https://github.com/ACEsuit/mace/issues/555) . If this is occurring, simply upgrade `e3nn` by running `pip install -U e3nn`.


## Other Resources

### Notes available at this [overleaf link](https://www.overleaf.com/4172646251vhtykqjwrqpb#6ea18b).

### Google Cloud Platform bucket

The `franken-ml` bucket is open access.