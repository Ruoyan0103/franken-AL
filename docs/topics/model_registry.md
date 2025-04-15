(model-registry)=
# Backbones Registry

The available pre-trained GNNs can be listed by running `franken.backbones list`, and downloaded by:

```bash

   franken.backbones download <gnn_backbone_id>
```

As of today, the available models are:

```
                               DOWNLOADED MODELS                                
--------------------(/path/to/.franken/gnn_checkpoints)--------------------
MACE-L0 (MACE)                          
--------------------------------AVAILABLE MODELS--------------------------------
MACE-L1 (MACE)                                                                  
MACE-L2 (MACE)                                                                  
SchNet-S2EF-OC20-200k (fairchem)                                                
SchNet-S2EF-OC20-2M (fairchem)                                                  
SchNet-S2EF-OC20-20M (fairchem)                                                 
SchNet-S2EF-OC20-All (fairchem)                                                 
--------------------------------------------------------------------------------
```

One can get help by typing `franken.backbones --help`

```

NAME
    franken.backbones download - Download the model if it's not already present locally.

SYNOPSIS
    franken.backbones download GNN_BACKBONE_ID <flags>

DESCRIPTION
    Download the model if it's not already present locally.

POSITIONAL ARGUMENTS
    GNN_BACKBONE_ID
        Type: str

FLAGS
    -c, --cache_dir=CACHE_DIR
        Type: Optional[str | None]
        Default: None
```