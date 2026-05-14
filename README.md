# ConsCF

ConsCF is a RecBole-based recommendation model that introduces a consistency-style objective into flow-based collaborative filtering. It uses an EMA teacher, a boundary-constrained prediction function, and a masked-history prior.

## Project Structure

```text
.
├── run.py
├── conscf.yaml
├── README.md
└── model/
    ├── __init__.py
    └── conscf.py
```

`model/__init__.py` should contain:

```python
from .conscf import ConsCF
```

## Requirements

```bash
pip install torch recbole pyyaml numpy pandas scipy
```

Use versions compatible with your local RecBole installation.

## Dataset

Prepare the dataset in RecBole format, for example:

```text
dataset/
└── ml-1m/
    └── ml-1m.inter
```

The interaction file should contain the fields specified in `conscf.yaml`, such as `user_id`, `item_id`, `rating`, and `timestamp`.

## RecBole Dataloader Registration

ConsCF should use the autoencoder-style dataloader in RecBole.

Open:

```text
recbole/data/utils.py
```

Find the `register_table` inside `get_dataloader`, and add:

```python
"ConsCF": _get_AE_dataloader,
```

The modified part should look like:

```python
register_table = {
    "MultiDAE": _get_AE_dataloader,
    "MultiVAE": _get_AE_dataloader,
    "MacridVAE": _get_AE_dataloader,
    "CDAE": _get_AE_dataloader,
    "ENMF": _get_AE_dataloader,
    "RaCT": _get_AE_dataloader,
    "RecVAE": _get_AE_dataloader,
    "DiffusionRec": _get_AE_dataloader,
    "LDiffRec": _get_AE_dataloader,
    "FlowCF": _get_AE_dataloader,
    "ConsCF": _get_AE_dataloader,
}
```

## Run

```bash
python run.py --config conscf.yaml
```

The script loads the configuration, builds the RecBole dataset, trains `ConsCF`, and reports Recall and NDCG on the test set.

## Model

ConsCF contains three main components:

- a boundary-constrained flow model;
- an EMA teacher for consistency regularization;
- a masked-history prior for constructing partially observed user histories.

The training objective combines reconstruction loss and consistency loss. In this implementation, the consistency weight is fixed to `1.0`.
