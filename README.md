# Welcome DAPM-CDR
## Our DAPM-CDR is split into five stages:
 1. Drug representation learning
 2. Cell-line representation learning
 3. Cell-line drug response learning
 4. Prompt design
 5. Domain adaptation contrastive learning.
## Datasets
Our experiments were conducted on two public data sets: GDSC and PubChem.
```
PubChem: https://pubchem.ncbi.nlm.nih.gov/
GDSC: https://www.cancerrxgene.org/
```
## Requirements

```
python = 3.8
dgl-cuda10.2 = 0.7.1
numpy = 1.24.4
torch-scatter = 2.0.9
torch = 1.12.1+cu113
```

## Data Loading

1. Downloading the Required Datasets.
2. Data Loading:
```
dataloader.py
```
## Running the Main Program
```
main.py
```


