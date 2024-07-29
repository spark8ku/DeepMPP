# D4C molecular property prediction
![License](https://img.shields.io/badge/license-MIT-red.svg)
![Version](https://img.shields.io/badge/version-0.3.0-brightgreen.svg)

## Introduction
This project is a deep learning application designed to predict molecular properties. The models implemented in this project feature interpretable and hierarchical architectures, including conventional graph convolutional models. D4CMPP is designed to be user-friendly and extremely convenient for deep learning applications.      

---
## Installation
```sh
$ pip install D4CMPP
```
It is recommended to install dgl and torch first with the appropriate versions for your system.   
Note that this package was developed with dgl\==2.3.0 and torch\==2.1.2.  

---
## How to start training
### Using module
1. Place the CSV file in your working directory
    - The SMILES of molecules should be in the "compound" column.
    - The SMILES of corresponding solvent should be in the "solvent" column. (optional)
    - There needs to be at least one molecular property for each corresponding molecule.
    - There are some inherent datasets such as 'Aqsoldb' and 'lipophilicity'. Use the 'test' dataset for test execution.
2. Import the train module of the D4CMPP package
```sh
> from D4CMPP import train
```
3. Check and choose the ID of the deep learning model in 'network_refer.yaml'. Or, give any invalid ID as the 'network' argument. This will show you the list of IDs of implemented networks.
    - Note that the networks incorporating the effect of solvents have the postfix 'wS', which indicates 'with solvent' (e.g. GCNwS).
```sh
> train(network="invalidID", data="test")
```
4. Write the network ID for the argument 'network', the CSV file name for the argument 'data', and the column name in the file with the target property for the argument 'target'.
```sh
> train(network="GCN", data="test", target=["Abs","Emi"])
```
5. Then, the graph will be generated, training will start, and the result of the training will be saved under the './_Models/' directory.
   
### Using source code
You can directly execute training with the source code as below, which is equal to an above example.
```sh
$ python main.py -n GCN -d test -t Abs,Emi
```
   
   
### Continue to training
You can load the saved model and continue training by

```sh
> train(LOAD_PATH="GCN_model_test_Abs,Emi_20240101_000000")
```
or
```sh
$ python main.py -l GCN_model_test_Abs,Emi_20240101_000000
```
   
### Transfer learning
You can try transfer learning from the pretrained model by
```sh
> train(TRANSFER_PATH="GCN_model_test_Abs,Emi_20240101_000000", data="Aqsoldb", target=["Solbility"] )
```
or
```sh
$ python main.py --transfer GCN_model_test_Abs,Emi_20240101_000000 -d Aqsoldb -t Solubility
```

---
## Analyzer
For additional tasks or analysis, the trained model can be loaded through the Analyzer module.
You should import an appropriate analyzer for your trained model. In general, MolAnalyzer supports every model.
```sh
from D4CMPP.Analyzer import MolAnalyzer
ma = MolAnalyzer("_Model/GCN_model_test_Abs,Emi_20240101_000000")
```
In general, Analyzer supports predicting external data.
```sh
ma.predict("CCC")
```

---

## Acknowledgements
This project includes code from the GC-GNN by Adem Rosenkvist Nielsen Aouichaoui (arnaou@kt.dtu.dk), licensed under the MIT License. 
- URL: [GC-GNN](https://github.com/gsi-lab/GC-GNN/tree/main )
- files: [AttentiveFP.py](https://github.com/gsi-lab/GC-GNN/blob/main/networks/AttentiveFP.py),  [DMPNN.py](https://github.com/gsi-lab/GC-GNN/blob/main/networks/DMPNN.py)
- Description: The source codes were adopted from this project.   
Additionally, we acknowledge that various other codes and workflows in this project are based on or inspired by these projects. While many of the original codes were modified, we recognize that the coding style and some workflows were influenced by the corresponding projects.
