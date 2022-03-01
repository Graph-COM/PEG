# PEG
The official implementation of PEG for our paper: Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks. (URL)

## Introduction
In this work we propose a principled way of using PE to build more powerful GNNs. \The key idea is to use separate channels to update the original node features and positional features. The GNN architecture keeps not only permutation equivariant w.r.t. node features but also rotation equivariant w.r.t. positional features. This idea applies to a broad range of PE techniques that can be formulated as matrix factorization such as Laplacian Eigenmap (LE and Deepwalk~\citep{perozzi2014deepwalk}. We design a GNN layer \proj that satisfies such requirements. Figure 1 shows the architecture of PEG.

<p align="center"><img src="./data/PEG.png" width=85% height=85%></p>
<p align="center"><em>Figure 1.</em> The architecture of PEG.</p>

## Code

#### Required Packages
0. Python 3.7
1. PyTorch 1.8.1 Cuda 11.2
2. NetworkX 2.6.2
3. dgl 0.6.1
4. Numpy 1.20.3
5. Scipy 1.6.2
6. Scikit-Learn 0.24.1
7. Tensorflow 1.14.0
8. torch-geometric 1.7.2

#### Run
task1 is traditional link prediction and task2 is domain shift link prediction.
##### Task1
```bash
cd task1
```
PEG-DW using node feature on cora
```bash
python main.py --PE_method DW --dataset cora --feature_type N
```
PEG-LE+ using constant feature on cora
```bash
python main.py --PE_method LE --dataset cora --feature_type C --random_partition
```

For ogbl-ddi and ogbl-collab
```bash
cd OGB
```

##### Task2
```bash
cd task2
```
PEG-DW using node feature on cora->citeseer
```bash
python main.py --PE_method DW --source_dataset cora --target_dataset citeseer --feature_type N
```
PEG-LE+ using constant feature on cora->citeseer
```bash
python main.py --PE_method LE --source_dataset cora --target_dataset citeseer --feature_type C --random_partition
```
For PPI dataset
```bash
python PPI.py --PE_method LE --feature_type N
```
