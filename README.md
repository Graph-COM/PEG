# PEG
code for paper 'Equivariant and Stable Positional Encoding for More Powerful Graph Neural Networks'

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
