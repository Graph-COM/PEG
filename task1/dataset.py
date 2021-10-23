
import os
from torch_geometric.data import InMemoryDataset, DataLoader, Batch
from torch_geometric import data as DATA
import torch
import numpy as np
import random
from utils import *

def create_dataloader(association, val_ratio = 0.1, test_ratio = 0.1, seed = 0):
    adj_triu = np.triu(association)
    none_zero_position=np.where(adj_triu==1)
    none_zero_row_index=none_zero_position[0]
    none_zero_col_index=none_zero_position[1]

    asso_index=[]
    for i in range(0,len(none_zero_row_index)):
        asso_index.append(i)

    test_index,val_index,train_index=data_split(asso_index, test_ratio, val_ratio)

    test_row_index=none_zero_row_index[test_index]
    test_col_index=none_zero_col_index[test_index]
    val_row_index=none_zero_row_index[val_index]
    val_col_index=none_zero_col_index[val_index]
    train_matrix=np.copy(association)
    train_matrix[test_row_index, test_col_index]=0
    train_matrix[test_col_index, test_row_index]=0
    train_matrix[val_row_index, val_col_index]=0
    train_matrix[val_col_index, val_row_index]=0

    np.random.seed(seed)
    
    neg_sampling_mat = np.copy(association)
    row,column = neg_sampling_mat.shape
    for i in range(row):
        for j in range(column):
            if i>=j:
                neg_sampling_mat[i,j] = 1
    
    
    zero_position = np.where(neg_sampling_mat == 0)
    negative_randomlist = [i for i in range(len(zero_position[0]))]
    random.shuffle(negative_randomlist)
    selected_negative = []
    for i in range(len(asso_index)):
        selected_negative.append(negative_randomlist[i])

    train_negative_index = selected_negative[:len(train_index)]
    val_negative_index = selected_negative[len(train_index):len(train_index)+len(val_index)]
    test_negative_index = selected_negative[len(train_index)+len(val_index):]

    id_train = []
    train_label=[]
    id_val = []
    val_label=[]
    id_test = []
    test_label=[]
    train_edge_index = []
    
    for i in range(len(train_index)):
        id_train.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_edge_index.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_edge_index.append([none_zero_col_index[train_index][i], none_zero_row_index[train_index][i]])
        train_label.append(1)
    for i in range(len(val_index)):
        id_val.append([val_row_index[i], val_col_index[i]])
        val_label.append(1)
    for i in range(len(test_index)):
        id_test.append([test_row_index[i],test_col_index[i]])
        test_label.append(1)
        
        
    for i in train_negative_index:
        id_train.append([zero_position[0][i],zero_position[1][i]])
        train_label.append(0)
    for i in val_negative_index:
        id_val.append([zero_position[0][i],zero_position[1][i]])
        val_label.append(0)
    for i in test_negative_index:
        id_test.append([zero_position[0][i],zero_position[1][i]])
        test_label.append(0)



    train_dataset = lkpDataset(root='data', dataset='data/' + '_train',id_map=id_train, label = train_label)
    val_dataset = lkpDataset(root='data', dataset='data/' + '_val',id_map=id_val, label = val_label)
    test_dataset = lkpDataset(root='data', dataset='data/' + '_test',id_map=id_test, label = test_label)
    return train_dataset, val_dataset, test_dataset, train_matrix, train_edge_index


def create_dataloader_plus(association, val_ratio = 0.1, test_ratio = 0.1, seed = 0):
    adj_triu = np.triu(association)
    none_zero_position=np.where(adj_triu==1)
    none_zero_row_index=none_zero_position[0]
    none_zero_col_index=none_zero_position[1]

    asso_index=[]
    for i in range(0,len(none_zero_row_index)):
        asso_index.append(i)

    test_index,val_index,train_index=data_split(asso_index, test_ratio, val_ratio)

    test_row_index=none_zero_row_index[test_index]
    test_col_index=none_zero_col_index[test_index]
    val_row_index=none_zero_row_index[val_index]
    val_col_index=none_zero_col_index[val_index]
    train_matrix=np.copy(association)
    train_matrix[test_row_index, test_col_index]=0
    train_matrix[test_col_index, test_row_index]=0
    train_matrix[val_row_index, val_col_index]=0
    train_matrix[val_col_index, val_row_index]=0

    np.random.seed(seed)
    
    neg_sampling_mat = np.copy(association)
    row,column = neg_sampling_mat.shape
    for i in range(row):
        for j in range(column):
            if i>=j:
                neg_sampling_mat[i,j] = 1
    
    
    zero_position = np.where(neg_sampling_mat == 0)
    negative_randomlist = [i for i in range(len(zero_position[0]))]
    random.shuffle(negative_randomlist)
    selected_negative = []
    for i in range(len(asso_index)):
        selected_negative.append(negative_randomlist[i])

    train_negative_index = selected_negative[:len(train_index)]
    val_negative_index = selected_negative[len(train_index):len(train_index)+len(val_index)]
    test_negative_index = selected_negative[len(train_index)+len(val_index):]

    id_train_positive = []
    id_train_negative = []
    train_label=[]
    id_val = []
    val_label=[]
    id_test = []
    test_label=[]
    train_edge_index = []
    
    for i in range(len(train_index)):
        id_train_positive.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_edge_index.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_edge_index.append([none_zero_col_index[train_index][i], none_zero_row_index[train_index][i]])
        train_label.append(1)
    for i in range(len(val_index)):
        id_val.append([val_row_index[i], val_col_index[i]])
        val_label.append(1)
    for i in range(len(test_index)):
        id_test.append([test_row_index[i],test_col_index[i]])
        test_label.append(1)
        
        
    for i in train_negative_index:
        id_train_negative.append([zero_position[0][i],zero_position[1][i]])
        train_label.append(0)
    for i in val_negative_index:
        id_val.append([zero_position[0][i],zero_position[1][i]])
        val_label.append(0)
    for i in test_negative_index:
        id_test.append([zero_position[0][i],zero_position[1][i]])
        test_label.append(0)



    #train_dataset = lkpDataset(root='data', dataset='data/' + '_train',id_map=id_train, label = train_label)
    val_dataset = lkpDataset(root='data', dataset='data/' + '_val',id_map=id_val, label = val_label)
    test_dataset = lkpDataset(root='data', dataset='data/' + '_test',id_map=id_test, label = test_label)
    return val_dataset, test_dataset, train_matrix, train_edge_index, id_train_positive, id_train_negative



def data_split(full_list, ratio1, ratio2, shuffle=True):

    n_total = len(full_list)
    offset1 = int(n_total * ratio1)
    offset2 = offset1 + int(n_total * ratio2)
    if n_total == 0 or offset1 < 1 or offset2 < 1:
        return [],[], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset1]
    sublist_2 = full_list[offset1:offset2]
    sublist_3 = full_list[offset2:]
    return sublist_1, sublist_2, sublist_3



class lkpDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='data',
                 id_map=None, transform=None,
                 pre_transform=None, label=None):

        super(lkpDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.id_map = id_map
        self.len = len(id_map)
        self.label = label

    @property
    def raw_file_names(self):
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + 'id.txt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self):
        pass

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        index_x = self.id_map[idx][0]
        index_y = self.id_map[idx][1]
        y = self.label[idx]
        return y, [index_x, index_y]
    
