import argparse
import dgl
from utils import *
from dataset import *
from model import *
from train import *
from ge import DeepWalk
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn import random_projection
import json
import os
import enum

import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import torch
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as sp
from sklearn.preprocessing import normalize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Supported datasets - only PPI
class DatasetType(enum.Enum):
    PPI = 0

    
class GraphVisualizationTool(enum.Enum):
    IGRAPH = 0


# We'll be dumping and reading the data from this directory
DATA_DIR_PATH = os.path.join(os.getcwd(), 'PPI_data')
PPI_PATH = os.path.join(DATA_DIR_PATH, 'ppi')
PPI_URL = 'https://data.dgl.ai/dataset/ppi.zip'  # preprocessed PPI data from Deep Graph Library

#
# PPI specific constants
#

PPI_NUM_INPUT_FEATURES = 50
PPI_NUM_CLASSES = 121

def json_read(path):
    with open(path, 'r') as file:
        data = json.load(file)

    return data

def load_graph_data(training_config, device):
    dataset_name = training_config['dataset_name'].lower()
    should_visualize = training_config['should_visualize']

    if dataset_name == DatasetType.PPI.name.lower():  # Protein-Protein Interaction dataset

        # Instead of checking PPI in, I'd rather download it on-the-fly the first time it's needed (lazy execution ^^)
        if not os.path.exists(PPI_PATH):  # download the first time this is ran
            os.makedirs(PPI_PATH)

            # Step 1: Download the ppi.zip (contains the PPI dataset)
            zip_tmp_path = os.path.join(PPI_PATH, 'ppi.zip')
            download_url_to_file(PPI_URL, zip_tmp_path)

            # Step 2: Unzip it
            with zipfile.ZipFile(zip_tmp_path) as zf:
                zf.extractall(path=PPI_PATH)
            print(f'Unzipping to: {PPI_PATH} finished.')

            # Step3: Remove the temporary resource file
            os.remove(zip_tmp_path)
            print(f'Removing tmp file {zip_tmp_path}.')

        # Collect train/val/test graphs here
        edge_index_list = []
        node_features_list = []
        node_labels_list = []

        # Dynamically determine how many graphs we have per split (avoid using constants when possible)
        num_graphs_per_split_cumulative = [0]

        # Small optimization "trick" since we only need test in the playground.py
        splits = ['test'] if training_config['ppi_load_test_only'] else ['train', 'valid', 'test']

        for split in splits:
            # PPI has 50 features per node, it's a combination of positional gene sets, motif gene sets,
            # and immunological signatures - you can treat it as a black box (I personally have a rough understanding)
            # shape = (NS, 50) - where NS is the number of (N)odes in the training/val/test (S)plit
            # Note: node features are already preprocessed
            node_features = np.load(os.path.join(PPI_PATH, f'{split}_feats.npy'))

            # PPI has 121 labels and each node can have multiple labels associated (gene ontology stuff)
            # SHAPE = (NS, 121)
            node_labels = np.load(os.path.join(PPI_PATH, f'{split}_labels.npy'))

            # Graph topology stored in a special nodes-links NetworkX format
            nodes_links_dict = json_read(os.path.join(PPI_PATH, f'{split}_graph.json'))
            # PPI contains undirected graphs with self edges - 20 train graphs, 2 validation graphs and 2 test graphs
            # The reason I use a NetworkX's directed graph is because we need to explicitly model both directions
            # because of the edge index and the way GAT implementation #3 works
            collection_of_graphs = nx.DiGraph(json_graph.node_link_graph(nodes_links_dict))
            # For each node in the above collection, ids specify to which graph the node belongs to
            graph_ids = np.load(os.path.join(PPI_PATH, F'{split}_graph_id.npy'))
            num_graphs_per_split_cumulative.append(num_graphs_per_split_cumulative[-1] + len(np.unique(graph_ids)))

            # Split the collection of graphs into separate PPI graphs
            for graph_id in range(np.min(graph_ids), np.max(graph_ids) + 1):
                mask = graph_ids == graph_id  # find the nodes which belong to the current graph (identified via id)
                graph_node_ids = np.asarray(mask).nonzero()[0]
                graph = collection_of_graphs.subgraph(graph_node_ids)  # returns the induced subgraph over these nodes
                print(f'Loading {split} graph {graph_id} to CPU. '
                      f'It has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.')

                # shape = (2, E) - where E is the number of edges in the graph
                # Note: leaving the tensors on CPU I'll load them to GPU in the training loop on-the-fly as VRAM
                # is a scarcer resource than CPU's RAM and the whole PPI dataset can't fit during the training.
                edge_index = torch.tensor(list(graph.edges), dtype=torch.long).transpose(0, 1).contiguous()
                edge_index = edge_index - edge_index.min()  # bring the edges to [0, num_of_nodes] range
                edge_index_list.append(edge_index)
                # shape = (N, 50) - where N is the number of nodes in the graph
                node_features_list.append(torch.tensor(node_features[mask], dtype=torch.float))
                # shape = (N, 121), BCEWithLogitsLoss doesn't require long/int64 so saving some memory by using float32
                node_labels_list.append(torch.tensor(node_labels[mask], dtype=torch.float))

                if should_visualize:
                    plot_in_out_degree_distributions(edge_index.numpy(), graph.number_of_nodes(), dataset_name)
                    visualize_graph(edge_index.numpy(), node_labels[mask], dataset_name)

        #
        # Prepare graph data loaders
        #

        # Optimization, do a shortcut in case we only need the test data loader
        if training_config['ppi_load_test_only']:
            data_loader_test = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                batch_size=training_config['batch_size'],
                shuffle=False
            )
            return data_loader_test
        else:

            data_loader_train = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                node_labels_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                edge_index_list[num_graphs_per_split_cumulative[0]:num_graphs_per_split_cumulative[1]],
                batch_size=training_config['batch_size'],
                shuffle=False
            )

            data_loader_val = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                node_labels_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                edge_index_list[num_graphs_per_split_cumulative[1]:num_graphs_per_split_cumulative[2]],
                batch_size=training_config['batch_size'],
                shuffle=False  # no need to shuffle the validation and test graphs
            )

            data_loader_test = GraphDataLoader(
                node_features_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                node_labels_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                edge_index_list[num_graphs_per_split_cumulative[2]:num_graphs_per_split_cumulative[3]],
                batch_size=training_config['batch_size'],
                shuffle=False
            )

            return data_loader_train, data_loader_val, data_loader_test
    else:
        raise Exception(f'{dataset_name} not yet supported.')

class GraphDataLoader(DataLoader):
    """
    When dealing with batches it's always a good idea to inherit from PyTorch's provided classes (Dataset/DataLoader).

    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list, batch_size=1, shuffle=False):
        graph_dataset = GraphDataset(node_features_list, node_labels_list, edge_index_list)
        # We need to specify a custom collate function, it doesn't work with the default one
        super().__init__(graph_dataset, batch_size, shuffle, collate_fn=graph_collate_fn)


class GraphDataset(Dataset):
    """
    This one just fetches a single graph from the split when GraphDataLoader "asks" it

    """
    def __init__(self, node_features_list, node_labels_list, edge_index_list):
        self.node_features_list = node_features_list
        self.node_labels_list = node_labels_list
        self.edge_index_list = edge_index_list

    # 2 interface functions that need to be defined are len and getitem so that DataLoader can do it's magic
    def __len__(self):
        return len(self.edge_index_list)

    def __getitem__(self, idx):  # we just fetch a single graph
        return self.node_features_list[idx], self.node_labels_list[idx], self.edge_index_list[idx]


def graph_collate_fn(batch):
    """
    The main idea here is to take multiple graphs from PPI as defined by the batch size
    and merge them into a single graph with multiple connected components.

    It's important to adjust the node ids in edge indices such that they form a consecutive range. Otherwise
    the scatter functions in the implementation 3 will fail.

    :param batch: contains a list of edge_index, node_features, node_labels tuples (as provided by the GraphDataset)
    """

    edge_index_list = []
    node_features_list = []
    node_labels_list = []
    num_nodes_seen = 0

    for features_labels_edge_index_tuple in batch:
        # Just collect these into separate lists
        node_features_list.append(features_labels_edge_index_tuple[0])
        node_labels_list.append(features_labels_edge_index_tuple[1])

        edge_index = features_labels_edge_index_tuple[2]  # all of the components are in the [0, N] range
        edge_index_list.append(edge_index + num_nodes_seen)  # very important! translate the range of this component
        num_nodes_seen += len(features_labels_edge_index_tuple[1])  # update the number of nodes we've seen so far

    # Merge the PPI graphs into a single graph with multiple connected components
    node_features = torch.cat(node_features_list, 0)
    node_labels = torch.cat(node_labels_list, 0)
    edge_index = torch.cat(edge_index_list, 1)

    return node_features, node_labels, edge_index

config = {
    'dataset_name': DatasetType.PPI.name,
    'should_visualize': False,
    'batch_size': 1,
    'ppi_load_test_only': False  # small optimization for loading test graphs only, we won't use it here
}

data_loader_train, data_loader_val, data_loader_test = load_graph_data(config, device)


node_features_list, node_labels_list, edge_index_list = ([], [], [])
for i, (node_features, node_labels, edge_index) in enumerate(data_loader_train):
    node_features_list.append(node_features)
    node_labels_list.append(node_labels)
    edge_index_list.append(edge_index)
for i, (node_features, node_labels, edge_index) in enumerate(data_loader_val):
    node_features_list.append(node_features)
    node_labels_list.append(node_labels)
    edge_index_list.append(edge_index)
for i, (node_features, node_labels, edge_index) in enumerate(data_loader_test):
    node_features_list.append(node_features)
    node_labels_list.append(node_labels)
    edge_index_list.append(edge_index)

def build_adj_from_edgeind(node_num, edge_index):
    adj = np.zeros((node_num, node_num))
    for i in range(len(edge_index[0])):
        adj[edge_index[0][i]][edge_index[1][i]] = 1
    return adj


def create_dataloader_train_val(association, ratio = 0.1, seed = 0):
    association = association - np.diag(np.ones(len(association[0])))
    adj_triu = np.triu(association)
    none_zero_position=np.where(adj_triu==1)
    none_zero_row_index=none_zero_position[0]
    none_zero_col_index=none_zero_position[1]

    asso_index=[]
    for i in range(0,len(none_zero_row_index)):
        asso_index.append(i)

    test_index,train_index=data_split(asso_index, 0.1)


    test_row_index=none_zero_row_index[test_index]
    test_col_index=none_zero_col_index[test_index]
    train_matrix=np.copy(association)
    train_matrix[test_row_index, test_col_index]=0
    train_matrix[test_col_index, test_row_index]=0

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
    test_negative_index = selected_negative[len(train_index):len(train_index)+len(test_index)]

    id_train = []
    train_label=[]
    id_test = []
    test_label=[]
    train_edge_index = []
    
    for i in range(len(train_index)):
        id_train.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_edge_index.append([none_zero_row_index[train_index][i], none_zero_col_index[train_index][i]])
        train_edge_index.append([none_zero_col_index[train_index][i], none_zero_row_index[train_index][i]])
        train_label.append(1)
    for i in range(len(test_index)):
        id_test.append([test_row_index[i], test_col_index[i]])
        test_label.append(1)

        
        
    for i in train_negative_index:
        id_train.append([zero_position[0][i],zero_position[1][i]])
        train_label.append(0)
    for i in test_negative_index:
        id_test.append([zero_position[0][i],zero_position[1][i]])
        test_label.append(0)




    test_dataset = lkpDataset(root='data', dataset='data/' + '_train',id_map=id_test, label = test_label)
    return test_dataset, train_matrix, train_edge_index


train_num = random.randint(0,23)
val_num = random.randint(0,23)
test_num = random.randint(0,23)
train_node_features, train_node_labels, train_edge_index = (node_features_list[train_num], node_labels_list[train_num], edge_index_list[train_num])

print('*' * 20)
print("training data")
print(train_node_features.shape, train_node_features.dtype)
print(train_node_labels.shape, train_node_labels.dtype)
print(train_edge_index.shape, train_edge_index.dtype)

val_node_features, val_node_labels, val_edge_index = (node_features_list[val_num], node_labels_list[val_num], edge_index_list[val_num])

print('*' * 20)
print("validation data")
print(val_node_features.shape, val_node_features.dtype)
print(val_node_labels.shape, val_node_labels.dtype)
print(val_edge_index.shape, val_edge_index.dtype)

test_node_features, test_node_labels, test_edge_index = (node_features_list[test_num], node_labels_list[test_num], edge_index_list[test_num])

print('*' * 20)
print("testing data")
print(test_node_features.shape, test_node_features.dtype)
print(test_node_labels.shape, test_node_labels.dtype)
print(test_edge_index.shape, test_edge_index.dtype)
# Data settings
parser = argparse.ArgumentParser(description='PEG')
#parser.add_argument('--source_dataset', type=str, default='cora')
#parser.add_argument('--target_dataset', type=str, default='citeseer')
parser.add_argument('--PE_method', type=str, default="DW")
parser.add_argument('--feature_type', type=str, default="N")
# GNN settings
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--PE_dim', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=128)

parser.add_argument('--batch_size', type=int, default=128)
# Training settings
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--epochs', type=int, default=100)
#parser.add_argument('--runs', type=int, default=1)
#parser.add_argument('--random_partition', action='store_true')

args = parser.parse_args()

sum_metric = np.zeros((1, 2))
auc = []
ap = []
for i in [115,105,100]:

    adj_train = build_adj_from_edgeind(len(train_node_labels), train_edge_index)
    train_dataset, train_matrix, train_edge_index = create_dataloader_train_val(adj_train, 
                                                                                    ratio = 0.1, 
                                                                                    seed = i)
    print("train data done!")
    #print("There are" + str(len(train_edge_index)) + "in training dataset")
    adj_val = build_adj_from_edgeind(len(val_node_labels), val_edge_index)
    val_dataset, val_matrix, val_edge_index = create_dataloader_train_val(adj_val, ratio = 0.1, seed = i)
    
    print("validation data done!")
    
    adj_test = build_adj_from_edgeind(len(test_node_labels), test_edge_index)
    test_dataset, test_matrix, test_edge_index = create_dataloader_train_val(adj_test, ratio = 0.1, seed = i)
    
    print("test data done!")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    if args.feature_type == 'N':
        features_train = train_node_features
        features_val = val_node_features
        features_test = test_node_features
    elif args.feature_type == 'C':
        nparray_adj_source = train_matrix
        degree_source = np.sum(nparray_adj_source, axis=1)
        degree_source = degree_source.reshape(1,len(nparray_adj_source[0]))
        degree_source = degree_source.T
        constant_feature_source = np.matrix(degree_source)
        features_train = torch.Tensor(constant_feature_source)
    
        nparray_adj_val = val_matrix
        degree_val = np.sum(nparray_adj_val, axis=1)
        degree_val = degree_val.reshape(1,len(nparray_adj_val[0]))
        degree_val = degree_val.T
        constant_feature_val = np.matrix(degree_val)
        features_val = torch.Tensor(constant_feature_val)
    
        nparray_adj_target = test_matrix
        degree_target = np.sum(nparray_adj_target, axis=1)
        degree_target = degree_target.reshape(1,len(nparray_adj_target[0]))
        degree_target = degree_target.T
        constant_feature_target = np.matrix(degree_target)
        features_test = torch.Tensor(constant_feature_target)
    

    if args.PE_method == 'DW':
        #for training dataset
        G = nx.DiGraph(train_matrix)
        model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
        model_emb.train(window_size=5,iter=3, embed_size = args.PE_dim)# train model
        emb = model_emb.get_embeddings()# get embedding vectors
        embeddings = []
        for i in range(len(emb)):
            embeddings.append(emb[i])
        embeddings = np.array(embeddings)
    
        #for val dataset
        G = nx.DiGraph(val_matrix)
        model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
        model_emb.train(window_size=5,iter=3, embed_size = args.PE_dim)# train model
        emb = model_emb.get_embeddings()# get embedding vectors
        val_embeddings = []
        for i in range(len(emb)):
            val_embeddings.append(emb[i])
        val_embeddings = np.array(val_embeddings)
    
        #for test dataset
        G = nx.DiGraph(test_matrix)
        model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
        model_emb.train(window_size=5,iter=3, embed_size = args.PE_dim)# train model
        emb = model_emb.get_embeddings()# get embedding vectors
        test_embeddings = []
        for i in range(len(emb)):
            test_embeddings.append(emb[i])
        test_embeddings = np.array(test_embeddings)

    elif args.PE_method == 'LE':
        #LAP
        sp_adj = sp.coo_matrix(train_matrix)
        g = dgl.from_scipy(sp_adj)
        embeddings = np.array(laplacian_positional_encoding(g, 128))
        train_embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)
    
        sp_adj = sp.coo_matrix(val_matrix)
        g = dgl.from_scipy(sp_adj)
        embeddings = np.array(laplacian_positional_encoding(g, 128))
        val_embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)
    
        sp_adj = sp.coo_matrix(test_matrix)
        g = dgl.from_scipy(sp_adj)
        embeddings = np.array(laplacian_positional_encoding(g, 128))
        test_embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)
    
        
    #train dta->GPU
    x_train = torch.cat((torch.tensor(train_embeddings), features_train), 1)    
    train_edge_index = np.array(train_edge_index).transpose()
    train_edge_index = torch.from_numpy(train_edge_index)
    
    x_train = x_train.cuda(device)
    train_edge_index = train_edge_index.cuda(device)
    
    #val dta->GPU
    x_val = torch.cat((torch.tensor(val_embeddings), features_val), 1)    
    val_edge_index = np.array(val_edge_index).transpose()
    val_edge_index = torch.from_numpy(val_edge_index)
    
    x_val = x_val.cuda(device)
    val_edge_index = val_edge_index.cuda(device)
    
    # target dataset
    x_test = torch.cat((torch.tensor(test_embeddings), features_test), 1)
    
    test_edge_index = np.array(test_edge_index).transpose()
    test_edge_index = torch.from_numpy(test_edge_index)
    
    x_test = x_test.cuda(device)
    test_edge_index = test_edge_index.cuda(device)
    
    model = Net(feats_dim = len(features_train[1]), pos_dim = args.PE_dim, m_dim = len(features_train[1]),
               use_former_information = False, update_coors = False)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 5e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    results = train_model_ppi(model, optimizer, x_train, train_edge_index, x_val, val_edge_index,
                          x_test, test_edge_index,
                          train_loader, val_loader, test_loader, device = device)
    auc.append(results[0])
    ap.append(results[1])
    sum_metric += results
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))