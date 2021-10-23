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



def load_data(dataset_name):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        adj, features= load_data_citation(dataset_name)
    elif dataset_name in ['PTBR', 'RU', 'ENGB', 'ES']:
        adj, features = load_data_twitch(dataset_name)
    elif dataset_name in ['chameleon']:
        adj, features= load_data_cham(dataset_name)
    
    return adj, features



# Data settings
parser = argparse.ArgumentParser(description='PEG')
parser.add_argument('--source_dataset', type=str, default='cora')
parser.add_argument('--target_dataset', type=str, default='citeseer')
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
parser.add_argument('--random_partition', action='store_true')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sum_metric = np.zeros((1, 2))
auc = []
ap = []
for i in [115,105,100]:

    adj, features_source= load_data(args.source_dataset)
    adj_target, features_target = load_data(args.target_dataset)

    if args.random_partition:
        val_dataset, train_matrix, train_edge_index, id_train_positive, id_train_negative = create_dataloader_source_plus(adj, 
                                                                                                                    val_ratio = 0.1, seed = i)
        test_dataset, test_matrix, test_edge_index = create_dataloader_target_plus(adj_target, test_ratio = 0.1, seed = i)
    
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    else:

        train_dataset, val_dataset, train_matrix, train_edge_index = create_dataloader_source(adj, 
                                                                                        val_ratio = 0.1, 
                                                                                        seed = i)
        test_dataset, test_matrix, test_edge_index = create_dataloader_target(adj_target, test_ratio = 0.1, seed = i)
    
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True)

    if args.feature_type == 'N':
        transformer = random_projection.GaussianRandomProjection(n_components = 128)
        features_source = transformer.fit_transform(np.array(features_source))
        features_target = transformer.fit_transform(np.array(features_target))
        features_source = normalize(features_source, norm='l2', axis=1, copy=True, return_norm=False)
        features_target = normalize(features_target, norm='l2', axis=1, copy=True, return_norm=False)
        features_source = torch.tensor(features_source)
        features_target = torch.tensor(features_target)
    
        features_source = features_source.type(torch.FloatTensor)
        features_target = features_target.type(torch.FloatTensor)
    elif args.feature_type == 'C':
        nparray_adj_source = train_matrix
        degree_source = np.sum(nparray_adj_source, axis=1)
        degree_source = degree_source.reshape(1,len(nparray_adj_source[0]))
        degree_source = degree_source.T
        constant_feature_source = np.matrix(degree_source)
        features_source = torch.Tensor(constant_feature_source)
    
        nparray_adj_target = test_matrix
        degree_target = np.sum(nparray_adj_target, axis=1)
        degree_target = degree_target.reshape(1,len(nparray_adj_target[0]))
        degree_target = degree_target.T
        constant_feature_target = np.matrix(degree_target)
        features_target = torch.Tensor(constant_feature_target)
    

    if args.PE_method == 'DW':
        #DeepWalk
        #for source dataset
        G = nx.DiGraph(train_matrix)
        model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
        model_emb.train(window_size=5,iter=3, embed_size = args.PE_dim)# train model
        emb = model_emb.get_embeddings()# get embedding vectors
        embeddings = []
        for i in range(len(emb)):
            embeddings.append(emb[i])
        embeddings = np.array(embeddings)
    
        #for target dataset
        G = nx.DiGraph(test_matrix)
        model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
        model_emb.train(window_size=5,iter=3, embed_size = args.PE_dim)# train model
        emb = model_emb.get_embeddings()# get embedding vectors
        target_embeddings = []
        for i in range(len(emb)):
            target_embeddings.append(emb[i])
        target_embeddings = np.array(target_embeddings)

    elif args.PE_method == 'LE':
        #LAP
        sp_adj = sp.coo_matrix(train_matrix)
        g = dgl.from_scipy(sp_adj)
        embeddings = np.array(laplacian_positional_encoding(g, args.PE_dim))
        embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)

        sp_adj = sp.coo_matrix(test_matrix)
        g = dgl.from_scipy(sp_adj)
        target_embeddings = np.array(laplacian_positional_encoding(g, args.PE_dim))
        target_embeddings = normalize(target_embeddings, norm='l2', axis=1, copy=True, return_norm=False)
    
        
    x = torch.cat((torch.tensor(embeddings), features_source), 1)    
    edge_index = np.array(train_edge_index).transpose()
    edge_index = torch.from_numpy(edge_index)
    
    x = x.cuda(device)
    edge_index = edge_index.cuda(device)
    
    # target dataset
    x_target = torch.cat((torch.tensor(target_embeddings), features_target), 1)
    
    edge_index_target = np.array(test_edge_index).transpose()
    edge_index_target = torch.from_numpy(edge_index_target)
    
    x_target = x_target.cuda(device)
    edge_index_target = edge_index_target.cuda(device)
    
    model = Net(feats_dim = len(features_source[1]), pos_dim = args.PE_dim, m_dim = len(features_source[1]),
               use_former_information = False, update_coors = False)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay = 5e-4)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    
    model = model.to(device)
    if args.random_partition:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 5e-4)
        results,loss_history,loss_val = train_model_plus(model, optimizer, x, edge_index, x_target, edge_index_target, 
                                                id_train_positive, id_train_negative,train_matrix, features_source, 
                                                val_loader, test_loader, PE_dim = args.PE_dim, PE_method = args.PE_method, device = device)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
        results = train_model(model, optimizer, x, edge_index, x_target, edge_index_target, train_loader, val_loader, test_loader, device = device)
    auc.append(results[0])
    ap.append(results[1])
    sum_metric += results
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))