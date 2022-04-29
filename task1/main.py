import argparse
import dgl
import sys
from dataset import *

from train import *
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
sys.path.append("..")
from Graph_embedding import DeepWalk
from model import *
from utils import *

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
parser.add_argument('--device', type=int, default=3)
parser.add_argument('--dataset', type=str, default='cora', help = 'dataset name', 
                    choices = ['cora', 'citeseer', 'pubmed', 'PTBR', 'RU', 'ENGB', 'ES', 'chameleon'])
parser.add_argument('--PE_method', type=str, default="DW", help = 'positional encoding techniques',
                    choices = ['DW', 'LE'])
parser.add_argument('--feature_type', type=str, default="N", help = 'features type, N means node feature, C means constant feature (node degree)',
                    choices = ['N', 'C'])
# GNN settings
parser.add_argument('--num_layers', type=int, default=2, help = 'number of layers')
parser.add_argument('--PE_dim', type=int, default=128, help = 'dimension of positional encoding')
parser.add_argument('--hidden_dim', type=int, default=128, help = 'hidden dimension')

parser.add_argument('--batch_size', type=int, default=128, help = 'batch size')
# Training settings
parser.add_argument('--lr', type=float, default=0.01, help = 'learning rate')
parser.add_argument('--weight_decay', type=float, default= 5e-4, help = 'weight decay')
parser.add_argument('--epochs', type=int, default=100, help = 'number of epochs to train')
parser.add_argument('--val_ratio', type=float, default=0.05, help = 'validation ratio')
parser.add_argument('--test_ratio', type=float, default=0.1, help = 'testing ratio')
#parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--random_partition', action='store_true', help = 'whether to use random partition while training')

args = parser.parse_args()
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
seed_list = [115, 105, 100]
sum_metric = np.zeros((1, 2))
auc = []
ap = []
for i in seed_list:
    dataset = args.dataset
    feature_type = args.feature_type
    set_random_seed(i)
    adj, features= load_data(args.dataset)

    if args.random_partition:
        val_dataset, test_dataset, train_matrix, train_edge_index, id_train_positive, id_train_negative = create_dataloader_plus(adj, val_ratio = args.val_ratio, test_ratio = args.test_ratio, seed = i)                                                                                                          
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= args.batch_size, shuffle=True)

    else:

        train_dataset, val_dataset, test_dataset, train_matrix, train_edge_index = create_dataloader(adj, 
                                                                                                 val_ratio = args.val_ratio, 
                                                                                                 test_ratio = args.test_ratio, 
                                                                                                 seed = i)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= args.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size= args.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size= args.batch_size, shuffle=True)

    if args.feature_type == 'N':
        pca=PCA(n_components=128)
        features=pca.fit_transform(np.array(features))
        features = torch.tensor(features)
        features = features.type(torch.FloatTensor)
    elif args.feature_type == 'C':
        nparray_adj = train_matrix
        degree = np.sum(nparray_adj, axis=1)
        degree = degree.reshape(1,len(nparray_adj[0]))
        degree = degree.T
        constant_feature = np.matrix(degree)
        constant_feature = normalize(constant_feature, norm='l2', axis=0, copy=True, return_norm=False)
        features = torch.Tensor(constant_feature)
    

    if args.PE_method == 'DW':
        #deepwalk
        G = nx.DiGraph(train_matrix)
        model_emb = DeepWalk(G,walk_length=80, num_walks=10,workers=1)#init model
        model_emb.train(embed_size = args.PE_dim)# train model
        emb = model_emb.get_embeddings()# get embedding vectors
        embeddings = []
        for i in range(len(emb)):
            embeddings.append(emb[i])
        embeddings = np.array(embeddings)
    elif args.PE_method == 'LE':
        #LAP
        sp_adj = sp.coo_matrix(train_matrix)
        g = dgl.from_scipy(sp_adj)
        embeddings = np.array(laplacian_positional_encoding(g, args.PE_dim))
        embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)
    
        
    x = torch.cat((torch.tensor(embeddings), features), 1)
    edge_index = np.array(train_edge_index).transpose()
    edge_index = torch.from_numpy(edge_index)
    
    x = x.cuda(device)
    edge_index = edge_index.cuda(device)
        
    model = Net(in_feats_dim = len(features[1]), pos_dim = args.PE_dim, hidden_dim = args.hidden_dim)
    model = model.to(device)
    if args.random_partition:
        if args.feature_type == 'N':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif args.feature_type == 'C':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
        results = train_model_plus(model, optimizer, x, edge_index,id_train_positive, id_train_negative,train_matrix, features, 
                    test_loader, val_loader, PE_dim = args.PE_dim, PE_method = args.PE_method, training_batch_size = args.batch_size, device = device)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        results = train_model(model, optimizer, x, edge_index, train_loader, val_loader, test_loader, device = device)

    auc.append(results[0])
    ap.append(results[1])
    sum_metric += results
print('auc_test: {:.4f}'.format((sum_metric/len(seed_list))[0][0]),
      'ap_test: {:.4f}'.format((sum_metric/len(seed_list))[0][1]))
print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
