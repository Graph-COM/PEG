import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import networkx as nx
from logger import Logger
from PEGlayer import *
import scipy.sparse as sp
import tensorflow
import dgl
import networkx as nx
import random
import math
from sklearn.preprocessing import normalize
import sys
sys.path.append("../..")
from Graph_embedding import DeepWalk

#modified from: https://github.com/snap-stanford/ogb/tree/master/examples/linkproppred/ddi
class PEG(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(PEG, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(PEGconv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                PEGconv(hidden_channels, hidden_channels, cached=True))
        self.output = PEGconv(hidden_channels, out_channels, cached=True)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, embeddings):
        for conv in self.convs:
            x = conv(x, adj_t, embeddings)
            #x = F.relu(x)
            #x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.output(x, adj_t, embeddings)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(PESAGEconv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(PESAGEconv(hidden_channels, hidden_channels))
        self.output = PESAGEconv(hidden_channels, out_channels)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, embeddings):
        for conv in self.convs:
            #x = conv(x, adj_t, embeddings)
            x = conv(x, adj_t, embeddings)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        #x = self.output(x, adj_t, embeddings)
        x = self.output(x, adj_t, embeddings)
        return x

class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.output = torch.nn.Linear(2,1)
        #self.up = torch.nn.Linear(1,4)
        #self.down = torch.nn.Linear(4,1)
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j, pos_i, pos_j):
        x = x_i * x_j
        pos_encode = ((pos_i - pos_j)**2).sum(dim=-1, keepdim=True)
        
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        out = self.output(torch.cat([x, pos_encode], 1))
        #out = x
        #out = self.up(pos_encode)
        #out = self.down(out)
        return torch.sigmoid(out)


def train(model, predictor, x, embeddings, adj_t, split_edge, optimizer, batch_size):

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)

    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(x, edge_index, embeddings)
        #h = model(x, edge_index)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                 num_neg_samples=perm.size(0), method='dense')

        neg_out = predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, embeddings, adj_t, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    row, col, _ = adj_t.coo()
    edge_index = torch.stack([col, row], dim=0)

    h = model(x, edge_index, embeddings)
    #h = model(x, edge_index)

    pos_train_edge = split_edge['eval_train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]], embeddings[edge[0]], embeddings[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results

def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    #EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR', tol=1e-2) # for 40 PEs
    EigVec = EigVec[:, EigVal.argsort()] # increasing order
    out = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    return out

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI (GNN)')
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--PE_method', type=str, default='DW')
    parser.add_argument('--PE_dim', type=int, default=128)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64 * 512)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygLinkPropPredDataset(name='ogbl-ddi',
                                     transform=T.ToSparseTensor())
    
    Anodataset = PygLinkPropPredDataset(name='ogbl-ddi')

    data = dataset[0]
    adj_t = data.adj_t.to(device)


    if args.PE_method == 'DW':
        G = nx.from_edgelist(np.array(Anodataset[0].edge_index).T)
        model_emb = DeepWalk(G,walk_length=80,num_walks=10,workers=1)#init model
        model_emb.train(embed_size = args.PE_dim)# train model
        emb = model_emb.get_embeddings()# get embedding vectors
        embeddings = []
        for i in range(len(emb)):
            embeddings.append(emb[i])
        embeddings = torch.tensor(np.array(embeddings))
        embeddings = embeddings.to(device)
    
    elif args.PE_method == 'LE':
        G = nx.from_edgelist(np.array(Anodataset[0].edge_index).T)
        G = nx.to_scipy_sparse_matrix(G)
        g = dgl.from_scipy(G)
        embeddings = laplacian_positional_encoding(g, args.PE_dim)
        #embeddings = normalize(embeddings, norm='l2', axis=1, copy=True, return_norm=False)
        embeddings = torch.tensor(embeddings)
        embeddings = embeddings.type(torch.FloatTensor)
        embeddings = embeddings.to(device)

    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    if args.use_sage:
        model = SAGE(args.hidden_channels, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout).to(device)
    else:
        model = PEG(args.hidden_channels, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout).to(device)

    emb = torch.nn.Embedding(data.num_nodes, args.hidden_channels).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-ddi')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@20': Logger(args.runs, args),
        'Hits@30': Logger(args.runs, args),
    }

    
    for run in range(args.runs):
        torch.nn.init.xavier_uniform_(emb.weight)
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(list(model.parameters()) + list(emb.parameters()) + list(predictor.parameters()), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, emb.weight, embeddings, adj_t, split_edge,
                         optimizer, args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, emb.weight, embeddings, adj_t, split_edge,
                               evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()



if __name__ == "__main__":
    main()
