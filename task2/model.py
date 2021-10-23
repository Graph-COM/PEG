import torch.nn.functional as F
from utils import *
from dataset import *
class Net(torch.nn.Module):
    def __init__(self, feats_dim, pos_dim, m_dim, use_former_information = True, update_coors=True):
        super(Net, self).__init__()
        
        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.use_former_information = use_former_information
        self.update_coors = update_coors
        
        self.conv1 = AnoGCNConv(feats_dim = feats_dim, pos_dim = pos_dim, m_dim = m_dim,
                               use_formerinfo = use_former_information, update_coors = update_coors)
        self.conv2 = AnoGCNConv(feats_dim = feats_dim, pos_dim = pos_dim, m_dim = m_dim, 
                               use_formerinfo = use_former_information, update_coors = update_coors)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.fc = nn.Linear(2, 1)
        #self.output = nn.Linear(4,1)

    def forward(self, x, edge_index, idx):
        # (batch of) graph object(s) containing all the tensors we want
        #x = F.relu(self.conv1(x, positional_features, edge_index))
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        pos_dim = self.pos_dim
        
        nodes_first = x[ : , pos_dim: ][idx[0]]
        nodes_second = x[ : , pos_dim: ][idx[1]]
        pos_first = x[ : , :pos_dim ][idx[0]]
        pos_second = x[ : , :pos_dim ][idx[1]]
        
        positional_encoding = ((pos_first - pos_second)**2).sum(dim=-1, keepdim=True)
        #link = torch.cat([nodes_first, nodes_second, positional_encoding], dim = 1)
        #positional_encoding = torch.sum(pos_first * pos_second, dim=-1)
        pred = torch.sum(nodes_first * nodes_second, dim=-1)
        out = self.fc(torch.cat([pred.reshape(len(pred), 1),positional_encoding.reshape(len(positional_encoding), 1)], 1))
        #link_pred = self.fc(positional_encoding.reshape(len(positional_encoding), 1))
        #link_pred = self.output(link_pred)
        return out

    def loss(self, pred, link_label):
        return self.loss_fn(pred, link_label)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

        
def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

import math

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
import torch
from torch import nn, einsum, broadcast_tensors
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum as sparsesum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj, Size, OptTensor, Tensor

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps
        self.fn = nn.Sequential(nn.LayerNorm(1), nn.GELU())

    def forward(self, coors):
        norm = coors.norm(dim = -1, keepdim = True)
        normed_coors = coors / norm.clamp(min = self.eps)
        phase = self.fn(norm)
        return (phase * normed_coors)
    
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


class AnoGCNConv(MessagePassing):
    """

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and compute
            symmetric normalization coefficients on the fly.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, feats_dim: int, pos_dim: int, m_dim: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, update_coors: bool = False,
                 use_formerinfo: bool = False, norm_coors = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(AnoGCNConv, self).__init__(**kwargs)

        self.feats_dim = feats_dim
        self.pos_dim = pos_dim
        self.m_dim = m_dim
        self.update_coors = update_coors
        self.use_formerinfo = use_formerinfo
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        
        self.coors_norm = CoorsNorm() if norm_coors else nn.Identity()

        self._cached_edge_index = None
        self._cached_adj_t = None
        
        self.edge_mlp1 = nn.Linear(1, 32)
        self.edge_mlp2 = nn.Linear(32, 1)
        #self.edge_mlp = nn.Linear(feats_dim + 1, feats_dim)
        self.weight_withformer = Parameter(torch.Tensor(feats_dim + m_dim, feats_dim))
        self.weight_noformer = Parameter(torch.Tensor(m_dim, feats_dim))
        self.coors_mlp = nn.Sequential(
            nn.Linear(m_dim, m_dim * 4),
            nn.Linear(m_dim * 4, 1)
        ) if update_coors else None

        if bias:
            self.bias = Parameter(torch.Tensor(feats_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight_withformer)
        glorot(self.weight_noformer)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self,x: Tensor, edge_index: Adj, 
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        
        coors, feats = x[:, :self.pos_dim], x[:, self.pos_dim:]
        
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, feats.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, feats.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        
        
        rel_coors = coors[edge_index[0]] - coors[edge_index[1]]
        rel_dist  = (rel_coors ** 2).sum(dim=-1, keepdim=True)
        #rel_dist = torch.sum(coors[edge_index[0]] * coors[edge_index[1]], dim=-1)
        #rel_dist = rel_dist.reshape(len(rel_dist),1)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        hidden_out, coors_out = self.propagate(edge_index, x = feats, edge_weight=edge_weight, pos=rel_dist,coors=coors, rel_coors=rel_coors,
                             size=None)
        
        

        if self.bias is not None:
            hidden_out += self.bias

        return torch.cat([coors_out, hidden_out], dim=-1)


    def message(self, x_i: Tensor, x_j: Tensor, edge_weight: OptTensor, pos) -> Tensor:
        temp = self.edge_mlp1(pos)
        temp = self.edge_mlp2(temp)
        temp = torch.sigmoid(temp)
        return x_j if edge_weight is None else temp * edge_weight.view(-1, 1) * x_j
    
    def propagate(self, edge_index: Adj, size: Size = None, **kwargs):
        """The initial call to start propagating messages.
            Args:
            `edge_index` holds the indices of a general (sparse)
                assignment matrix of shape :obj:`[N, M]`.
            size (tuple, optional) if none, the size will be inferred
                and assumed to be quadratic.
            **kwargs: Any additional data which is needed to construct and
                aggregate messages, and to update node embeddings.
        """
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__,
                                     edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)


        m_i = self.aggregate(m_ij, **aggr_kwargs)
        
        # update coors if specified
        if self.update_coors:
            coor_wij = self.coors_mlp(m_ij)
            kwargs["rel_coors"] = self.coors_norm(kwargs["rel_coors"])
            mhat_i = self.aggregate(coor_wij * kwargs["rel_coors"], **aggr_kwargs)
            coors_out = kwargs["coors"] + mhat_i
        else:
            coors_out = kwargs["coors"]
        
        
        hidden_feats = kwargs["x"]
        if self.use_formerinfo:
            hidden_out = torch.cat([hidden_feats, m_i], dim = -1)
            hidden_out = hidden_out @ self.weight_withformer
        else:
            hidden_out = m_i
            hidden_out = hidden_out @ self.weight_noformer
        



        # return tuple
        return self.update((hidden_out, coors_out), **update_kwargs)

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
