import torch
import torch.nn as nn
from torch_scatter import scatter_max, scatter_sum
from collections import OrderedDict

# TODO: rewrite! It's an old one from old_xlvin/gnn_executor/sparse_models.py

class SparseMessagePassing(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 hidden_dim,
                 message_function=None,
                 message_function_depth=None,
                 neighbour_state_aggr='sum',
                 activation=False,
                 layernorm=False):
        super().__init__()
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.message_function = message_function
        self.message_function_depth = message_function_depth
        self.neighbour_state_aggr = neighbour_state_aggr

        if message_function == 'mpnn':
            self.message_proj1 = nn.Linear(3 * hidden_dim, hidden_dim)
            if message_function_depth == 2:
                self.message_proj2 = nn.Linear(hidden_dim, hidden_dim)

        #self.readout_node_proj = nn.Linear(hidden_dim, hidden_dim)
        #self.readout_msg_proj = nn.Linear(hidden_dim, hidden_dim)  # TODO: where?
        self.activation = activation
        if activation:
            self.relu = nn.ReLU()
        self.layernorm = layernorm
        if layernorm:
            self.ln = nn.LayerNorm(hidden_dim)


    def forward(self, x, senders, receivers, edge_feat):
        msg = self.message_proj1(torch.cat((x[senders], x[receivers], edge_feat), dim=-1))
        if self.activation:
            msg = self.relu(msg)
        if self.neighbour_state_aggr == 'sum':
            aggr_messages = torch.zeros((x.shape[0], self.hidden_dim), device=x.device)  # -1, adj_rows, msg)
            aggr_messages.index_add_(0, receivers, msg)
        elif self.neighbour_state_aggr == 'max':
            import warnings
            warnings.filterwarnings("ignore")
            aggr_messages = torch.ones((x.shape[0], self.hidden_dim), device=x.device) * -1e9  # -1, adj_rows, msg)
            scatter_max(msg, receivers, dim=0, out=aggr_messages)
            indegree = scatter_sum(torch.ones_like(msg), receivers, dim=0, out=torch.zeros_like(aggr_messages))
            aggr_messages = aggr_messages * (indegree > 0)
        else:
            raise NotImplementedError

        x = aggr_messages + x
        if self.layernorm:
            x = self.ln(x)
        return x


class SparseMPNN(nn.Module):
    def __init__(self,
                 node_features,
                 edge_features,
                 hidden_dim=None,
                 out_features=None,
                 message_function=None,
                 message_function_depth=None,
                 neighbour_state_aggr=None,
                 gnn_steps=1,
                 msg_activation=False,
                 layernorm=False):
        super().__init__()
        self.node_proj = nn.Sequential(
            nn.Linear(node_features, hidden_dim, bias=False))
        self.edge_proj = nn.Linear(edge_features, hidden_dim)
        self.hidden_dim = hidden_dim
        self.mps = SparseMessagePassing(node_features=node_features,
                                       edge_features=edge_features,
                                       hidden_dim=hidden_dim,
                                       message_function=message_function,
                                       message_function_depth=message_function_depth,
                                       neighbour_state_aggr=neighbour_state_aggr, activation=msg_activation,
                                        layernorm=layernorm)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=out_features)
        self.gnn_steps = gnn_steps

    def forward(self, data):
        # node.shape: a, s, 2
        x, adj, adj_mask = data
        senders, receivers, edge_feat = adj
        num_actions = x.shape[0]
        num_states = x.shape[1]
        outputs = []

        enc_x = self.node_proj(x)
        edge_feat = self.edge_proj(edge_feat)

        prev_gnnstep_x = torch.zeros_like(enc_x)

        for i in range(self.gnn_steps):
            prev_gnnstep_x = prev_gnnstep_x + enc_x

            onestepx = self.mps(prev_gnnstep_x.reshape(-1, self.hidden_dim), senders, receivers, edge_feat)

            onestepx = onestepx.reshape(num_actions, num_states, -1)
            onestepx, ind = torch.max(onestepx, dim=0, keepdim=True)
            prev_gnnstep_x = onestepx
            output = self.fc(onestepx.squeeze())
            outputs += [output]

        return outputs
