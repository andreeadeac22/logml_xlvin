import torch
import copy
import numpy as np
from generate_mdps import generate_mdp, value_iteration, find_policy


class GraphData(torch.utils.data.IterableDataset):
    def __init__(self,
                 num_states, num_actions, epsilon,
                 graph_type='random', seed=None, device=None,
                 sparse=False,
                 # cartpole args
                 depth=10, delta=0.1, accel=0.05, thresh=0.5, initial_position=0,
                 # freeway args
                 N=15, stages=3, max_jumpback=5, p_jumpback_if_hit=0.8, p_jumpback=0.2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = epsilon
        self.graph_type = graph_type
        self.seed = seed
        self.device=device
        self.sparse = sparse

        # cartpole args
        self.depth = depth
        self.delta = delta
        self.accel = accel
        self.thresh = thresh
        self.initial_position = initial_position

        # freeway args
        self.N = N
        self.stages = stages
        self.max_jumpback = max_jumpback
        self.p_jumpback_if_hit = p_jumpback_if_hit
        self.p_jumpback = p_jumpback

    def build_graph(self):
        """
        p = torch.tensor([[[0., 1.], [0., 1.]]])
        r = torch.tensor([[0.1], [0.7]])
        discount = 0.1
        """
        p, r, discount = generate_mdp(num_states=self.num_states, num_actions=self.num_actions,
                                      graph_type=self.graph_type, seed=self.seed,
                                      # cartpole args
                                      depth=self.depth, delta=self.delta, accel=self.accel, thresh=self.thresh,
                                      initial_position=self.initial_position,
                                      # freeway args
                                      N=self.N, stages=self.stages, max_jumpback=self.max_jumpback,
                                      p_jumpback_if_hit=self.p_jumpback_if_hit, p_jumpback=self.p_jumpback)

        self.num_actions = p.shape[0]
        self.num_states = p.shape[1]

        vs = value_iteration(p=p, r=r, discount=discount, eps=self.eps).to(self.device)
        policy = find_policy(p, r, discount, vs[-1])
        policy_dict = {
            'p': p.to(self.device),
            'r': r.to(self.device),
            'discount': discount,
            'policy': policy.to(self.device),
        }

        p = torch.transpose(p, dim0=-1, dim1=-2)
        #print("Iterations ", vs.shape[0])
        # p: a, s, s'
        # r: s, a
        # discount: 1
        # vs: iter, s
        np.set_printoptions(threshold=np.infty)
        #print("VS ", vs.numpy())
        #print("policy ", policy)
        #exit(0)
        ones = torch.ones_like(p)
        zeros = torch.zeros_like(p)
        adj_mask = torch.where(p > 0, ones, zeros).unsqueeze(dim=-1)  # a, s, s', 1

        adj_mat_p = p.unsqueeze(dim=-1)  # a, s, s', 1
        discount_mat = torch.ones_like(adj_mat_p) * discount
        adj_mat = torch.cat((adj_mat_p, discount_mat), dim=-1).to(self.device)  # a, s, s, 2

        v_node_feat = vs.unsqueeze(dim=1).repeat(1, p.shape[0], 1)  # iter, a, s
        r_node_feat = r.transpose(dim0=0, dim1=1)  # a, s
        r_node_feat = r_node_feat.unsqueeze(dim=0).repeat(v_node_feat.shape[0], 1, 1)  # iter, a, s
        node_feat = torch.cat((v_node_feat.unsqueeze(dim=-1), r_node_feat.unsqueeze(dim=-1)), dim=-1).to(self.device)  # iter, a, s, 2

        # adj_mat_r = r.transpose(dim0=0, dim1=1) # a, s
        # adj_mat_r = adj_mat_r.unsqueeze(dim=-1).repeat(1, 1, self.num_states) # a, s, s
        # adj_mat_r = adj_mat_r.unsqueeze(dim=-1)
        # adj_mat = torch.cat((adj_mat_p, adj_mat_r), dim=-1)

        if self.sparse:
            adj_mat = self.convert_dense_to_sparse(adj_mat, adj_mask)
            yield (node_feat, adj_mat, None, vs, policy_dict)
        else:
            yield (node_feat, adj_mat, adj_mask, vs, policy_dict)


    def convert_dense_to_sparse(self, adj_mat, adj_mask):
        # from adjacency matrix (a,s,s',2):AxSxSx2 to
        # three tensors: (senders,...):AxE, (receivers,...):AxE and edge features AxExF'
        action_adj_list_senders = []
        action_adj_list_receivers = []
        action_adj_list_edge_features = []

        for action_type in range(adj_mat.shape[0]):
            action_adj_mat = adj_mat[action_type]
            action_adj_mask = adj_mask[action_type]

            indices = action_adj_mask.nonzero()
            sender = indices[:, 0]
            receiver = indices[:, 1]
            edge_features = action_adj_mat[sender, receiver, :]
            gid_sender = action_type * self.num_states + sender
            gid_receiver = action_type * self.num_states + receiver

            action_adj_list_senders += [gid_sender]
            action_adj_list_receivers += [gid_receiver]
            action_adj_list_edge_features += [edge_features]

        action_adj_list_senders = torch.cat(action_adj_list_senders, 0).to(self.device)
        action_adj_list_receivers = torch.cat(action_adj_list_receivers, 0).to(self.device)
        action_adj_list_edge_features = torch.cat(action_adj_list_edge_features, 0).to(self.device)

        return (action_adj_list_senders, action_adj_list_receivers,
                action_adj_list_edge_features)

    def __iter__(self):
        return self.build_graph()
