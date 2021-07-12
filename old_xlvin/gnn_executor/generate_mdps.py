import torch
from gnn_executor.graph_types import *


def build_mask_given_graph_type(num_actions, graph_type_fn, num_states, seed=None, degree=None):
    states_degree_seed = {erdos_renyi, barabasi_albert}
    states_seed = {tree, caterpillar, lobster}
    states = {grid, caveman, ladder, line, star}

    p_a = []
    for i in range(num_actions):
        if graph_type_fn in states_degree_seed:
            adj = nx.adjacency_matrix(graph_type_fn(num_states, degree, seed)).todense()
        elif graph_type_fn in states_seed:
            adj = nx.adjacency_matrix(graph_type_fn(num_states, seed)).todense()
        elif graph_type_fn in states:
            adj = nx.adjacency_matrix(graph_type_fn(num_states)).todense()
        p_a += [torch.Tensor(adj)]
    p = torch.stack(p_a, dim=0)
    return p


def generate_mdp(num_states, num_actions, discount=0.9, graph_type='random', seed=1111,
                 # for cartpole
                 depth=10, delta=0.1, accel=0.05, thresh=0.5, initial_position=0,
                 # for freeway
                 N=15, stages=3, max_jumpback=5, p_jumpback_if_hit=0.8, p_jumpback=0.2):
    # P: a, s, s'
    # R: s, a
    #
    if graph_type == 'maze8':
        p, r, _ = process(file='../datasets/vin_maze/gridworld_8x8.npz')
        return p, r, discount
    elif graph_type == 'maze16':
        p, r, _ = process(file='../datasets/vin_maze/gridworld_16x16.npz')
        return p, r, discount
    elif graph_type == 'maze28':
        p, r, _ = process(file='../datasets/vin_maze/gridworld_28x28.npz')
        return p, r, discount
    elif graph_type == 'cartpole':
        p, r = cartpole_graph(depth=depth, delta=delta, accel=accel, thresh=thresh, initial_position=initial_position)
        return p, r, discount
    elif graph_type == 'freeway':
        p,r = freeway_graph(N=N, stages=stages, max_jumpback=max_jumpback, p_jumpback_if_hit=p_jumpback_if_hit,
                            p_jumpback=p_jumpback)
        return p, r, discount
    elif graph_type == 'det':
        p, r = deterministic_k_mdp(num_states, num_actions)
        return p, r, discount

    attempt_no = 0
    p = torch.rand(num_actions, num_states, num_states)

    while True:
        if graph_type == 'random':
            mask = torch.randint(0, 2, (num_actions, num_states, num_states))
        # state_fn
        elif graph_type == 'grid':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=grid, num_states=num_states)
        elif graph_type == 'caveman':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=caveman, num_states=num_states)
        elif graph_type == 'ladder':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=ladder, num_states=num_states)
        elif graph_type == 'line':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=line, num_states=num_states)
        elif graph_type == 'star':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=star, num_states=num_states)
        # states, seed
        elif graph_type == 'tree':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=tree, num_states=num_states,
                                               seed=seed)
        elif graph_type == 'caterpillar':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=caterpillar,
                                               num_states=num_states,
                                               seed=seed)
        elif graph_type == 'lobster':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=lobster, num_states=num_states,
                                               seed=seed)
        # state_degree_seed_fn
        elif graph_type == 'erdos':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=erdos_renyi,
                                               num_states=num_states,
                                               seed=seed, degree=int(num_states * 0.5))
        elif graph_type == 'barabasi':
            mask = build_mask_given_graph_type(num_actions=num_actions, graph_type_fn=barabasi_albert,
                                               num_states=num_states,
                                               seed=seed, degree=int(num_states * 0.5))
        else:
            raise NotImplementedError

        p = p * mask
        r = torch.rand(num_states, num_actions) * (-1. - 1.) + 1.  # between -1, 1
        as_sum = torch.sum(p, dim=-1, keepdim=True)
        p = p / as_sum

        if not torch.isnan(p).any():
            return p, r, discount
        attempt_no += 1
        # print("Attempts required to generate non-NaN transition matrix ", attempt_no)


def bellman_optimality_operator(v, P, R, discount):
    pv = torch.einsum('ijk,k->ji', P, v)
    newv, _ = torch.max(torch.add(R, discount * pv), dim=1)
    return newv


def value_iteration(p, r, discount, v0=None, eps=1e-8):
    if v0 is None:
        v0 = torch.zeros(r.shape[0])
    iter_diff = float("inf")
    v_prev = v0
    vs = [v0]
    while iter_diff > eps:
        newv = bellman_optimality_operator(v_prev, p, r, discount)
        vs += [newv]
        iter_diff = torch.norm(newv - v_prev)
        v_prev = newv
    return torch.stack(vs, dim=0)


def find_policy(p, r, discount, v):
    max_a, argmax_a = torch.max(r + discount * torch.einsum('ijk,k->ji', p, v), dim=1)
    return argmax_a


""""
p, r, discount = generate_mdp(20, 2, graph_type='grid')
print("p ", p)
print("r ", r)
vs = value_iteration(p, r, discount)
pol = find_policy(p, r, discount, vs[-1])
"""""

"""
elif graph_type == 'caveman':
    p_a = []
    for i in range(num_actions):
        adj = nx.adjacency_matrix(caveman(num_states, int(num_states*0.4), seed)).todense()
    p = torch.stack(p_a, dim=0)
"""
