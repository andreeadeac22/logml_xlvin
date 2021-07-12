import networkx as nx
import numpy as np
import torch
import math


def deterministic_k_mdp(nb_states, nb_actions):
    P = torch.zeros(nb_actions, nb_states, nb_states)
    R = torch.randn(nb_states, nb_actions)
    for s in range(nb_states):
        for act in range(nb_actions):
            s_prime = np.random.choice(nb_states)
            P[act][s][s_prime] = 1.0
    return P, R
    

def freeway_graph(N=15, stages=3, max_jumpback=5, p_jumpback_if_hit=0.8, p_jumpback=0.2):
    P = torch.zeros((3, N * stages, N * stages))
    R = torch.zeros((N * stages, 3))

    hit_stage = np.random.randint(stages, size=N)

    act_steps = [0, 1, -1]

    for i in range(N):
        for stage in range(stages):
            s = i * stages + stage
            nxt_stage = (stage + 1) % stages
            for act in range(3):
                nxt_i = i + act_steps[act]
                if nxt_i >= N:
                    P[act][s][nxt_stage] = 1.0
                    R[s][act] = 1.0
                elif nxt_i < 0:
                    P[act][s][nxt_stage] = 1.0
                    R[s][act] = 0.0
                else:
                    R[s][act] = 0.0
                    s_prime = nxt_i * stages + nxt_stage
                    if hit_stage[nxt_i] == nxt_stage:
                        P[act][s][s_prime] = 1.0 - p_jumpback_if_hit
                        jumpback_low = max(nxt_i - max_jumpback, 0)
                        gap = nxt_i - jumpback_low
                        if gap == 0:
                            P[act][s][s_prime] = 1.0
                        else:
                            extra_p = p_jumpback_if_hit / gap
                            for j in range(jumpback_low, nxt_i):
                                P[act][s][j * stages + nxt_stage] = extra_p
                    else:
                        P[act][s][s_prime] = 1.0 - p_jumpback
                        jumpback_low = max(nxt_i - max_jumpback, 0)
                        gap = nxt_i - jumpback_low
                        if gap == 0:
                            P[act][s][s_prime] = 1.0
                        else:
                            extra_p = p_jumpback / gap
                            for j in range(jumpback_low, nxt_i):
                                P[act][s][j * stages + nxt_stage] = extra_p
    return P, R


def cartpole_graph(depth=10, delta=0.1, accel=0.05, thresh=0.5, initial_position=0):
    x = [initial_position]
    is_terminal = [False]
    links = {0: []}
    last_chd = list(x)
    last_ind = list([0])
    tail_ind = 1
    for d in range(depth - 1):
        nxt_x = []
        nxt_ind = []
        for i in range(len(last_chd)):
            nxt_pos = last_chd[i] + last_chd[i] * accel
            nxt_pos_1 = nxt_pos + delta
            x.append(nxt_pos_1)
            links[last_ind[i]].append(tail_ind)
            links[tail_ind] = []
            if nxt_pos_1 > thresh or nxt_pos_1 < -thresh:
                is_terminal.append(True)
            else:
                is_terminal.append(False)
                nxt_x.append(nxt_pos_1)
                nxt_ind.append(tail_ind)
            tail_ind += 1
            nxt_pos_2 = nxt_pos - delta
            x.append(nxt_pos_2)
            links[last_ind[i]].append(tail_ind)
            links[tail_ind] = []
            if nxt_pos_2 > thresh or nxt_pos_2 < -thresh:
                is_terminal.append(True)
            else:
                is_terminal.append(False)
                nxt_x.append(nxt_pos_2)
                nxt_ind.append(tail_ind)
            tail_ind += 1
        last_chd = list(nxt_x)
        last_ind = list(nxt_ind)

    for i in range(len(last_chd)):
        nxt_pos = last_chd[i] + last_chd[i] * accel
        nxt_pos_1 = nxt_pos + delta
        x.append(nxt_pos_1)
        is_terminal.append(True)
        links[last_ind[i]].append(tail_ind)
        links[tail_ind] = []
        tail_ind += 1
        nxt_pos_2 = nxt_pos + delta
        x.append(nxt_pos_2)
        is_terminal.append(True)
        links[last_ind[i]].append(tail_ind)
        links[tail_ind] = []
        tail_ind += 1

    for i in range(len(is_terminal)):
        if is_terminal[i]:
            assert len(links[i]) == 0

    P = torch.zeros((2, tail_ind, tail_ind))
    R = torch.zeros((tail_ind, 2))

    for i in range(len(x)):
        if is_terminal[i]:
            for j in range(2):
                P[j][i][i] = 1.0
                R[i][j] = 0.0
        else:
            for j in range(len(links[i])):
                P[j][i][links[i][j]] = 1.0
                if is_terminal[links[i][j]]:
                    R[i][j] = 0.0
                else:
                    R[i][j] = 1.0

    return P, R


def erdos_renyi(N, degree, seed):
    """ Creates an Erdős-Rényi or binomial graph of size N with degree/N probability of edge creation """
    return nx.fast_gnp_random_graph(N, degree / N, seed, directed=False)


def barabasi_albert(N, degree, seed):
    """ Creates a random graph according to the Barabási–Albert preferential attachment model
        of size N and where nodes are attached with degree edges """
    return nx.barabasi_albert_graph(N, degree, seed)


def grid(N):
    """ Creates a m x k 2d grid graph with N = m*k and m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.grid_2d_graph(m, N // m)


def caveman(N):
    """ Creates a caveman graph of m cliques of size k, with m and k as close as possible """
    m = 1
    for i in range(1, int(math.sqrt(N)) + 1):
        if N % i == 0:
            m = i
    return nx.caveman_graph(m, N // m)


def tree(N, seed):
    """ Creates a tree of size N with a power law degree distribution """
    return nx.random_powerlaw_tree(N, seed=seed, tries=10000)


def ladder(N):
    """ Creates a ladder graph of N nodes: two rows of N/2 nodes, with each pair connected by a single edge.
        In case N is odd another node is attached to the first one. """
    G = nx.ladder_graph(N // 2)
    if N % 2 != 0:
        G.add_node(N - 1)
        G.add_edge(0, N - 1)
    return G


def line(N):
    """ Creates a graph composed of N nodes in a line """
    return nx.path_graph(N)


def star(N):
    """ Creates a graph composed by one center node connected N-1 outer nodes """
    return nx.star_graph(N - 1)


def caterpillar(N, seed):
    """ Creates a random caterpillar graph with a backbone of size b (drawn from U[1, N)), and N − b
        pendent vertices uniformly connected to the backbone. """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, N):
        G.add_edge(i, np.random.randint(B))
    return G


def lobster(N, seed):
    """ Creates a random Lobster graph with a backbone of size b (drawn from U[1, N)), and p (drawn
        from U[1, N − b ]) pendent vertices uniformly connected to the backbone, and additional
        N − b − p pendent vertices uniformly connected to the previous pendent vertices """
    np.random.seed(seed)
    B = np.random.randint(low=1, high=N)
    F = np.random.randint(low=B + 1, high=N + 1)
    G = nx.empty_graph(N)
    for i in range(1, B):
        G.add_edge(i - 1, i)
    for i in range(B, F):
        G.add_edge(i, np.random.randint(B))
    for i in range(F, N):
        G.add_edge(i, np.random.randint(low=B, high=F))
    return G


# the initial process fn for VIN mazes, with random rewards.
def old_process(file='gridworld_8x8.npz', train=False):
    with np.load(file, mmap_mode='r') as f:
        if train:
            images = f['arr_0']
        else:
            images = f['arr_4']
    images = images.astype(np.float32)

    nb_images = images.shape[0]
    nb_actions = 8

    dx = [1, 0, -1, 0, 1, 1, -1, -1]
    dy = [0, 1, 0, -1, 1, -1, 1, -1]
    r = [-0.1, -0.1, -0.1, -0.1, -0.1414, -0.1414, -0.1414, -0.1414]

    # Ps = []
    # Rs = []

    # Print number of samples
    """
    if train:
        print("Number of Train Samples: {0}".format(images.shape[0]))
    else:
        print("Number of Test Samples: {0}".format(images.shape[0]))
    """
    img_index = np.random.randint(images.shape[0])

    # for img in indices:
    grid = images[img_index, 0]
    reward = images[img_index, 1] / 10.0

    ind = []
    rev_map = {}

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0.0:
                rev_map[(x, y)] = len(ind)
                ind.append((x, y))

    nb_states = len(ind)

    P = torch.zeros((nb_actions, nb_states, nb_states))
    R = torch.zeros((nb_states, nb_actions))

    for s, (x, y) in enumerate(ind):
        for act in range(nb_actions):
            if reward[x, y] > 0.0:
                P[act][s][s] = 1.0
            else:
                next_x = x + dx[act]
                next_y = y + dy[act]
                if (next_x, next_y) not in rev_map:
                    next_x = x
                    next_y = y
                next_r = r[act] + reward[next_x, next_y]
                s_prime = rev_map[(next_x, next_y)]
                P[act][s][s_prime] = 1.0
                # train on similar R distributions?!
                # R[s][act] = next_r * 2.5

    R = torch.rand(nb_states, nb_actions) * (-1. - 1.) + 1.

    # Ps.append(P)
    # Rs.append(R)
    return P, R


"""
Ps, Rs = process(file='gridworld_28x28.npz', train=False)
lengths = [p.shape[1] for p in Ps]
print(min(lengths), max(lengths))
"""


def process(file='gridworld_8x8.npz', train=False, given_maze=None):
    nb_actions = 8
    dx = [1, 0, -1, 0, 1, 1, -1, -1]
    dy = [0, 1, 0, -1, 1, -1, 1, -1]

    if given_maze is None:
        with np.load(file, mmap_mode='r') as f:
            if train:
                images = f['arr_0']
            else:
                images = f['arr_4']
        images = images.astype(np.float32)

        nb_images = images.shape[0]

        """
        if train:
            print("Number of Train Samples: {0}".format(images.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(images.shape[0]))
        """
        img_index = np.random.randint(images.shape[0])

        grid = images[img_index, 0]
        reward = images[img_index, 1] / 10.0
    else:
        grid = given_maze[0]
        reward = given_maze[1] / 10.0

    ind = []
    rev_map = {}

    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            if grid[x, y] == 0.0:
                rev_map[(x, y)] = len(ind)
                ind.append((x, y))

    terminal_state = len(ind)

    nb_states = len(ind) + 1

    P = torch.zeros((nb_actions, nb_states, nb_states))
    R = torch.zeros((nb_states, nb_actions))

    for s, (x, y) in enumerate(ind):
        for act in range(nb_actions):
            if reward[x, y] > 0.0:
                P[act][s][s] = 1.0
                R[s][act] = 1.0
            else:
                next_x = x + dx[act]
                next_y = y + dy[act]
                if (next_x, next_y) not in rev_map:
                    next_x = x
                    next_y = y
                    s_prime = terminal_state
                    next_r = -1.0
                else:
                    s_prime = rev_map[(next_x, next_y)]
                    if reward[next_x, next_y] > 0.0:
                        next_r = 1.0
                    else:
                        next_r = -0.01
                P[act][s][s_prime] = 1.0
                R[s][act] = next_r

    for act in range(nb_actions):
        P[act][terminal_state][terminal_state] = 1.0
        R[terminal_state][act] = -1.0

    return P, R, rev_map
