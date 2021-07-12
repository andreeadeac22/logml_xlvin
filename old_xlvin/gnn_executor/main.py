import argparse
import time
import os
import torch
import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.parameter import Parameter

from dataset import GraphData
from uni_dataset import UniGraphData
from generate_mdps import find_policy


def loss_fn(output, target, reduction='mean'):
    loss = (target.float().squeeze() - output.squeeze()) ** 2
    return loss.sum() if reduction == 'sum' else loss.mean()


def train(data, gnn_steps):
    model.train()
    start = time.time()
    train_loss, n_samples = 0, 0

    node_feat, adj_mat, adj_mask, vs, policy_dict = data
    # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
    # adj_mat.shape: a, s, s, 2 (p, gamma)
    iteration_steps = node_feat.shape[0]

    for step in range(iteration_steps - 1):
        optimizer.zero_grad()
        outputs = model((node_feat[step], adj_mat, adj_mask))
        loss = 0
        for i in range(min(gnn_steps, iteration_steps - 1 - step)):
            loss += loss_fn(outputs[i], vs[step + i + 1] - vs[step + i])

        loss.backward()
        optimizer.step()
        time_iter = time.time() - start
        train_loss += loss.item() * len(outputs[0])
        n_samples += len(outputs[0])
    if epoch % 20 == 0:
        print('Train Epoch: {}, samples: {} \t Last step loss {:.6f} (avg: {:.6f}) \tsec/iter: {:.4f}'.format(
            epoch + 1, n_samples, loss.item(), train_loss / n_samples, time_iter))


def evaluate(data, gnn_steps):
    model.eval()
    node_feat, adj_mat, adj_mask, vs, policy_dict = data

    policy_dict['p'] = policy_dict['p'].to(args.device)
    policy_dict['r'] = policy_dict['r'].to(args.device)
    policy_dict['policy'] = policy_dict['policy'].to(args.device)

    # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
    # node_feat.shape: value_iter_steps, a, s, 2  (v, r)
    # adj_mat.shape: a, s, s, 2 (p, gamma)
    num_actions = node_feat.shape[1]
    iteration_steps = node_feat.shape[0]
    input_node_feat = node_feat[0]  # a,s,2

    values = torch.zeros(gnn_steps, node_feat.shape[2], 1, device=args.device)

    accs = [[] for _ in range(gnn_steps)]
    gt_accs = [[] for _ in range(gnn_steps)]
    losses = [[] for _ in range(gnn_steps)]
    gt_losses = [[] for _ in range(gnn_steps)]

    for step in range(iteration_steps - 1):
        outputs = model((input_node_feat, adj_mat, adj_mask))

        values[0] += outputs[0]
        for i in range(gnn_steps - 1):
            values[i + 1] = values[i] + outputs[i + 1]

        # output: s, 1 -> a,s,1
        # print("vs delta, Loss ", vs[step+1]-vs[step], loss.item())
        input_node_feat = torch.cat((outputs[0].unsqueeze(dim=0).repeat(num_actions, 1, 1) +  # a, s, 1
                                     input_node_feat[:, :, 0:1],
                                     node_feat[step + 1, :, :, 1:2]), dim=-1)
        for i in range(min(gnn_steps, iteration_steps - 1 - step)):
            losses[i] += [loss_fn(values[i], vs[-1]).detach().cpu().item()]
            gt_losses[i] += [loss_fn(vs[step + i], vs[-1]).detach().cpu().item()]
            gt_policy = find_policy(policy_dict['p'], policy_dict['r'], policy_dict['discount'], vs[step + i])
            gt_accs[i] += [100. * torch.eq(gt_policy, policy_dict['policy']).detach().cpu().sum() / len(outputs[i])]
            predicted_policy = find_policy(policy_dict['p'], policy_dict['r'], policy_dict['discount'],
                                           values[i].squeeze())
            accs[i] += [100. * torch.eq(predicted_policy, policy_dict['policy']).detach().cpu().sum() / len(outputs[i])]

    if epoch % 10 == 0:
        for i in range(gnn_steps):
            print('Test set (step {}, epoch {}): \t Last step accuracy {} \t Last step loss {:.6f} ,'
                  ' Average loss: {:.6f} \n'.format(i, epoch + 1, accs[i][-1], losses[i][-1],
                                                    np.mean(np.array(losses[i]))))

    return [l[-1] for l in losses], [a[-1] for a in accs], losses, accs, gt_losses, gt_accs


def uni_evaluate(data):
    model.eval()
    node_feat, adj_mat, adj_mask, vs, policy_dict = data

    policy_dict['p'] = policy_dict['p'].to(args.device)
    policy_dict['r'] = policy_dict['r'].to(args.device)
    policy_dict['policy'] = policy_dict['policy'].to(args.device)

    # node_feat.shape: value_iter_steps, s, 1+a  (v, r_1...a)
    # node_feat.shape: value_iter_steps, a, s, 1+a  (v, r_1...a)
    # adj_mat.shape: a, s, s, 2 (p, gamma)
    iteration_steps = node_feat.shape[0]
    input_node_feat = node_feat[0]  # s, 1+a

    values = torch.zeros(node_feat.shape[1], 1, device=args.device)
    accs = []
    gt_accs = []
    losses = []
    gt_losses = []
    for step in range(iteration_steps - 1):
        output = model((input_node_feat, adj_mat, adj_mask))
        values += output

        # output: s, 1 -> a,s,1
        # print("vs delta, Loss ", vs[step+1]-vs[step], loss.item())
        input_node_feat = torch.cat((output +  # s, 1
                                     input_node_feat[:, 0:1],
                                     node_feat[step + 1, :, 1:]), dim=-1)

        losses += [loss_fn(values, vs[-1]).detach().cpu().item()]
        gt_losses += [loss_fn(vs[step], vs[-1]).detach().cpu().item()]

        gt_policy = find_policy(policy_dict['p'], policy_dict['r'], policy_dict['discount'], vs[step])
        gt_accs += [100. * torch.eq(gt_policy, policy_dict['policy']).detach().cpu().sum() / len(output)]
        predicted_policy = find_policy(policy_dict['p'], policy_dict['r'], policy_dict['discount'], values.squeeze())
        accs += [100. * torch.eq(predicted_policy, policy_dict['policy']).detach().cpu().sum() / len(output)]

    if epoch % 10 == 0:
        print('Test set (epoch {}): \t Last step accuracy {} \t Last step loss {:.6f} , Average loss: {:.6f} \n'.format(
            epoch + 1, accs[-1], losses[-1], np.mean(np.array(losses))))
    return losses[-1], accs[-1], losses, accs, gt_losses, gt_accs


parser = argparse.ArgumentParser(description='Graph Convolutional Networks')
parser.add_argument('--num_train_graphs', type=int, default=100, help='Number of graphs used for training')
parser.add_argument('--num_test_graphs', type=int, default=40, help='Number of graphs used for testing')

parser.add_argument('--test_state_action_tuple', type=int, default=[(20, 5), (20, 10), (20, 20),
                                                                    (50, 5), (50, 10), (50, 20),
                                                                    (100, 5), (100, 10), (100, 20)])

parser.add_argument('--train_num_states', type=int, default=20)
parser.add_argument('--train_num_actions', type=int, default=8)

parser.add_argument('--train_graph_type', type=str, default=None,
                    choices=['maze8', 'maze16', 'maze28', 'cartpole', 'freeway', 'det',
                             'random', 'grid', 'caveman', 'erdos', 'ladder', 'line'])
parser.add_argument('--test_graph_type', type=str, default=None)

# For cartpole
parser.add_argument('--cartpole_depth', type=int, default=10, help='Graph depth')
parser.add_argument('--cartpole_delta', type=float, default=0.1)
parser.add_argument('--cartpole_accel', type=float, default=0.05)
parser.add_argument('--cartpole_thresh', type=float, default=0.5)

# For freeway
parser.add_argument('--freeway_N', type=int, default=15, help='Number of states at every stage')
parser.add_argument('--freeway_stages', type=int, default=3)
parser.add_argument('--freeway_max_jumpback', type=int, default=5)
parser.add_argument('--freeway_p_jumpback_if_hit', type=float, default=0.8)
parser.add_argument('--freeway_p_jumpback', type=float, default=0.2)


parser.add_argument('--epsilon', type=float, default=1e-4, help='termination condition (difference between two '
                                                                'consecutive values)')

parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dim for node/edge')
parser.add_argument('--message_function', type=str, default=None)
parser.add_argument('--message_function_depth', type=int, default=None)
parser.add_argument('--neighbour_state_aggr', type=str, default=None)
parser.add_argument('--gnn_steps', type=int, default=1, help="How many GNN propagation steps are applied ")

parser.add_argument('--action_aggr', type=str, default='max')

parser.add_argument('--patience', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')

parser.add_argument('--load_model', type=str, default=None)
# TODO: run with different seeds and do avg/std
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--save_dir', type=str, default='results')
parser.add_argument('--on_cpu', action='store_true')
parser.add_argument('--sparse_gnn', action='store_true')

parser.add_argument('--graph_data_format', type=str, choices=['separate', 'uni'], default='separate',
                    help='treat each action type differentyl or unified through integrating it as edge information')

parser.add_argument('--msg_activation', action='store_true', default=False, help='Whether to apply ReLU to messages')
parser.add_argument('--layernorm', action='store_true', default=False,
                    help='Whether to apply Layernorm to new representation')



def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True


args = parser.parse_args()

if args.on_cpu:
    args.save_dir = args.save_dir + 'cpu_'
print(args)
args.save_dir = args.save_dir + args.train_graph_type + '_' +  args.test_graph_type + '_' + str(args.gnn_steps) + "step" + args.message_function + \
                '_neighbaggr_' + args.neighbour_state_aggr + '_hidden_' + str(args.hidden_dim)
if args.layernorm:
    args.save_dir += "_layernorm"
if args.train_graph_type in ['det', 'erdos']:
    args.save_dir += '_num_actions' + str(args.train_num_actions)
args.device = torch.device("cuda" if not args.on_cpu and torch.cuda.is_available() else "cpu")

i = 0
while os.path.exists(args.save_dir + "_" + str(i)):
    i += 1
args.save_dir = args.save_dir + "_" + str(i)

# Creates an experimental directory and dumps all the args to a text file
os.makedirs(args.save_dir, exist_ok=True)

print("\nPutting log in %s" % args.save_dir)
argsdict = args.__dict__
argsdict['save_dir'] = args.save_dir
exp_config_file = open(os.path.join(args.save_dir, 'exp_config.txt'), 'w')
for key in sorted(argsdict):
    print(key + '    ' + str(argsdict[key]) + '\n', file=exp_config_file)

edge_features = 2
node_features = 2

if args.sparse_gnn:
    if args.graph_data_format == 'separate':
        from sparse_models import SparseMPNN

        MPNN = SparseMPNN
    elif args.graph_data_format == 'uni':
        from uni_sparse_models import UniSparseMPNN

        MPNN = UniSparseMPNN
        edge_features = 2 + args.train_num_actions
        node_features = 1 + args.train_num_actions
        evaluate = uni_evaluate
    else:
        raise NotImplementedError
else:
    from dense_models import DenseMPNN

    MPNN = DenseMPNN

model = MPNN(node_features=2,
             edge_features=2,
             hidden_dim=args.hidden_dim,
             out_features=1,
             message_function=args.message_function,
             message_function_depth=args.message_function_depth,
             neighbour_state_aggr=args.neighbour_state_aggr,
             gnn_steps=args.gnn_steps,
             msg_activation=args.msg_activation,
             layernorm=args.layernorm).to(args.device)

print('\nInitialize model', file=exp_config_file)
print(model, file=exp_config_file)
train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
print('N trainable parameters:', np.sum([p.numel() for p in train_params]), file=exp_config_file)
optimizer = optim.Adam(train_params, lr=args.lr)

set_seed(args.seed)

if args.load_model is not None:
    model.load_state_dict(torch.load(args.load_model + '/mpnn.pt'))
else:
    if args.graph_data_format == 'separate':
        iterable_train_dataset = GraphData(num_states=args.train_num_states, num_actions=args.train_num_actions,
                                           epsilon=args.epsilon, graph_type=args.train_graph_type, seed=args.seed,
                                           device=args.device, sparse=args.sparse_gnn,
                                           # for cartpole
                                           depth=args.cartpole_depth, delta=args.cartpole_delta,
                                           accel=args.cartpole_accel, thresh=args.cartpole_thresh,
                                           # for freeway
                                           N=args.freeway_N, stages=args.freeway_stages,
                                           max_jumpback=args.freeway_max_jumpback,
                                           p_jumpback_if_hit=args.freeway_p_jumpback_if_hit,
                                           p_jumpback=args.freeway_p_jumpback)
    elif args.graph_data_format == 'uni':
        iterable_train_dataset = UniGraphData(num_states=args.train_num_states, num_actions=args.train_num_actions,
                                              epsilon=args.epsilon, graph_type=args.train_graph_type, seed=args.seed,
                                              device=args.device, sparse=args.sparse_gnn)
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(iterable_train_dataset, batch_size=None)
    for epoch in range(args.num_train_graphs):
        train(next(iter(train_loader)), args.gnn_steps)
    torch.save(model.state_dict(), args.save_dir + '/mpnn.pt')

import pickle

test_loop = True
test_step = 0
initial_position = 0  # default
if args.test_graph_type == 'cartpole':
    initial_positions = [i for i in np.arange(-args.cartpole_thresh, args.cartpole_thresh, 0.05)]

while test_loop:
    if args.test_graph_type not in ['maze8', 'maze16', 'maze28', 'cartpole', 'freeway']:
        if test_step < len(args.test_graph_type):
            sa_pair = args.test_state_action_tuple[test_step]
            print("State-action ", sa_pair)
            states, actions = sa_pair
            test_step += 1
            if states == 100 or (states == 50 and actions == 20):
                args.device = torch.device("cpu")
                model.to(args.device)
        else:
            test_loop = False
    else:
        states = None
        actions = None
        if args.test_graph_type == 'cartpole' and test_step < len(initial_positions):
            initial_position = initial_positions[test_step]
            print("Initial position ", initial_position)
            test_step += 1
        else:
            test_loop = False

    torch.cuda.empty_cache()
    if args.graph_data_format == 'separate':
        iterable_test_dataset = GraphData(num_states=states, num_actions=actions, epsilon=args.epsilon,
                                          graph_type=args.test_graph_type, seed=args.seed, sparse=args.sparse_gnn,
                                          depth=args.cartpole_depth, delta=args.cartpole_delta,
                                          accel=args.cartpole_accel, thresh=args.cartpole_thresh,
                                          initial_position=initial_position,
                                          N=args.freeway_N, stages=args.freeway_stages,
                                          max_jumpback=args.freeway_max_jumpback,
                                          p_jumpback_if_hit=args.freeway_p_jumpback_if_hit,
                                          p_jumpback=args.freeway_p_jumpback)
    elif args.graph_data_format == 'uni':
        iterable_test_dataset = UniGraphData(num_states=states, num_actions=actions, epsilon=args.epsilon,
                                             graph_type=args.test_graph_type, seed=args.seed, sparse=args.sparse_gnn)
    else:
        raise NotImplementedError

    test_loader = torch.utils.data.DataLoader(iterable_test_dataset, batch_size=None)

    test_last_losses = [[] for _ in range(args.gnn_steps)]
    test_all_losses = [[] for _ in range(args.gnn_steps)]
    test_last_accs = [[] for _ in range(args.gnn_steps)]
    test_all_accs = [[] for _ in range(args.gnn_steps)]
    all_gt_losses = [[] for _ in range(args.gnn_steps)]
    all_gt_accs = [[] for _ in range(args.gnn_steps)]

    for epoch in range(args.num_test_graphs):
        last_loss, last_acc, losses, accs, gt_losses, gt_accs = evaluate(next(iter(test_loader)), args.gnn_steps)
        for i in range(args.gnn_steps):
            test_last_losses[i] += [last_loss[i]]
            test_last_accs[i] += [last_acc[i]]
            test_all_losses[i] += [losses[i]]
            test_all_accs[i] += [accs[i]]
            all_gt_losses[i] += [gt_losses[i]]
            all_gt_accs[i] += [gt_accs[i]]

    for i in range(args.gnn_steps):
        print("Step {}: States {}, actions {} \t Test last step loss mean {}, std {} \n".format(
            i, states, actions,
            np.mean(
                np.array(test_last_losses[i])),
            np.std(
                np.array(test_last_losses[i]))),
            file=exp_config_file)

        print("Step {}: States {}, actions {} \t Test last step acc mean {}, std {} \n".format(i, states, actions,
                                                                                               np.mean(np.array(
                                                                                                   test_last_accs[i])),
                                                                                               np.std(np.array(
                                                                                                   test_last_accs[i]))),
              file=exp_config_file)

    print('\n', file=exp_config_file)
    results = {
        'losses': test_all_losses,
        'accs': test_all_accs,
        'gt_losses': all_gt_losses,
        'gt_accs': all_gt_accs
    }
    pickle.dump(results,
                open(args.save_dir + '/results_states_' + str(states) + '_actions_' + str(actions) + '.p', 'wb'))