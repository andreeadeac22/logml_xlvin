import argparse
import os

import gym
import torch

from logml_xlvin.src.encoder import Encoder
from logml_xlvin.src.executor import SparseMPNN
from logml_xlvin.src.transition import Transition
from logml_xlvin.src.xlvin_policy import XLVINModel


class CartPoleGame:
    def __init__(self, args):
        self.env = gym.make("MountainCar-v0")
        # TODO: check dimensions
        self._encoder = Encoder(input_dim=4, hidden_dim=64, output_dim=50)
        self._transition = Transition(embedding_dim=50, hidden_dim=64, action_dim=2)
        self._executor = SparseMPNN(
            node_features=2,
            edge_features=2,
            hidden_dim=args.hidden_dim,
            out_features=1,
            message_function=args.message_function,
            message_function_depth=args.message_function_depth,
            neighbour_state_aggr=args.neighbour_state_aggr,
            gnn_steps=args.gnn_steps,
            msg_activation=args.msg_activation,
            layernorm=args.layernorm
        )
        # edge_features_dim, gamma ?
        self.model = XLVINModel(
            self.env.action_space, self._encoder, self._transition, 64, -1, self._executor, 2, True, False, False, -1
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.num_episodes = args.num_episodes
        self.exp_config_file = self._create_exp_config_file(args)

    def _create_exp_config_file(self, args):
        # TODO: refactor
        os.makedirs(args.save_dir, exist_ok=True)
        exp_config_file = open(os.path.join(args.save_dir, 'exp_config.txt'), 'w')
        argsdict = args.__dict__
        for key in sorted(argsdict):
            print(key + '    ' + str(argsdict[key]) + '\n', file=exp_config_file, flush=True)
        print(self._transition, file=exp_config_file) # TODO: not only transition

        return exp_config_file

    def train_model(self, transe_loss_coef=1., batch_size=128):
        # TODO
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_episodes', type=int, default=50000)
    parser.add_argument('--save_dir', type=str, default='./cartpole/')

    args = parser.parse_args()

    cartpole_game = CartPoleGame(args)
    cartpole_game.train_model()

    torch.save(cartpole_game.model.state_dict(), os.path.join(args.save_dir, 'cartpole_xvlin.pt'))
