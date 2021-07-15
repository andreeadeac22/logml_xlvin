import argparse
import os

import gym
import torch

from logml_xlvin.src.encoder import Encoder
from logml_xlvin.src.executor import SparseMPNN
from logml_xlvin.src.ppo import PPO, gather_fixed_ep_rollout
from logml_xlvin.src.transition import Transition, retrain_transition_on_cartpole
from logml_xlvin.src.xlvin_policy import XLVINModel
import numpy as np


class CartPoleGame:
    def __init__(self, args):
        self.env = gym.make("MountainCar-v0")
        # TODO: check dimensions
        self._encoder = Encoder(input_dim=4, hidden_dim=64, output_dim=50)
        self._transition = Transition(embedding_dim=50, hidden_dim=64, action_dim=2)
        gnn_steps=2
        self._executor = SparseMPNN(
            node_features=2,
            edge_features=2,
            hidden_dim=50,
            out_features=1,
            message_function="mpnn",
            message_function_depth=2,
            neighbour_state_aggr="sum",
            gnn_steps=gnn_steps
        )
        # edge_features_dim, gamma ?
        self.model = XLVINModel(
            self.env.action_space, self._encoder, self._transition, 64, 2, self._executor, 2, True, False, False, args.gamma
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
        print(self._transition, file=exp_config_file)  # TODO: not only transition

        return exp_config_file

    def train_model(self):
        policy = self.model  # TODO refactor
        ppo = PPO(
            actor_critic=policy,
            clip_param=0.2,
            ppo_epoch=10,
            num_mini_batch=5,
            value_loss_coef=0.8,
            entropy_coef=0.1,
            transe_loss_coef=0.,
            optimizer=self.optimizer
        )

        print('\nInitialize model', file=self.exp_config_file)
        print(policy, file=self.exp_config_file)
        train_params = list(filter(lambda p: p.requires_grad, policy.parameters()))
        print(
            'N trainable parameters:', np.sum([p.numel() for p in train_params]), file=self.exp_config_file, flush=True
        )

        # testing
        if args.load_model_path is not None:
            #results = test_maze(test_maze_size=args.test_maze_size,
            #                    test_indices=test_indices,
            #                   policy=policy, exp_config_file=exp_config_file)
            pass
        else:
            train_indices = {0: 'N/A'}
            for length in sorted(train_indices.keys()):

                test_env = gym.make("MountainCar-v0")

                done = False
                all_ep = 0
                all_passed = 0
                for j in range(args.num_rollouts):
                    rollouts = gather_fixed_ep_rollout(
                        env=self.env,
                        policy=policy,
                        num_episodes=args.num_episodes,
                        gamma=args.gamma,
                        num_processes=args.num_processes,
                        device=args.device
                    )

                    for k in range(args.ppo_updates):
                        value_loss_epoch, action_loss_epoch, dist_entropy_epoch, transe_loss_epoch = ppo.update(
                            rollouts
                        )
                        if args.retrain_transe:
                            retrain_transition_on_cartpole(
                                transition=policy.transition,
                                optimizer=self.optimizer,
                                num_episodes=args.num_episodes,
                                exp_config_file=self.exp_config_file,
                                transe_loss_coef=args.transe_loss_coef
                            )

                        passed_lvl_metric, ep_done = self.run_fixed_nb_episodes(
                            env=test_env,
                            rollout=j,
                            exp_config_file=self.exp_config_file,
                            nb_episodes=args.lvl_nb_deterministic_episodes
                        )

                    if j == args.num_rollouts - 1 or (args.env_type == 'maze' and passed_lvl_metric):
                        # save models
                        save_path = os.path.join(args.save_dir_path, 'checkpoint_length' + str(int(length)) + '.pt')
                        if args.model == 'XLVIN':
                            torch.save({
                                'length': length,
                                'transe_model': policy.transition.state_dict(),
                                'gnn_model': policy.executor.state_dict() if args.gnn else None,
                                'policy': policy.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                            }, save_path)

    def run_fixed_nb_episodes(self, env, rollout, exp_config_file, nb_episodes):
        policy = self.model
        ep = 0
        obs = env.reset()
        all_passed = 0
        while ep < nb_episodes:
            with torch.no_grad():
                deterministic = True
                value, action, log_probs = policy.act(obs.to(args.device), deterministic=deterministic)
            obs, reward, done, solved_obs = env.step(action)
            reward = torch.tensor(reward, device=args.device)

            done_ep, passed_metric = CartPoleGame._evaluate_step_batch(reward, done)
            ep += done_ep
            all_passed += passed_metric

        print("Rollout {} : Number of steps passed {:.3f} from total episodes {} (perc {:.3f})".format(
            rollout, all_passed, ep, (all_passed * 1.0) / (ep * 1.0)))
        print("Rollout {} : Number of steps passed {:.3f} from total episodes {} (perc {:.3f})".format(
            rollout, all_passed, ep, (all_passed * 1.0) / (ep * 1.0)), file=exp_config_file, flush=True)
        return all_passed, ep

    @staticmethod
    def _evaluate_step_batch(reward, done):
        passed = torch.sum(reward)
        done_ep = torch.sum(torch.tensor(done, device=args.device))
        return done_ep, passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_episodes', type=int, default=50000)
    parser.add_argument('--save_dir', type=str, default='./cartpole/')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--num_rollouts', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    cartpole_game = CartPoleGame(args)
    cartpole_game.train_model()

    torch.save(cartpole_game.model.state_dict(), os.path.join(args.save_dir, 'cartpole_xvlin.pt'))
