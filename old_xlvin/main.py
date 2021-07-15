import os
import argparse
import gym
import pickle
import numpy as np
import torch
import torch.optim as optim
from gym import spaces

import random

from gnn_executor.sparse_models import SparseMPNN, SparseMessagePassing
from rl.ppo import XLVINPolicy, PPO, gather_rollout
from utils import get_arguments, make_vec_envs, test_vec_envs
from TransE.cartpole_main import train_transe
from TransE.mountaincar_main import train_transe

def evaluate_step_batch(env_type, reward, done):
    if env_type in ['cartpole', 'mountaincar', 'acrobot']:
        passed = torch.sum(reward)
        done_ep = torch.sum(torch.tensor(done, device=args.device))
    return done_ep, passed


def run_fixed_nb_episodes(env, policy, level, rollout, exp_config_file, nb_episodes, env_type,
                          lvl_threshold=95.):
    ep = 0
    obs = env.reset()
    all_passed = 0
    while ep < nb_episodes:
        with torch.no_grad():
            if env_type in ['cartpole', 'mountaincar', 'acrobot',
                    'mini', 'mini-avoid', 'mini-hunt', 'mini-ambush', 'mini-rush']:
                deterministic = True
            else:
                raise NotImplementedError
            value, action, log_probs = policy.act(obs.to(args.device), deterministic=deterministic)
        obs, reward, done, solved_obs = env.step(action)
        reward = torch.tensor(reward, device=args.device)

        done_ep, passed_metric = evaluate_step_batch(env_type=env_type, reward=reward, done=done)
        ep += done_ep
        all_passed += passed_metric

    if env_type in ['cartpole', 'mountaincar', 'acrobot']:
        print("Rollout {} : Number of steps passed {:.3f} from total episodes {} (perc {:.3f})".format(
            rollout, all_passed, ep, (all_passed * 1.0) / (ep * 1.0)))
        print("Rollout {} : Number of steps passed {:.3f} from total episodes {} (perc {:.3f})".format(
            rollout, all_passed, ep, (all_passed * 1.0) / (ep * 1.0)), file=exp_config_file, flush=True)
        return all_passed, ep


if __name__ == '__main__':
    args = get_arguments()
    print("\nPutting log in %s" % args.save_dir_path, flush=True)
    argsdict = args.__dict__
    argsdict['save_dir_path'] = args.save_dir_path
    exp_config_file = open(os.path.join(args.save_dir_path, 'exp_config.txt'), 'w')
    for key in sorted(argsdict):
        print(key + '    ' + str(argsdict[key]) + '\n', file=exp_config_file, flush=True)

    torch.autograd.set_detect_anomaly(True)

    if args.load_model_path is not None and not args.qualitative_value_study:
        load_path = os.path.join(args.load_model_path,
                                 'checkpoint_length' + str(int(args.load_model_lvl)) + '.pt')
        checkpoint = torch.load(load_path, map_location=args.device)
    if args.train_load_model_path is not None:
        load_path = os.path.join(args.train_load_model_path,
                                 'checkpoint_length' + str(int(args.train_load_model_lvl)) + '.pt')
        checkpoint = torch.load(args.train_load_model_path, map_location=args.device)

    if args.model == 'XLVIN':
        if args.env_type == 'cartpole':
            from TransE.cartpole_main import CartPoleTransE as TransE
        elif args.env_type == 'mountaincar':
            from TransE.mountaincar_main import MountainCarTransE as TransE

        transe_model = TransE(embedding_dim=args.transe_embedding_dim, hidden_dim=args.transe_hidden_dim,
                              action_dim=args.env_action_dim,
                              input_dims=args.env_input_dims, sigma=0.5, hinge=1.,
                              encoder_type=args.transe_encoder_type, pool_type=args.transe_pool_type,
                              encoder_num_linear=args.transe_encoder_num_linear,
                              transe_vin_type=args.transe_vin_type).to(args.device)
        if args.pretrained_encoder:
            if args.load_model_path is not None:
                transe_model.load_state_dict(checkpoint['transe_model'])
            else:
                transe_model.load_state_dict(torch.load(args.transe_weights_path, map_location=args.device))

        if args.gnn:
            gnn_model = SparseMPNN(node_features=2,
                                   edge_features=2,
                                   hidden_dim=args.gnn_hidden_dim,
                                   out_features=1,
                                   message_function=args.message_function,
                                   message_function_depth=args.message_function_depth,
                                   neighbour_state_aggr=args.neighbour_state_aggr,
                                   gnn_steps=args.gnn_steps,
                                   msg_activation=args.msg_activation,
                                   layernorm=args.gnn_layernorm).to(args.device)
            if args.load_model_path is not None:
                gnn_model.load_state_dict(checkpoint['gnn_model'])
            elif args.gnn_weights_path is not None:
                gnn_model.load_state_dict(torch.load(os.path.join(args.gnn_weights_path, 'mpnn.pt'),
                                                     map_location=args.device))
            elif args.env_type in ['cartpole', 'mountaincar', 'acrobot']:
                gnn_model.load_state_dict(
                    torch.load('./gnn_executor/trained_model/results' + 'cartpole' + '_' + str(args.gnn_steps) +
                               'stepmpnn_neighbaggr_sum_hidden_' + str(args.gnn_hidden_dim) + '/mpnn.pt',
                               map_location=args.device))
            else:
                gnn_name = args.env_type + '_' + str(args.gnn_steps) + \
                    'stepmpnn_neighbaggr_sum_hidden_' + str(args.gnn_hidden_dim)
                if args.gnn_layernorm:
                    gnn_name += '_layernorm'

                gnn_model.load_state_dict(
                    torch.load('./gnn_executor/trained_model/results' + gnn_name + '/mpnn.pt',
                        map_location=args.device))
            gnn_layer = gnn_model.mps
        else:
            gnn_layer = None

        policy = XLVINPolicy(action_space=spaces.Discrete(args.env_action_dim), gamma=args.gamma, transe=transe_model,
                             edge_feat=args.env_action_dim + 1,  # +1 as we add gamma as edge_feature too
                             transe_hidden_dim=args.transe_embedding_dim,
                             gnn_hidden_dim=args.gnn_hidden_dim,
                             gnnx=gnn_layer, num_processes=args.num_processes,
                             include_gnn=args.gnn, cat_enc_gnn=args.cat_enc_gnn, full_cat_gnn=args.full_cat_gnn,
                             freeze_encoder=args.freeze_encoder, freeze_gnn=args.freeze_gnn,
                             transe2gnn=args.transe2gnn,
                             gnn_decoder=args.gnn_decoder,
                             gnn_steps=args.gnn_steps,
                             vin_attention=args.vin_attention,
                             graph_detach=args.graph_detach,
                             transe_vin_type=args.transe_vin_type).to(args.device)
        params = list(policy.parameters())
        # policy.parameters already includes encoder's and gnn's parameters
        optimizer = optim.Adam(params)  # TODO: params lr, weight_decay

        if args.load_model_path is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
            policy.load_state_dict(checkpoint['policy'])
        if args.train_load_model_path is not None:
            transe_model.load_state_dict(checkpoint['transe_model'])
            gnn_model.load_state_dict(checkpoint['gnn_model'])
            policy.load_state_dict(checkpoint['policy'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    if args.debug:
        for name, param in policy.named_parameters():
            if param.requires_grad:
                print(name, param.data, flush=True)

    ppo = PPO(actor_critic=policy,
              clip_param=args.clip_param, ppo_epoch=args.ppo_epoch, num_mini_batch=args.num_mini_batch,
              value_loss_coef=args.value_loss_coef, entropy_coef=args.entropy_coef,
              transe_loss_coef=0. if args.retrain_transe else args.transe_loss_coef,
              optimizer=optimizer,
              max_grad_norm=args.max_grad_norm,
              mini_batch_size=args.mini_batch_size,
              transe_detach=args.transe_detach)

    print('\nInitialize model', file=exp_config_file)
    print(policy, file=exp_config_file)
    train_params = list(filter(lambda p: p.requires_grad, policy.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]), file=exp_config_file, flush=True)

    # testing
    if args.load_model_path is not None:
        results = test_maze(test_maze_size=args.test_maze_size,
                            test_indices=test_indices,
                            policy=policy, exp_config_file=exp_config_file)
    else:
        train_indices = {0 : 'N/A'}
        for length in sorted(train_indices.keys()):
            curriculum_train_indices = ['N/A']

            env = make_vec_envs(maze_size=args.train_maze_size, train_maze=True,
                                maze_indices_list=curriculum_train_indices,
                                device=args.device,
                                num_processes=args.num_processes,
                                env_type=args.env_type,
                                explode_maze=args.explode_maze)

            test_curriculum_lvl_env = make_vec_envs(maze_size=args.train_maze_size, train_maze=True,
                                                    maze_indices_list=list(train_indices[length]),
                                                    device=args.device,
                                                    num_processes=args.num_processes,
                                                    env_type=args.env_type,
                                                    fail_on_repeated_state=False,
                                                    explode_maze=args.explode_maze)

            done = False
            all_ep = 0
            all_passed = 0
            for j in range(args.num_rollouts):
                if args.env_type in ['cartpole', 'mountaincar', 'acrobot', 'sokoban', 'freeway', 'enduro']:
                    rollouts = gather_fixed_ep_rollout(env=env, policy=policy, num_episodes=args.num_train_episodes,
                                                       gamma=args.gamma,
                                                       num_processes=args.num_processes,
                                                       device=args.device)
           
                for k in range(args.ppo_updates):
                    value_loss_epoch, action_loss_epoch, dist_entropy_epoch, transe_loss_epoch = ppo.update(rollouts)
                    if args.retrain_transe:
                        train_transe(transe=transe_model, optimizer=optimizer, num_episodes=args.num_train_episodes,
                                     exp_config_file=exp_config_file,
                                     transe_loss_coef=args.transe_loss_coef)

                    passed_lvl_metric, ep_done = run_fixed_nb_episodes(env=test_curriculum_lvl_env,
                                                              policy=policy, level=length, rollout=j,
                                                              exp_config_file=exp_config_file,
                                                              nb_episodes=args.lvl_nb_deterministic_episodes,
                                                              env_type=args.env_type,
                                                              lvl_threshold=args.lvl_threshold)

                if j == args.num_rollouts - 1 or (args.env_type == 'maze' and passed_lvl_metric):
                    # save models
                    save_path = os.path.join(args.save_dir_path, 'checkpoint_length' + str(int(length)) + '.pt')
                    if args.model == 'XLVIN':
                        torch.save({
                            'length': length,
                            'transe_model': transe_model.state_dict(),
                            'gnn_model': gnn_model.state_dict() if args.gnn else None,
                            'policy': policy.state_dict(),
                            'optimizer': optimizer.state_dict(),
                        }, save_path)
