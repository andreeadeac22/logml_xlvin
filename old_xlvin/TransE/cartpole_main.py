import torch
import gym
import argparse
from torch import nn

import os
import numpy as np


class CartPoleTransE(nn.Module):
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim, hinge=1., sigma=0.5,
                 encoder_type=None, pool_type=None, encoder_num_linear=None, transe_vin_type=None):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.hinge = hinge
        self.sigma = sigma

        self.pos_loss = 0
        self.neg_loss = 0

        self.encoder = Encoder(input_dims, hidden_dim, embedding_dim)

        self.transition_model = TransitionMLP(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim)

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""
        norm = 0.5 / (self.sigma ** 2)
        if no_trans:
            diff = state - next_state
        else:
            pred_trans = self.transition_model(state, action)
            diff = state + pred_trans - next_state

        return norm * diff.pow(2).sum(1)

    def transition_loss(self, state, action, next_state, transe_detach=False):
        enc_state = self.encoder(state)
        enc_next_state = self.encoder(next_state)
        if transe_detach:
            enc_state = enc_state.detach()
            enc_next_state = enc_next_state.detach()
        return self.energy(enc_state, action, enc_next_state).mean()

    def contrastive_loss(self, obs, action, next_obs):
        state = self.encoder(obs)
        next_state = self.encoder(next_obs)

        self.pos_loss = self.energy(state, action, next_state)
        self.pos_loss = self.pos_loss.mean()

        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)

        neg_obs = next_obs.clone().detach()
        neg_obs[:, 1] = neg_obs[:, 1][perm]
        neg_state = self.encoder(neg_obs)

        zeros = torch.zeros_like(self.pos_loss)

        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state)).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        return self.encoder(obs)


class Encoder(nn.Module):
    """Encodes state to vector of hidden_dim"""
    def __init__(self, input_dim, hidden_dim, output_dim, act_fn=nn.ReLU(),
                 act_fn_hid=nn.ReLU()):
        super().__init__()
        self.dims = output_dim

        self.fc1 = nn.Linear(input_dim, 2*hidden_dim)
        self.fc1_act = act_fn
        self.fc2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc2_act = act_fn
        self.fc3=nn.Linear(hidden_dim, output_dim)

    def forward(self, obs):
        # print(f'obs {obs.shape}')
        h = self.fc1_act(self.fc1(obs))
        h = self.fc2_act(self.fc2(h))
        h = self.fc3(h)
        return h


class TransitionMLP(torch.nn.Module):
    """transition function."""

    def __init__(self, input_dim, hidden_dim, action_dim=4, act_fn=nn.ReLU()):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.transition_mlp = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, input_dim))

    def _action_to_one_hot(self, action):
        """Get one-hot encoding of index tensors."""
        zeros = torch.zeros(
            action.size()[0], self.action_dim, dtype=torch.float32,
            device=action.device)
        return zeros.scatter_(1, action.unsqueeze(1), 1)

    def forward(self, state, action):
        # state: [batch_size, embeddin_dim]
        # action: [batch_size]

        # transform action to one-hot
        action_vector = self._action_to_one_hot(action)

        action_vector = action_vector.view(-1, self.action_dim)

        # Attach action to state
        state_action = torch.cat([state, action_vector], dim=-1)

        return self.transition_mlp(state_action)


def train_transe(transe, optimizer, num_episodes, exp_config_file, transe_loss_coef=1., batch_size=128):
    env = gym.make("CartPole-v0")

    overall_loss = 0.
    episodes = 0
    prev_state = env.reset()
    obs = []
    next_obs = []
    actions = []
    while episodes < num_episodes:
        action = torch.randint(0, 2, (1,))
        state, reward, done, _ = env.step(action.item())
        if done:
            episodes += 1
            state = env.reset()
        else:
            obs += [torch.Tensor(prev_state)]
            actions += [action]
            next_obs += [torch.Tensor(state)]
        prev_state = state

    obs = torch.stack(obs, 0)
    next_obs = torch.stack(next_obs, 0)
    actions = torch.stack(actions, 0)
    num_batches = int(obs.shape[0]/ batch_size)
    print("Obs ", obs.shape)
    print("Num batches ", num_batches)
    for i in range(num_batches):
        optimizer.zero_grad()
        batch_obs = obs[i*batch_size: (i+1)*batch_size]
        batch_next_obs = next_obs[i*batch_size: (i+1)*batch_size]
        batch_actions = actions[i*batch_size: (i+1)*batch_size]

        loss = transe_loss_coef * transe.contrastive_loss(batch_obs, batch_actions.squeeze(), batch_next_obs)  #/ batch_obs.shape[0]
        overall_loss += loss.detach().item()
        if i % 5 == 0:
            print("Loss ", overall_loss)
            print("Loss ", overall_loss, file=exp_config_file)

            overall_loss = 0.
        loss.backward(retain_graph=True)
        optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_episodes', type=int, default=50000)
    parser.add_argument('--save_dir', type=str, default='./cartpole/')

    args = parser.parse_args()

    transe = CartPoleTransE(embedding_dim=50, input_dims=4, hidden_dim=64, action_dim=2)
    optimizer = torch.optim.Adam(transe.parameters())

    os.makedirs(args.save_dir, exist_ok=True)
    exp_config_file = open(os.path.join(args.save_dir, 'exp_config.txt'), 'w')
    argsdict = args.__dict__
    for key in sorted(argsdict):
        print(key + '    ' + str(argsdict[key]) + '\n', file=exp_config_file, flush=True)
    print(transe, file=exp_config_file)

    train_transe(transe=transe, optimizer=optimizer, num_episodes=args.num_episodes, exp_config_file=exp_config_file,
                 batch_size=args.batch_size)
    torch.save(transe.state_dict(), os.path.join(args.save_dir, 'cartpole_transe.pt'))
