import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np


class TransE(nn.Module):
    def __init__(self, embedding_dim, input_dims, hidden_dim, action_dim,
                 hinge=1., sigma=0.5, new_sample=False, num_conv=3, encoder_type='general', pool_type='mean',
                 encoder_num_linear=2, transe_vin_type='pool'):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.hinge = hinge
        self.sigma = sigma

        self.pos_loss = 0
        self.neg_loss = 0

        num_channels = input_dims[0]
        width_height = input_dims[1:]
        self.transe_vin_type = transe_vin_type

        if encoder_type == 'original':
            self.encoder = Encoder(width_height, num_channels, hidden_dim,
                                   embedding_dim, num_conv=num_conv)
        elif encoder_type == 'general':
            self.encoder = SizeAgnosticEncoder(width_height, num_channels, hidden_dim,
                                               embedding_dim, num_conv=num_conv, pool_type=pool_type,
                                               encoder_num_linear=encoder_num_linear,
                                               transe_vin_type=transe_vin_type)
        else:
            raise NotImplementedError

        self.transition_model = TransitionMLP(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim)

        self.width = width_height[0]
        self.height = width_height[1]

        if new_sample:
            self.contrastive_loss = self.contrastive_loss_new
        else:
            self.contrastive_loss = self.contrastive_loss_old

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
        if self.transe_vin_type in ["slice_pool", "vin_slice_pool"]:
            enc_state, _ = self.encoder(state)
            enc_next_state, _ = self.encoder(next_state)
        else:
            enc_state = self.encoder(state)
            enc_next_state = self.encoder(next_state)
        if transe_detach:
            enc_state = enc_state.detach()
            enc_next_state = enc_next_state.detach()
        return self.energy(enc_state, action, enc_next_state).mean()

    def contrastive_loss_old(self, obs, action, next_obs):
        state = self.encoder(obs)
        next_state = self.encoder(next_obs)

        self.pos_loss = self.energy(state, action, next_state)

        transitions = self.transition_model(state, action)

        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)

        neg_obs = next_obs.clone().detach()
        neg_obs[:, 1] = neg_obs[:, 1][perm]
        neg_state = self.encoder(neg_obs)

        zeros = torch.zeros_like(self.pos_loss)
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state))

        self.pos_loss = (self.pos_loss).mean()
        self.neg_loss = self.neg_loss.mean()
        self.force_transitions = torch.max(zeros, 0.5 - transitions.abs().sum(1))
        weight = (obs - next_obs).abs().sum((1, 2, 3))
        loss = self.pos_loss + self.neg_loss + self.force_transitions @ weight.T

        return loss

    def contrastive_loss_new(self, grid, start, action, end, fake, reward):
        obs = torch.cat((grid, start, reward), axis=1)
        next_obs = torch.cat((grid, end, reward), axis=1)
        neg_obs = torch.cat((grid, fake, reward), axis=1)

        state = self.encoder(obs)
        next_state = self.encoder(next_obs)
        self.pos_loss = self.energy(state, action, next_state)
        self.pos_loss = self.pos_loss.mean()

        neg_state = self.encoder(neg_obs)
        zeros = torch.zeros_like(self.pos_loss)
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state))
        self.neg_loss = self.neg_loss.mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def forward(self, obs):
        return self.encoder(obs)


class Encoder(nn.Module):
    """Encodes state to vector of hidden_dim"""

    def __init__(self, width_height, num_channels, hidden_dim, output_dim, act_fn=nn.ReLU(),
                 act_fn_hid=nn.ReLU(), num_conv=3):
        super().__init__()

        self.num_conv = num_conv

        self.input_dim = np.prod(width_height)

        self.dims = output_dim

        self.conv_layers = [
            nn.Conv2d(num_channels, hidden_dim, (3, 3), stride=1, padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim),
            act_fn_hid,
        ]
        for i in range(num_conv - 2):
            self.conv_layers += [
                nn.Conv2d(hidden_dim, hidden_dim, (3, 3), stride=1, padding=(1, 1)),
                nn.BatchNorm2d(hidden_dim),
                act_fn_hid
            ]
        self.conv_layers += [
            nn.Conv2d(hidden_dim, 1, (1, 1), stride=1),
            act_fn
        ]

        self.cnn = nn.Sequential(*self.conv_layers)

        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.fc1_act = act_fn

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_act = act_fn

        self.ln = nn.LayerNorm(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, obs):
        h = self.cnn(obs)
        # TODO: replace flat with pool, to generalize to 16x16
        h_flat = h.view(-1, self.input_dim)
        h = self.fc1_act(self.fc1(h_flat))
        h = self.fc2_act(self.ln(self.fc2(h)))
        return self.fc3(h)


class SizeAgnosticEncoder(nn.Module):
    """ Encodes state to vector of hidden_dim, replacing the hidden_dim -> input -> hidden_dim from the original encoder
        with max/mean global pooling, so that it can be later applied to 16x16 mazes. """

    def __init__(self, width_height, num_channels, hidden_dim, output_dim, act_fn=nn.ReLU(),
                 act_fn_hid=nn.ReLU(), num_conv=3, pool_type='mean', encoder_num_linear=2,
                 transe_vin_type='pool'):
        super().__init__()

        self.num_conv = num_conv

        self.hidden_dim = hidden_dim

        self.dims = output_dim

        if 'vin' in transe_vin_type:
            config = {
                'batch_size': 128,
                'datafile': 'dataset/gridworld_8x8.npz',
                'k': 5,
                'l_h': 150,
                'l_i': 3,
                'l_q': 10,
                'lr': 0.005
            }
            self.config = config
            self.hidden_layer = nn.Conv2d(in_channels=config['l_i'],
                out_channels=config['l_h'],
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=True)
            self.r = nn.Conv2d(
                in_channels=config['l_h'],
                out_channels=1,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=False)
            self.q = nn.Conv2d(
                in_channels=1,
                out_channels=hidden_dim,
                kernel_size=(3, 3),
                stride=1,
                padding=1,
                bias=False)
            self.fc = nn.Linear(in_features=config['l_q'], out_features=8, bias=False)
            self.w = Parameter(
                torch.zeros(hidden_dim, 1, 3, 3), requires_grad=True)
            self.sm = nn.Softmax(dim=1)

        else:
            self.conv_layers = [
                nn.Conv2d(num_channels, hidden_dim, (3, 3), stride=1, padding=(1, 1)),
                nn.BatchNorm2d(hidden_dim),
                act_fn_hid,
            ]
            for i in range(num_conv - 2):
                self.conv_layers += [
                    nn.Conv2d(hidden_dim, hidden_dim, (3, 3), stride=1, padding=(1, 1)),
                    nn.BatchNorm2d(hidden_dim),
                    act_fn_hid
                ]

            self.cnn = nn.Sequential(*self.conv_layers)

        # self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc1_act = act_fn
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc2_act = act_fn
        # self.ln = nn.LayerNorm(hidden_dim)
        # self.fc3 = nn.Linear(hidden_dim, output_dim)

        if transe_vin_type == 'cat':
            first_hidden = 2 * hidden_dim
        else:
            first_hidden = hidden_dim

        tail_layers = [nn.Linear(first_hidden, hidden_dim), act_fn]
        if encoder_num_linear == 2:
            tail_layers += [nn.Linear(hidden_dim, hidden_dim), act_fn, nn.LayerNorm(hidden_dim)]
        tail_layers += [nn.Linear(hidden_dim, output_dim)]
        self.tail = nn.Sequential(*tail_layers)
        self.pool_type = pool_type
        self.transe_vin_type = transe_vin_type




    def forward(self, obs):
        if 'vin' in self.transe_vin_type:
            h = self.hidden_layer(obs)
            r = self.r(h)
            q = self.q(r)
            v, _ = torch.max(q, dim=1, keepdim=True)
            for i in range(0, self.config['k'] - 1):
                q = F.conv2d(
                    torch.cat([r, v], 1),
                    torch.cat([self.q.weight, self.w], 1),
                    stride=1,
                    padding=1)
                v, _ = torch.max(q, dim=1, keepdim=True)

            h = F.conv2d(
                torch.cat([r, v], 1),
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)
        else:
            h = self.cnn(obs)

        if self.transe_vin_type in ['slice', 'cat', 'slice_pool', 'vin_slice_pool']:
            flat_img = obs[:,1].view(obs.shape[0], -1)
            
            maxind = torch.argmax(flat_img, dim=-1)
            S1, S2 = maxind // obs.shape[-2], maxind % obs.shape[-2]
            
            slice_s1 = S1.long().expand(obs.shape[-1], 1, self.hidden_dim, h.size(0))
            slice_s1 = slice_s1.permute(3, 2, 1, 0)
            h_out = h.gather(2, slice_s1).squeeze(2)

            slice_s2 = S2.long().expand(1, self.hidden_dim, h.size(0))
            slice_s2 = slice_s2.permute(2, 1, 0)

            h_flat_vin = h_out.gather(2, slice_s2).squeeze(2)

        if self.transe_vin_type in ['pool', 'cat', 'slice_pool', 'vin_slice_pool', 'vin_pool']:
            h_flat_pool = h.view(h.shape[0], self.hidden_dim, -1)
            if self.pool_type == 'max':
                h_flat_pool, _ = torch.max(h_flat_pool, dim=-1)
            elif self.pool_type == 'mean':
                h_flat_pool = torch.mean(h_flat_pool, dim=-1)

        if self.transe_vin_type == 'slice':
            h_flat = h_flat_vin
        elif self.transe_vin_type in ['pool', 'vin_pool']:
            h_flat = h_flat_pool
        elif self.transe_vin_type == 'cat':
            h_flat = torch.cat([h_flat_pool, h_flat_vin], dim=-1)
        elif self.transe_vin_type in ['slice_pool', 'vin_slice_pool']:
            return self.tail(h_flat_pool), self.tail(h_flat_vin)
        
        h_out = self.tail(h_flat)
        return h_out


class TransitionMLP(torch.nn.Module):
    """transition function."""

    def __init__(self, input_dim, hidden_dim, action_dim=4, act_fn=nn.ReLU()):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        self.transition_mlp = nn.Sequential(
            #nn.Sigmoid(),
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


class RewardPredictor(torch.nn.Module):
    def __init__(self, encoder, input_dim=10, action_dim=8, hidden_dim=128, new_sample=False):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.encoder = encoder

        self.reward_mlp = nn.Sequential(
            nn.Linear(input_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, 3))

        self.weights = torch.tensor([-1., -0.01, 1.]).to('cuda')

        if new_sample:
            self.get_obs_action = self.get_obs_action_new
        else:
            self.get_obs_action = self.get_obs_action_old

    def get_obs_action_new(self, data_batch):
        grid, start, action, end, fake, reward = data_batch
        obs = torch.cat((grid, start, reward), axis=1)
        return obs, action

    def get_obs_action_old(self, data_batch):
        obs, action, next_obs = data_batch
        return obs, action

    def _action_to_one_hot(self, action):
        """Get one-hot encoding of index tensors."""
        zeros = torch.zeros(
            action.size()[0], self.action_dim, dtype=torch.float32,
            device=action.device)
        return zeros.scatter_(1, action.unsqueeze(1), 1)

    def reward_loss(self, data_batch, reward):
        predicted_reward = self.forward(data_batch)
        weight = (reward == 1.) * 20 + (reward == 0.01) * 2 + (reward == -1.) * 1.
        return (((reward - predicted_reward) ** 2) * weight).mean()

    def forward(self, data_batch):
        obs, action = self.get_obs_action(data_batch)
        enc_obs = self.encoder(obs)
        action_vector = self._action_to_one_hot(action)
        action_vector = action_vector.view(-1, self.action_dim)

        # Attach action to state
        state_action = torch.cat([enc_obs, action_vector], dim=-1)

        return (self.reward_mlp(state_action) * self.weights).sum(1)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
