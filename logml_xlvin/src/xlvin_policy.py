import torch
import torch.nn.functional as F
from torch import nn

from logml_xlvin.src.create_graph import create_graph_v2
from logml_xlvin.src.executor import SparseMPNN
from old_xlvin.rl.ppo import FixedCategorical

# TODO: rewrite! It's an old one (refactored a little bit)

class XLVINModel(nn.Module):

    def __init__(
            self,
            action_space,
            encoder,
            transition,
            transition_hidden_dim,
            edge_features_dim,
            executor,
            num_processes,
            cat_enc_gnn,
            full_cat_gnn,
            freeze_encoder,
            gamma,
            transition2gnn_n_layers=0,
            gnn_decoder_n_layers=0,
            gnn_steps=2,
            vin_attention=False,
            graph_detach=False,
            transe_vin_type=None
    ):
        super(XLVINModel, self).__init__()
        self.transition = transition
        self.encoder = encoder
        self.graph_detach = graph_detach
        self.transe_vin_type = transe_vin_type
        if freeze_encoder:
            print("Freeze encoder")
            for param in transition.parameters():
                param.requires_grad = False
        fc_input_dim = transition_hidden_dim  # why?

        fc_input_dim = self._add_executor(
            transition_hidden_dim, transition2gnn_n_layers, edge_features_dim, executor,
            gnn_decoder_n_layers, gnn_steps, cat_enc_gnn, full_cat_gnn
        )

        self.num_processes = num_processes

        self.gamma = gamma
        # TODO support non-discrete action worlds
        if action_space.__class__.__name__ == 'Discrete':
            self.num_actions = action_space.n
            self.vin_attention = vin_attention
            if self.vin_attention:
                fc_input_dim += self.num_actions * executor.hidden_dim
            self.actor_linear = nn.Linear(fc_input_dim, self.num_actions)
        else:
            raise NotImplementedError
        self.critic_linear = nn.Linear(fc_input_dim, 1)

    def _add_executor(
            self, transition_hidden_dim, transition2gnn_n_layers, edge_features_dim, executor: SparseMPNN,
            gnn_decoder_n_layers, gnn_steps, cat_enc_gnn, full_cat_gnn
    ):
        fc_input_dim = executor.hidden_dim
        self._add_transition2gnn_fc_layers(executor.hidden_dim, transition2gnn_n_layers, transition_hidden_dim)
        self.gnn_steps = gnn_steps
        self.edge_proj = nn.Linear(edge_features_dim, executor.hidden_dim)
        self.executor = executor
        self._add_executor_decoder_fc_layers(executor.hidden_dim, gnn_decoder_n_layers)

        print("Freeze GNN")
        for param in executor.parameters():
            param.requires_grad = False

        self.cat_enc_gnn = cat_enc_gnn # what is it?
        self.full_cat_gnn = full_cat_gnn # what is it?
        if full_cat_gnn:
            fc_input_dim = transition_hidden_dim + 2 * executor.hidden_dim
        elif cat_enc_gnn:
            fc_input_dim = transition_hidden_dim + executor.hidden_dim

        return fc_input_dim

    def _add_transition2gnn_fc_layers(self, executor_hidden_dim, transition2gnn_n_layers, transition_hidden_dim):
        self.transition2gnn_n_fc_layers = transition2gnn_n_layers
        if self.transition2gnn_n_fc_layers > 0:
            layers = []
            layers += [nn.Linear(transition_hidden_dim, executor_hidden_dim)]
            for i in range(self.transition2gnn_n_fc_layers - 1):
                layers += [nn.ReLU(), nn.Linear(executor_hidden_dim, executor_hidden_dim)]
            self.transe2gnn_fc_layers = nn.Sequential(*layers)

    def _add_executor_decoder_fc_layers(self, executor_hidden_dim, executor_decoder_n_fc_layers):
        self.gnn_decoder_n_fc_layers = executor_decoder_n_fc_layers
        if self.gnn_decoder_n_fc_layers > 0:
            layers = []
            for i in range(self.gnn_decoder_n_fc_layers - 1):
                layers += [nn.Linear(executor_hidden_dim, executor_hidden_dim), nn.ReLU()]
            layers += [nn.Linear(executor_hidden_dim, executor_hidden_dim)]
            self.executor_decoder_fc = nn.Sequential(*layers)

    # TODO support recurrent updates (with GRU hidden state)
    def act(self, observations, deterministic=False):
        if self.transe_vin_type in ["slice_pool", "vin_slice_pool"]:
            latents, latents_slice = self.encoder(observations)
        else:
            latents = self.encoder(observations)
        if self.include_gnn:
            num_states = latents.shape[0]
            assert num_states == self.num_processes
            node_features, senders, receivers, edge_features = create_graph_v2(latents, self.transition,
                                                                               self.gamma, self.num_actions,
                                                                               self.gnn_steps, self.graph_detach)
            if self.transition2gnn_n_fc_layers > 0:
                node_features = self.transe2gnn_fc_layers(node_features)
            embedded_edge_features = self.edge_proj(edge_features)
            all_latents = self.gnnx(node_features, senders, receivers, embedded_edge_features)
            if self.full_cat_gnn:
                cur_latents = all_latents
                if self.gnn_decoder_n_fc_layers > 0:
                    cur_latents = self.executor_decoder_fc(cur_latents)
                gnn_latents_sum = F.normalize(cur_latents[:self.num_processes], p=2,
                                              dim=-1)  # torch.cat((latents, cur_latents[:self.num_processes]), dim=-1)
                gnn_latents_max = nn.Identity()(gnn_latents_sum)
            for i in range(self.gnn_steps - 1):
                all_latents = all_latents + node_features
                all_latents = self.gnnx(all_latents, senders, receivers, embedded_edge_features)
                if self.full_cat_gnn:
                    cur_latents = all_latents
                    if self.gnn_decoder_n_fc_layers > 0:
                        cur_latents = self.executor_decoder_fc(cur_latents)
                    gnn_latents_sum += F.normalize(cur_latents[:self.num_processes], p=2,
                                                   dim=-1)  # torch.cat((latents, cur_latents[:self.num_processes]), dim=-1)
                    gnn_latents_max = torch.max(gnn_latents_max.clone(), F.normalize(cur_latents[:self.num_processes]))
            if not self.full_cat_gnn and self.gnn_decoder_n_fc_layers > 0:
                all_latents = self.executor_decoder_fc(all_latents)
            if not self.full_cat_gnn and self.cat_enc_gnn:
                # this is the default
                if self.transe_vin_type in ["slice_pool", "vin_slice_pool"]:
                    latents = torch.cat((latents_slice, all_latents[:self.num_processes]), dim=-1)
                else:
                    latents = torch.cat((latents, all_latents[:self.num_processes]), dim=-1)
            elif not self.full_cat_gnn:
                latents = all_latents[:self.num_processes]
            elif self.full_cat_gnn:
                latents = torch.cat((latents, gnn_latents_sum, gnn_latents_max), dim=-1)
            if self.vin_attention:
                latents = torch.cat((latents, all_latents[self.num_processes:
                                                          self.num_processes + self.num_processes * self.num_actions]
                                     .reshape(self.num_processes, -1)), dim=-1)
        policy = self.actor_linear(latents)
        value = self.critic_linear(latents)
        actor = FixedCategorical(logits=policy)

        if deterministic:
            action = actor.mode()
        else:
            action = actor.sample()
        log_probs = actor.log_probs(action)
        entropy = actor.entropy().mean()
        return value, action, log_probs
