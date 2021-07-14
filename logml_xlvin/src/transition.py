import numpy as np
import torch
from torch import nn


class NegativeSampler:
    """
    Sample negative state across episodes at random
    """

    @staticmethod
    def get_random_negative_state(state: torch.Tensor, next_state: torch.Tensor):
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)

        neg_state = next_state.clone().detach()
        neg_state[:, 1] = neg_state[:, 1][perm]

        return neg_state


class ActionEncoder:
    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim

    def to_one_hot(self, action):
        zeros = torch.zeros(action.size()[0], self.embedding_dim, dtype=torch.float32, device=action.device)
        return zeros.scatter_(1, action.unsqueeze(1), 1)


class TransitionMLP(torch.nn.Module):
    """Multi-layered perceptron as a transition function"""

    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int, activation_function=nn.ReLU()
    ):
        super().__init__()

        self.action_dim = action_dim
        self.input_dim = state_dim + action_dim
        self.hidden_dim = hidden_dim

        self.transition_mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            activation_function,
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            activation_function,
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, encoded_state: torch.Tensor, encoded_action: torch.Tensor):
        # state: [batch_size, embeddin_dim]
        # action: [batch_size, embeddin_dim?]

        encoded_action = encoded_action.view(-1, self.action_dim)

        # Attach action to state
        state_action = torch.cat([encoded_state, encoded_action], dim=-1)

        return self.transition_mlp(state_action)


class Transition(nn.Module):
    """
    TransE based Transition model
    """

    def __init__(self, embedding_dim, hidden_dim, action_dim, hinge=1., sigma=0.5):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.action_dim = action_dim
        self.hinge = hinge
        self.sigma = sigma

        self.pos_loss = 0
        self.neg_loss = 0

        self.transition_model = TransitionMLP(
            state_dim=embedding_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim
        )

        self.contrastive_loss = self.contrastive_loss

    def energy(self, state, action, predicted_state_translation, next_state):
        predicted_translated_state = state + predicted_state_translation

        return self._state_dissimilarity_measure(predicted_translated_state, next_state)

    def _state_dissimilarity_measure(self, state_1, state_2):
        """Dissimilarity function based on normalized squared L2 norm."""
        norm = 0.5 / (self.sigma ** 2)

        diff = state_1 - state_2

        return norm * diff.pow(2).sum(1)  # no sqrt?

    def contrastive_loss(
            self, encoded_state, action, predicted_state_translation, encoded_next_state, encoded_neg_state
    ):
        self.pos_loss = self.energy(encoded_state, action, predicted_state_translation, encoded_next_state).mean()
        self.neg_loss = torch.max(
            torch.zeros_like(self.pos_loss),
            self.hinge - self.energy(encoded_state, action, predicted_state_translation, encoded_neg_state)
        ).mean()

        return self.pos_loss + self.neg_loss

    def forward(self, state, action):
        return self.transition_model(state, action)
