from torch import nn

# TODO: rewrite! It's an old one from old_xlvin/TransE/cartpole_main.py

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