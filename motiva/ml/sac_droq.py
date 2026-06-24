import torch

class Actor(torch.nn.Module):
    def __init__(self, hidden_layer_size, num_hidden_layers, num_observations, num_actions):
        super().__init__()

        layers = [
            torch.nn.Linear(num_observations, hidden_layer_size),
            torch.nn.ReLU()
        ]

        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(hidden_layer_size, num_actions * 2))

        self.layers = torch.nn.Sequential(*layers)

    def forward(self, state):
        return self.layers(state)
    
class Critic(torch.nn.Module):
    def __init__(self, hidden_layer_size, num_hidden_layers, num_observations, num_actions, dropout_probability):
        super().__init__()

        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(num_observations + num_actions, hidden_layer_size),
            torch.nn.LayerNorm(hidden_layer_size),
            torch.nn.ReLU()
        )

        self.hidden_layers = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(torch.nn.Sequential(
                torch.nn.Linear(hidden_layer_size, hidden_layer_size),
                torch.nn.LayerNorm(hidden_layer_size),
                torch.nn.ReLU()
            ))

        self.output_layer = torch.nn.Linear(hidden_layer_size, 1)

        self.dropout = torch.nn.Dropout(p=dropout_probability)

    def forward(self, state, action, dropout=True):
        X = torch.cat([state, action], dim=-1)

        X = self.input_layer(X)
        X = self.apply_dropout(X, dropout)

        for layer in self.hidden_layers:
            X = layer(X)
            X = self.apply_dropout(X, dropout)

        return self.output_layer(X).squeeze(-1)

    def apply_dropout(self, X, dropout):
        if dropout:
            X = self.dropout(X)

        return X