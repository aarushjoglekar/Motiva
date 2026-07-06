from ml.xavier_init import init_weights_xavier
import torch


class Actor(torch.nn.Module):
    def __init__(
        self,
        hidden_layer_size: int,
        num_hidden_layers: int,
        num_observations: int,
        num_actions: int,
    ):
        super().__init__()

        layers = [torch.nn.Linear(num_observations, hidden_layer_size), torch.nn.GELU()]

        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_layer_size, hidden_layer_size))
            layers.append(torch.nn.GELU())

        layers.append(torch.nn.Linear(hidden_layer_size, num_actions * 2))

        self.layers = torch.nn.Sequential(*layers)

        self.apply(init_weights_xavier)

    def forward(self, state: torch.Tensor):
        return self.layers(state)
