from ml.xavier_init import init_weights_xavier
import torch


class BatchedLinear(torch.nn.Module):
    def __init__(self, num_critics: int, in_features: int, out_features: int):
        super().__init__()

        self.weight = torch.nn.Parameter(
            torch.empty(num_critics, out_features, in_features)
        )
        self.bias = torch.nn.Parameter(torch.empty(num_critics, out_features))
        
        for w in self.weight:
            torch.nn.init.xavier_uniform_(w)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        return x @ self.weight.transpose(-2, -1) + self.bias.unsqueeze(1)


class BatchedLayerNorm(torch.nn.Module):
    def __init__(self, num_critics: int, size: int):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(num_critics, size))
        self.bias = torch.nn.Parameter(torch.zeros(num_critics, size))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)


class BatchedCritic(torch.nn.Module):
    def __init__(
        self,
        num_critics: int,
        hidden_layer_size: int,
        num_hidden_layers: int,
        num_observations: int,
        num_actions: int,
        dropout_probability: float,
    ):
        super().__init__()

        self.num_critics = num_critics

        self.input_layer = BatchedLinear(
            num_critics=num_critics,
            in_features=num_observations + num_actions,
            out_features=hidden_layer_size,
        )
        self.input_norm = BatchedLayerNorm(
            num_critics=num_critics, size=hidden_layer_size
        )

        self.hidden_linears = torch.nn.ModuleList()
        self.hidden_norms = torch.nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_linears.append(
                BatchedLinear(
                    num_critics=num_critics,
                    in_features=hidden_layer_size,
                    out_features=hidden_layer_size,
                )
            )
            self.hidden_norms.append(
                BatchedLayerNorm(num_critics=num_critics, size=hidden_layer_size)
            )

        self.output_layer = BatchedLinear(
            num_critics=num_critics, in_features=hidden_layer_size, out_features=1
        )

        self.dropout = torch.nn.Dropout(dropout_probability)
        self.gelu = torch.nn.GELU()
        
        self.apply(init_weights_xavier)

    def forward(self, state: torch.Tensor, action: torch.Tensor, dropout: bool):
        X = torch.cat([state, action], dim=-1)
        X = X.unsqueeze(0).expand(self.num_critics, -1, -1)
        
        X = self.gelu(self.apply_dropout(X=self.input_norm(self.input_layer(X)), dropout=dropout))

        for linear_layer, layer_norm in zip(self.hidden_linears, self.hidden_norms):
            X = self.gelu(self.apply_dropout(X=layer_norm(linear_layer(X)), dropout=dropout))
            
        return self.output_layer(X).squeeze(-1)

    def apply_dropout(self, X: torch.Tensor, dropout: bool):
        if dropout:
            X = self.dropout(X)

        return X