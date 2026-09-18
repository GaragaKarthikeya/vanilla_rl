"""F3: the paper's features extractor with the graph encoder removed.

Identical to src/training/gnn_extractor.py except that the two GCNConv layers,
the mean-pooled netlist embedding and the current-node embedding are gone. The
fused vector is

    [cnn_out (64), valid_wh (2)]   -> 66 dims

instead of the paper's

    [cnn_out (64), graph_embed (64), current_embed (64), valid_wh (2)] -> 194

(gnn_extractor.py:102). The fuse layer's input width is reduced accordingly and
NOT zero-padded, so the network genuinely has no graph pathway rather than a
masked-out one.

The CNN, the fuse layer's output width (features_dim=128) and the activation are
byte-for-byte the paper's (gnn_extractor.py:44-59).

src/ is not modified; f3_train.py binds this class in at runtime.
"""
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class NoGCNFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        gnn_hidden_dim: int = 64,   # accepted and ignored, so policy_kwargs stay identical
        cnn_hidden_dim: int = 64,
        features_dim: int = 128,
    ) -> None:
        super().__init__(observation_space, features_dim=features_dim)

        grid_w, grid_h, grid_c = observation_space["grid"].shape

        # identical to gnn_extractor.py:44-53
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_c, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, cnn_hidden_dim),
            nn.ReLU(),
        )

        combined_dim = cnn_hidden_dim + 2          # no graph dims, no padding
        self.final = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict):
        import torch
        grid = observations["grid"]
        valid_wh = observations["valid_wh"]
        cnn_out = self.cnn(grid.permute(0, 3, 1, 2))
        return self.final(torch.cat([cnn_out, valid_wh], dim=-1))
