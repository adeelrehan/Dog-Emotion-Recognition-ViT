import torch
import torch.nn as nn


class CNNPatchEmbedding(nn.Module):
    """
    CNN-based patch embedding.
    Extracts local visual features before Transformer encoding.
    """

    def __init__(self, in_channels=3, embed_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)              # (B, C, H, W)
        x = x.flatten(2)              # (B, C, H*W)
        x = x.transpose(1, 2)         # (B, N, C)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Single Transformer Encoder block.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_output, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x)
        )
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class HybridVisionTransformer(nn.Module):
    """
    Hybrid Vision Transformer model trained FROM SCRATCH.
    Combines CNN-based patch embedding with Transformer encoder.
    """

    def __init__(
        self,
        num_classes=4,
        embed_dim=128,
        depth=4,
        num_heads=4,
    ):
        super().__init__()

        self.patch_embed = CNNPatchEmbedding(
            in_channels=3,
            embed_dim=embed_dim
        )

        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(embed_dim, num_heads)
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)        # (B, N, D)
        x = self.encoder(x)
        x = self.norm(x)

        x = x.mean(dim=1)              # Global average pooling
        x = self.classifier(x)
        return x
