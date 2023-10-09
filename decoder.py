import torch.nn as nn
import torch
from helper import ResidualBlock
from helper import NonLocalBlock
from helper import UpSampleBlock
from helper import GroupNorm
from helper import Swish


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels)
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def forward_verbose(self, x:torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.model):
            x = layer(x)
            print(f"Layer {i+1}: {layer.__class__.__name__}, Output shape: {x.shape}")
        return x


class SimpleDecoder(nn.Module):
    def __init__(self, args):
        super(SimpleDecoder, self).__init__()
        channels = [256, 128, 64, 32, 32]  # Reduced and matched channel sizes
        attn_resolutions = [16]
        num_res_blocks = 1
        resolution = 16

        in_channels = channels[0]
        layers = [
            nn.Conv2d(args.latent_dim, in_channels, kernel_size=3, stride=1, padding=1),
            ResidualBlock(in_channels, in_channels)
        ]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, args.image_channels, kernel_size=3, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def forward_verbose(self, x:torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.model):
            x = layer(x)
            print(f"Layer {i+1}: {layer.__class__.__name__}, Output shape: {x.shape}")
        return x



def main():
    class args:
        image_channels = 3
        latent_dim = 64
    decoder = Decoder(args)
    z = torch.randn(16, args.latent_dim, 16, 16)
    print("Number of parameters for Decoder: {:,}".format(sum(p.numel() for p in decoder.parameters())))
    decoder.forward_verbose(z)

    simple_decoder = SimpleDecoder(args)
    z = torch.randn(16, args.latent_dim, 16, 16)
    print("Number of parameters for Decoder: {:,}".format(sum(p.numel() for p in simple_decoder.parameters())))
    simple_decoder.forward_verbose(z)

if __name__ == "__main__":
    main()