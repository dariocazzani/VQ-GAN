import torch.nn as nn
import torch
from helper import ResidualBlock
from helper import NonLocalBlock
from helper import DownSampleBlock
from helper import GroupNorm
from helper import Swish


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        channels = [128, 128, 128, 256, 256, 512]
        attn_resolutions = [16]
        num_res_blocks = 2
        resolution = 256
        layers = [nn.Conv2d(args.image_channels, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels)) #type: ignore
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels)) #type: ignore
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(channels[i+1])) #type: ignore
                resolution //= 2

        layers.append(ResidualBlock(channels[-1], channels[-1])) #type: ignore
        layers.append(NonLocalBlock(channels[-1])) #type: ignore
        layers.append(ResidualBlock(channels[-1], channels[-1])) #type: ignore
        layers.append(GroupNorm(channels[-1])) #type: ignore
        layers.append(Swish()) #type: ignore
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, kernel_size=3, stride=1, padding=1)) #type: ignore
        self.model = nn.Sequential(*layers)


    def forward(self, x):
        return self.model(x)

    def forward_verbose(self, x:torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.model):
            x = layer(x)
            print(f"Layer {i+1}: {layer.__class__.__name__}, Output shape: {x.shape}")
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, args):
        super(SimpleEncoder, self).__init__()
        channels = [32, 32, 32, 64, 128, 256]  # Reduced channel sizes
        attn_resolutions = [16]
        num_res_blocks = 1
        resolution = 256
        layers = [nn.Conv2d(args.image_channels, channels[0], kernel_size=3, stride=1, padding=1)]

        for i in range(len(channels)-1):
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels)) #type: ignore
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels)) #type: ignore
            if i != len(channels) - 2:
                layers.append(DownSampleBlock(channels[i+1])) #type: ignore
                resolution //= 2

        layers.append(ResidualBlock(channels[-1], channels[-1])) #type: ignore
        layers.append(GroupNorm(channels[-1])) #type: ignore
        layers.append(Swish()) #type: ignore
        layers.append(nn.Conv2d(channels[-1], args.latent_dim, kernel_size=3, stride=1, padding=1)) #type: ignore
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
    encoder = Encoder(args)
    print("Number of parameters for encoder: {:,}".format(sum(p.numel() for p in encoder.parameters())))
    img = torch.randn(16, 3, 256, 256)
    encoder.forward_verbose(img)

    simple_encoder = SimpleEncoder(args)
    print("Number of parameters for simple encoder: {:,}".format(sum(p.numel() for p in simple_encoder.parameters())))
    img = torch.randn(16, 3, 256, 256)
    simple_encoder.forward_verbose(img)

if __name__ == "__main__":
    main()
