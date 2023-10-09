import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [
                    nn.Conv2d(args.image_channels, num_filters_last, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2)
                  ]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult,
                          kernel_size=4, stride=2 if i < n_layers else 1, padding=1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, inplace=False)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def forward_verbose(self, x:torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.model):
            x = layer(x)
            print(f"Layer {i+1}: {layer.__class__.__name__}, Output shape: {x.shape}")
        return x

def main():
    img = torch.randn(16, 3, 256, 256)
    class args:
        image_channels = 3
    discriminator = Discriminator(args)
    print("Number of parameters for discriminator: {:,}".format(sum(p.numel() for p in discriminator.parameters())))
    # output_shape[16, 1, 30, 30]
    discriminator.forward_verbose(img)


if __name__ == "__main__":
    main()
