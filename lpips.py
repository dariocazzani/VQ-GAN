import os
import torch
import torch.nn as nn
from torchvision.models import vgg16
from collections import namedtuple
import requests
from tqdm import tqdm


URL_MAP = {
    "vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"
}

CKPT_MAP = {
    "vgg_lpips": "vgg.pth"
}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def get_ckpt_path(name, root):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path):
        print(f"Downloading {name} model from {URL_MAP[name]} to {path}")
        download(URL_MAP[name], path)
    return path


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer("shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = vgg16(pretrained=True).features
        slices = [vgg_pretrained_features[i] for i in range(30)]
        self.slice1 = nn.Sequential(*slices[0:4])
        self.slice2 = nn.Sequential(*slices[4:9])
        self.slice3 = nn.Sequential(*slices[9:16])
        self.slice4 = nn.Sequential(*slices[16:23])
        self.slice5 = nn.Sequential(*slices[23:30])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, verbose=False):
        h = self.slice1(x)
        h_relu1 = h
        if verbose: print(f"Layer Slice1, Output shape: {h.shape}")
        h = self.slice2(h)
        h_relu2 = h
        if verbose: print(f"Layer Slice2, Output shape: {h.shape}")
        h = self.slice3(h)
        h_relu3 = h
        if verbose: print(f"Layer Slice3, Output shape: {h.shape}")
        h = self.slice4(h)
        h_relu4 = h
        if verbose: print(f"Layer Slice4, Output shape: {h.shape}")
        h = self.slice5(h)
        h_relu5 = h
        if verbose: print(f"Layer Slice5, Output shape: {h.shape}")

        vgg_outputs = namedtuple("VGGOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        return vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


def norm_tensor(x):
    """
    Normalize images by their length to make them unit vector?
    :param x: batch of images
    :return: normalized batch of images
    """
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + 1e-10)


def spatial_average(x):
    """
     imgs have: batch_size x channels x width x height --> average over width and height channel
    :param x: batch of images
    :return: averaged images along width and height
    """
    return x.mean([2, 3], keepdim=True)


class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.scaling_layer = ScalingLayer()
        self.channels = [64, 128, 256, 512, 512]
        self.vgg = VGG16()
        self.lins = nn.ModuleList([
            NetLinLayer(self.channels[0]),
            NetLinLayer(self.channels[1]),
            NetLinLayer(self.channels[2]),
            NetLinLayer(self.channels[3]),
            NetLinLayer(self.channels[4])
        ])

        self.load_from_pretrained()

        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "vgg_lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)

    def forward(self, real_x, fake_x):
        features_real = self.vgg(self.scaling_layer(real_x))
        features_fake = self.vgg(self.scaling_layer(fake_x))
        diffs = {}

        for i in range(len(self.channels)):
            diffs[i] = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2

        return sum([spatial_average(self.lins[i].model(diffs[i])) for i in range(len(self.channels))])

    def forward_verbose(self, real_x, fake_x):
        real_x = self.scaling_layer(real_x)
        fake_x = self.scaling_layer(fake_x)
        print(f"After scaling layer, Real_x shape: {real_x.shape}, Fake_x shape: {fake_x.shape}")

        # Pass through VGG16 with verbose=True to print intermediate shapes
        features_real = self.vgg(real_x, verbose=True)
        features_fake = self.vgg(fake_x, verbose=True)

        diffs = {}
        for i in range(len(self.channels)):
            diffs[i] = (norm_tensor(features_real[i]) - norm_tensor(features_fake[i])) ** 2
            print(f"After computing difference for channel {i}, diffs[{i}] shape: {diffs[i].shape}")

        spatial_avg_diffs = []
        for i in range(len(self.channels)):
            spatial_avg_diff = spatial_average(self.lins[i].model(diffs[i]))
            spatial_avg_diffs.append(spatial_avg_diff)
            print(f"After linear layer and spatial average for channel {i}, spatial_avg_diff shape: {spatial_avg_diff.shape}")

        print(f"Final sum: {sum(spatial_avg_diffs)}")
        return sum(spatial_avg_diffs)


def main():
    img1 = torch.randn(16, 3, 256, 256)
    img2 = torch.randn(16, 3, 256, 256)
    class args:
        image_channels = 3
    lpsis = LPIPS()
    print("Number of parameters for discriminator: {:,}".format(sum(p.numel() for p in lpsis.parameters())))
    # output_shape[16, 1, 30, 30]
    lpsis.forward_verbose(img1, img2)


if __name__ == "__main__":
    main()
