import torch
import torch.nn as nn

class Codebook(nn.Module):
    def __init__(self, args):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta:float = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)


    def forward(self, z:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = z.permute(0, 2, 3, 1).contiguous() # [b, h, w, c]
        z_flattened = z.view(-1, self.latent_dim)

        distance = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
                   torch.sum(self.embedding.weight**2, dim=1) - \
                   2 * (torch.matmul(z_flattened, self.embedding.weight.t()))

        min_encoding_indices = torch.argmin(distance, dim=1)
        z_q:torch.Tensor = self.embedding(min_encoding_indices).view(z.shape)

        loss:torch.Tensor = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        # Copy gradients from quantized z_q to prior z (endoder output)
        z_q = z + (z_q - z).detach()

        z_q = z_q.permute(0, 3, 1, 2) # [b, c, h, w]
        return z_q, min_encoding_indices, loss
