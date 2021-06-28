import torch.nn as nn
import torch
from decoder import Decoder
from encoder import Encoder


class VAELinear(nn.Module):
    def __init__(self):
        super(VAELinear, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterization(mean, log_var)
        decoded = self.decoder(z)
        return decoded, mean, log_var

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var*epsilon
        return z