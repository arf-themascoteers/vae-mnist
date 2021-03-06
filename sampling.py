import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from vae import VAELinear


def sampling():
    vae = VAELinear()
    vae.load_state_dict(torch.load("models/vae.h5"))
    vae.eval()

    with torch.no_grad():
        for i in range(10):
            noise = torch.randn(1,2)
            generated_image = vae.decoder(noise)
            generated_image = generated_image[0].reshape(28, 28)
            plt.imshow(generated_image)
            plt.show()

if __name__ == "__main__":
    sampling()