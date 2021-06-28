import torch
from torchvision import datasets, transforms
import vae
import torch.nn.functional as F


def calculate_loss(x, decoded, mean, log_var):
    reproduction_loss = F.binary_cross_entropy(decoded, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD


transform = transforms.ToTensor()
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(dataset=mnist_data,
                                          batch_size=64,
                                          shuffle=True)

model = vae.VAELinear()
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

num_epochs = 10
outputs = []
for epoch in range(num_epochs):
    overall_loss = 0
    count = len(data_loader)
    for (img, _) in data_loader:
        img = img.reshape(-1, 28*28)
        decoded_image, mean, log_var = model(img)
        loss = calculate_loss(img, decoded_image, mean, log_var)
        overall_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch:{epoch + 1}, Loss:{overall_loss/count:.4f}')

torch.save(model.state_dict(), 'models/vae.h5')