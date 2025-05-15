import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim

from model import create_model
from diffusion import GaussianDiffusion, StudentTDiffusion, GammaDiffusion, FrechetDiffusion
from dataset import LongTailCIFAR100
import sampling


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = LongTailCIFAR100(root="./data",transform=transform,imb_factor=0.1)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    class_embed_size = 8

    model = create_model(class_embed_size=class_embed_size).to(device)
    #DDPM 
    diffusion = StudentTDiffusion(model,
                                  device = device,
                                  )
    diffusion.embed_class_labels.load_state_dict(torch.load("embeddings.pth"))
    optimizer = optim.AdamW(list(model.parameters()) + list(diffusion.embed_class_labels.parameters()), lr=2e-4)

    for epoch in range(20):
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            t = torch.randint(0, diffusion.timesteps, (x.size(0),), device=device).long()
            loss = diffusion.p_losses(x, t, y)

            # sampling.show_image(x[0],dataset.idx_to_class[y[0].cpu().item()])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "weights.pth")
    torch.save(diffusion.embed_class_labels.state_dict(),"embeddings.pth")

if __name__ == "__main__":
    train()
