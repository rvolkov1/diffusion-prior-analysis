import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions.studentT import StudentT
from torch.distributions.gamma import Gamma
import numpy as np

from models import UNet2

# ================================
#         CIFAR10 Dataset
# ================================
class ToMinusOneToOne:
    def __call__(self, x):
        return x * 2. - 1.
    
transform = transforms.Compose([
    transforms.ToTensor(),
    ToMinusOneToOne()  # Normalize to [-1, 1]
])
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
num_classes = len(train_data.classes)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True,num_workers=4)

# ================================
#       Linear Beta Schedule
# ================================
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# ================================
#      Helper Function
# ================================
def extract(a, t, x_shape):
    return a.gather(-1, t.to(torch.int64)).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

def student_t_nll(x, mu, nu=4.0):
    return torch.log(1 + ((x - mu) ** 2) / nu).mean()

# ================================
#      Visualization Functions
# ================================
def visualize_denoising(ddpm, label, steps_to_show=[0, 25, 50, 75, 99]):
    x = torch.randn((10, 3, 32, 32)).to(ddpm.device)
    label = torch.tensor([label], device=ddpm.device)
    # label = torch.arange(0,10,device=ddpm.device,dtype=torch.long)
    images = []

    for t in reversed(range(ddpm.timesteps)):
        t_batch = torch.tensor([t]).to(ddpm.device)
        x = ddpm.p_sample(x, label, t_batch)
        if t in steps_to_show:
            img = torch.clamp((x[0] + 1) / 2, 0, 1).cpu()
            images.append(img)

    grid = torch.stack(images)
    grid = torchvision.utils.make_grid(grid, nrow=len(images))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Denoising Progression")
    plt.axis("off")
    plt.show()

def visualize_one_sample_per_class(ddpm):
    ddpm.model.eval()
    with torch.no_grad():
        class_labels = torch.arange(10).to(ddpm.device)
        samples = ddpm.sample(class_labels, (10, 3, 32, 32)).cpu()
        samples = (samples + 1) / 2  # De-normalize to [0, 1]

        class_names = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i, ax in enumerate(axes):
            img = samples[i]
            ax.imshow(img.permute(1, 2, 0))
            ax.axis("off")
            ax.set_title(class_names[i], fontsize=8)
        plt.tight_layout()
        plt.show()

# Call the function

# ================================
#      Helper Function
# ================================
def extract(a, t, x_shape):
    return a.gather(-1, t.to(torch.int64)).reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

def sample_gamma_noise(shape, alpha=2.0, beta=1.0, device='cpu'):
    gamma = Gamma(torch.full(shape, alpha).to(device), torch.full(shape, beta).to(device))
    noise = gamma.sample()
    noise = noise - noise.mean(dim=(1,2,3), keepdim=True)
    return noise

def sample_frechet_noise(shape, alpha=2.0, scale=1.0, loc=0.0, device='cpu'):
    # Uniform samples
    u = torch.rand(shape, device=device)
    noise = loc + scale * (-torch.log(u)) ** (-1.0 / alpha)
    noise = noise - noise.mean(dim=(1,2,3), keepdim=True)
    return noise

def gamma_nll(x, alpha=2.0, beta=1.0):
    return -Gamma(alpha, beta).log_prob(x).mean()

def frechet_nll(x, alpha=2.0, scale=1.0, loc=0.0):
    eps = 1e-6
    z = (x - loc) / scale
    z = torch.clamp(z, min=eps)
    log_prob = torch.log(alpha / scale) - (1 + alpha) * torch.log(z) - z.pow(-alpha)
    return -log_prob.mean()

class GammaDDPM:
    def __init__(self, model, betas, gamma_alpha=2.0, gamma_beta=1.0):
        self.model = model
        self.gamma_alpha = gamma_alpha
        self.gamma_beta = gamma_beta
        self.device = next(model.parameters()).device

        self.timesteps = len(betas)
        self.betas = betas.to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        alpha_bar = extract(self.alpha_bars, t, x_start.shape)
        noise = sample_gamma_noise(x_start.shape, device=self.device)
        x_noisy = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise  # Return both!

    def p_losses(self, x_start, label, t):
        x_noisy, noise = self.q_sample(x_start, t)
        predicted = self.model(x_noisy, label, t)
        # noise = torch.clamp(noise, min=1e-5)
        # predicted = torch.clamp(predicted, min=1e-5)
        loss = nn.functional.smooth_l1_loss(predicted,noise)
        return loss
        # return gamma_nll(torch.clamp(noise - predicted,min=1e-5), self.gamma_alpha,self.gamma_beta)


    def p_sample(self, x, label, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        predicted_noise = self.model(x, label, t)

        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
 
        if t[0] == 0:
            return mean
        noise = sample_gamma_noise(x.shape, alpha=2.0, beta=1.0, device=self.device)

        return torch.sqrt(alphas_t) * mean + torch.sqrt(betas_t) * noise

    def sample(self, label, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * shape[0]).to(self.device)
            x = self.p_sample(x, label, t_batch)
        return x.clamp(-1, 1)
    
class FrechetDDPM:
    def __init__(self, model, betas, f_alpha=2.0, scale=1.0):
        self.model = model
        self.f_alpha = f_alpha
        self.scale = scale
        self.device = next(model.parameters()).device

        self.timesteps = len(betas)
        self.betas = betas.to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        alpha_bar = extract(self.alpha_bars, t, x_start.shape)
        noise = sample_frechet_noise(x_start.shape, alpha=self.f_alpha, scale=self.scale, device=self.device)
        x_noisy = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise  # Return both!

    def p_losses(self, x_start, label, t):
        x_noisy, noise = self.q_sample(x_start, t)
        predicted = self.model(x_noisy, label, t)
        # noise = torch.clamp(noise, min=1e-5)
        # predicted = torch.clamp(predicted, min=1e-5)
        loss = nn.functional.smooth_l1_loss(predicted,noise)
        return loss
        # return gamma_nll(torch.clamp(noise - predicted,min=1e-5), self.gamma_alpha,self.gamma_beta)


    def p_sample(self, x, label, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        predicted_noise = self.model(x, label, t)

        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
 
        if t[0] == 0:
            return mean
        noise = sample_frechet_noise(x.shape, alpha=self.f_alpha, scale=self.scale, device=self.device)

        return torch.sqrt(alphas_t) * mean + torch.sqrt(betas_t) * noise

    def sample(self, label, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * shape[0]).to(self.device)
            x = self.p_sample(x, label, t_batch)
        return x.clamp(-1, 1)

# ================================
#     Student-t Based DDPM
# ================================
class StudentTDDPM:
    def __init__(self, model, betas, nu=4.0):
        self.model = model
        self.nu = nu
        self.device = next(model.parameters()).device

        self.timesteps = len(betas)
        self.betas = betas.to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        alpha_bar = extract(self.alpha_bars, t, x_start.shape)
        noise = StudentT(df=self.nu, loc=0, scale=1).sample(x_start.shape).to(self.device)
        x_noisy = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise  # Return both!

    def p_losses(self, x_start, label, t):
        x_noisy, noise = self.q_sample(x_start, t)
        predicted = self.model(x_noisy, label, t)
        return student_t_nll(noise, predicted, self.nu)


    def p_sample(self, x, label, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        predicted_noise = self.model(x, label, t)

        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        if t[0] == 0:
            return mean
        noise = StudentT(df=self.nu, loc=0, scale=1).sample(x.shape).to(self.device)

        return torch.sqrt(alphas_t) * mean + torch.sqrt(betas_t) * noise

    def sample(self, label, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * shape[0]).to(self.device)
            x = self.p_sample(x, label, t_batch)
        return x.clamp(-1, 1)

# ================================
#     Gaussian Based DDPM
# ================================
class DDPM:
    def __init__(self, model, betas):
        self.model = model
        self.device = next(model.parameters()).device

        self.timesteps = len(betas)
        self.betas = betas.to(self.device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t):
        alpha_bar = extract(self.alpha_bars, t, x_start.shape)
        noise = torch.randn_like(x_start).to(self.device)
        x_noisy = torch.sqrt(alpha_bar) * x_start + torch.sqrt(1 - alpha_bar) * noise
        return x_noisy, noise  # Return both!

    def p_losses(self, x_start, label, t):
        x_noisy, noise = self.q_sample(x_start, t)
        encoder_hidden_states = torch.zeros((x_start.shape[0], 1, 1280), device=self.device)
        predicted = self.model(x_noisy, timestep=t, class_labels=label,encoder_hidden_states=encoder_hidden_states)  # <- fixed
        predicted = predicted.sample
        loss = nn.functional.mse_loss(predicted, noise)
        return loss


    def p_sample(self, x, label, t):
        betas_t = extract(self.betas, t, x.shape)
        alphas_t = extract(self.alphas,t,x.shape)
        alpha_bar_t = extract(self.alpha_bars, t, x.shape)
        encoder_hidden_states = torch.zeros((x.shape[0], 1, 1280), device=self.device)
        predicted_noise = self.model(x, timestep=t, class_labels=label,encoder_hidden_states=encoder_hidden_states)
        predicted_noise = predicted_noise.sample
        mean = (1 / torch.sqrt(alphas_t)) * (
                x - ((1 - alphas_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)

        if t[0] == 0:
            return mean
        noise = torch.randn_like(x).to(self.device)
        return torch.sqrt(alphas_t) * mean + torch.sqrt(betas_t) * noise

    def sample(self, label, shape):
        x = torch.randn(shape).to(self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t] * shape[0]).to(self.device)
            x = self.p_sample(x, label, t_batch)
        return x.clamp(-1, 1)



# ================================
#        Training Loop
# ================================
def train_ddpm(model, ddpm, dataloader, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    for epoch in tqdm(range(epochs)):
        for batch in dataloader:
            x = batch[0].to(ddpm.device)
            label = batch[1].to(ddpm.device)
            t = torch.randint(0, ddpm.timesteps, (x.size(0),), device=ddpm.device)
            loss = ddpm.p_losses(x, label, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch%10==0:
            model_save_file = f"model_saves/gammaddpm__conditional_epoch{epoch}.pth"
            torch.save(model.state_dict(),model_save_file)
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timesteps = 400
    betas = linear_beta_schedule(timesteps)
    model = UNet2(base_channels=128).to(device)
    ddpm = FrechetDDPM(model, betas, f_alpha=2.0, scale=1.0)



    train_ddpm(model, ddpm, train_loader, epochs=500)
# ================================
#         Run Training
# ================================
if __name__=="__main__":
    main()