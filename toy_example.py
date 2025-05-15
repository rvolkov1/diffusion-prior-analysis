import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Denoiser(nn.Module):
    def __init__(self, input_dim=3):  # 2D data + 1 sigma
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output denoised 2D vector
        )

    def forward(self, x, sigma):
        sigma = sigma.to(x.device)
        x_sigma = torch.cat([x, sigma.expand(x.shape[0], 1)], dim=1)
        return self.net(x_sigma)

class Frechet(torch.distributions.Distribution):
    arg_constraints = {'alpha': torch.distributions.constraints.positive, 'scale': torch.distributions.constraints.positive}
    support = torch.distributions.constraints.positive

    def __init__(self, alpha, scale=1.0, validate_args=None):
        self.alpha = torch.tensor(alpha)
        self.scale = torch.tensor(scale)
        super().__init__(self.alpha.shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape)
        return self.scale * (-torch.log(u)) ** (-1 / self.alpha)
    
def generate_neals_funnel(n_samples=1_000_000):
    x1 = torch.randn(n_samples) * 3
    x2 = torch.randn(n_samples) * torch.exp(x1 / 2)
    data = torch.stack([x1, x2], dim=1)
    return data

def zscore_normalize(data):
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True)
    return (data - mean) / std, mean, std

def edm_loss(model, x_clean, sigma, weighting='edm'):
    """
    x_clean: [B, 2] clean sample
    sigma: [B, 1] noise level
    """
    noise = torch.randn_like(x_clean) * sigma
    x_noisy = x_clean + noise
    denoised = model(x_noisy, sigma)

    if weighting == 'edm':
        # From EDM paper: λ(σ) = (σ^2 + 1)^0.5 / σ^2
        w = torch.sqrt(sigma**2 + 1) / sigma**2
    elif weighting == 'constant':
        w = 1.0
    elif weighting == 've':
        # For VE-SDE models, λ(σ) = 1
        w = 1.0 / sigma**2
    else:
        raise ValueError("Unknown weighting scheme")

    loss = w * ((denoised - x_clean) ** 2 / sigma**2)
    return loss.mean()


def train_denoiser(model, dataloader, epochs=5, lr=1e-3, device='cuda', tail='gaussian', nu=(4,20),alpha=2.0, scale=1.0,gamma_shape=1.5, gamma_scale=1.0):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    nu1, nu2 = nu if isinstance(nu, (tuple, list)) else (nu, nu)

    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x_clean = batch[0].to(device)
            sigma = torch.rand((x_clean.size(0), 1), device=device) * 1.5 + 0.5  # [0.5, 2]

            if tail == 'gaussian':
                noise = torch.randn_like(x_clean) * sigma
            elif tail == 'student':
                noise_x1 = torch.distributions.StudentT(df=nu1).rsample((x_clean.size(0), 1)).to(device)
                noise_x2 = torch.distributions.StudentT(df=nu2).rsample((x_clean.size(0), 1)).to(device)
                noise = torch.cat([noise_x1, noise_x2], dim=1) * sigma
            elif tail == 'frechet':
                # Sample from Fréchet(alpha), then symmetrize around 0

                frechet = Frechet(alpha=alpha, scale=scale)
                noise_x1 = frechet.rsample((x_clean.size(0), 1)).to(device)
                noise_x2 = frechet.rsample((x_clean.size(0), 1)).to(device)

                # Random sign to symmetrize (convert to "zero-mean")
                signs = torch.randint(0, 2, noise_x1.shape, device=device) * 2 - 1
                noise_x1 *= signs
                signs = torch.randint(0, 2, noise_x2.shape, device=device) * 2 - 1
                noise_x2 *= signs

                noise = torch.cat([noise_x1, noise_x2], dim=1) * sigma
            elif tail == 'gamma':
                # Gamma distribution is positive — symmetrize to make zero-mean
                shape, rate = alpha, scale  # Gamma params (shape k=α, rate=β)
                noise_x1 = torch.distributions.Gamma(concentration=shape, rate=rate).rsample((x_clean.size(0), 1)).to(device)
                noise_x2 = torch.distributions.Gamma(concentration=shape, rate=rate).rsample((x_clean.size(0), 1)).to(device)

                signs = torch.randint(0, 2, noise_x1.shape, device=device) * 2 - 1
                noise_x1 *= signs
                signs = torch.randint(0, 2, noise_x2.shape, device=device) * 2 - 1
                noise_x2 *= signs

                noise = torch.cat([noise_x1, noise_x2], dim=1) * sigma
            else:
                raise ValueError(f"Unsupported tail type: {tail}")

            x_noisy = x_clean + noise
            x_denoised = model(x_noisy, sigma)
            loss = F.mse_loss(x_denoised, x_clean)
            # loss = edm_loss(model, x_clean, sigma, weighting='edm')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{tail.upper()}] Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")


def sample_latents(n_samples=100_000, tail='gaussian', nu=(4,20),alpha=2.0, scale=1.0,gamma_shape=1.5,gamma_scale=1.0):
    nu1, nu2 = nu if isinstance(nu, (tuple, list)) else (nu, nu)
    if tail == 'gaussian':
        return torch.randn(n_samples, 2)
    elif tail == 'student':
        x1 = torch.distributions.StudentT(df=nu1).rsample((n_samples, 1)).to(device)
        x2 = torch.distributions.StudentT(df=nu2).rsample((n_samples, 1)).to(device)
        x = torch.cat([x1, x2], dim=1)
        return x
    elif tail == 'frechet':
        frechet = Frechet(alpha=alpha, scale=scale)
        x1 = frechet.rsample((n_samples, 1)).to(device)
        x2 = frechet.rsample((n_samples, 1)).to(device)

        signs = torch.randint(0, 2, x1.shape, device=x1.device) * 2 - 1
        x1 *= signs
        signs = torch.randint(0, 2, x2.shape, device=x2.device) * 2 - 1
        x2 *= signs
        return torch.cat([x1, x2], dim=1)
    elif tail == 'gamma':
        shape, rate = gamma_shape, gamma_scale
        gamma = torch.distributions.Gamma(concentration=shape, rate=rate)
        x1 = gamma.rsample((n_samples, 1)).to(device)
        x2 = gamma.rsample((n_samples, 1)).to(device)

        signs = torch.randint(0, 2, x1.shape, device=x1.device) * 2 - 1
        x1 *= signs
        signs = torch.randint(0, 2, x2.shape, device=x2.device) * 2 - 1
        x2 *= signs

        return torch.cat([x1, x2], dim=1)
        

def visualize_samples(samples, title="Samples"):
    samples = samples.detach().cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.hexbin(samples[:, 0], samples[:, 1], gridsize=100, cmap='inferno')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("equal")
    plt.colorbar()
    plt.show()


if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generate_data = True
    train = True
    n_samples = 1_000_000
    distribution = ['gamma'] #options are gaussian, student, frechet
    num_epochs = 30
    # student parameters
    nu = (4,20)
    # frechet parameters
    alpha = 4.0
    scale = 1.0
    #gamma parameters
    gamma_shape = 1.0
    gamma_scale = 1.0

    #visualizing
    num_generated_samples = 1_000_000
    if 'gaussian' in distribution:
        sigma = torch.full((num_generated_samples, 1), 1.0).to(device)
    else:
        sigma = torch.zeros((num_generated_samples,1)).to(device)

    if generate_data:
        raw_data = generate_neals_funnel(n_samples=n_samples)
        torch.save(raw_data,"funnel.pt")
    else:
        raw_data = torch.load("funnel.pt")

    raw_data_np = raw_data.numpy()
    raw_data_mean = raw_data_np.mean(axis=1)
    data, mean, std = zscore_normalize(raw_data)
    mean = mean.numpy()
    std = std.numpy()

    plt.scatter(raw_data[:,1],raw_data[:,0],s=0.5)
    plt.tight_layout()
    plt.xlim((-500,500))
    plt.ylim((-10,10))
    plt.title("Neal's Funnel")
    plt.savefig("Neal's Funnel.png")

    # Prepare dataset
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    if 'gaussian' in distribution:
        # Train EDM denoiser
        edm_model = Denoiser().to(device)
        if train:
            train_denoiser(edm_model, dataloader, epochs=num_epochs)
            torch.save(edm_model.state_dict(),"gaussian.pt")  
        else:
            edm_model.load_state_dict(torch.load("gaussian.pt"))
        
        

        # Generate samples (EDM baseline)
        edm_latents = sample_latents(num_generated_samples, tail='gaussian').to(device)
        

        edm_samples = edm_model(edm_latents, sigma)  # crude denoising
        edm_gen = edm_samples.detach().cpu().numpy()
        edm_gen = edm_gen*std+mean


        # plt.subplot(1, 2, 1)
        plt.figure()

        plt.scatter(raw_data_np[:, 1], raw_data_np[:, 0], s=0.5, alpha=0.5)
        plt.title(f"Gaussian Generated Neal’s Funnel Samples (sigma={sigma[0].item()})")
        plt.xlabel("x1"), plt.ylabel("x2")

        # plt.subplot(1, 2, 2)
        plt.scatter(edm_gen[:, 1], edm_gen[:, 0], s=0.5, alpha=0.5, color='orange')
        # plt.title("Generated Samples from Model")
        # plt.xlabel("x1"), plt.ylabel("x2")

        plt.tight_layout()
        plt.xlim((-500,500))
        plt.ylim((-10,10))
        plt.savefig("Gaussian.png")
        # visualize_samples(edm_samples, title="EDM (Gaussian)")
    if 'student' in distribution:
        # Generate samples (t-EDM with ν=4)
        t_edm_model = Denoiser().to(device)
        if train:
            train_denoiser(t_edm_model, dataloader, epochs=num_epochs, tail="student",nu=nu)
            torch.save(t_edm_model.state_dict(),"student.pt")
        else:
            t_edm_model.load_state_dict(torch.load("student.pt"))

        
        t_latents = sample_latents(num_generated_samples, tail='student', nu=nu).to(device)
        t_samples = t_edm_model(t_latents, sigma).to(device)
        t_gen = t_samples.detach().cpu().numpy()
        t_gen = t_gen*std+mean


        # plt.subplot(1, 2, 1)
        plt.figure()
        plt.scatter(raw_data_np[:, 1], raw_data_np[:, 0], s=0.5, alpha=0.5)
        plt.title(f"Student Neal’s Funnel Samples (nu1={nu[0]},nu2={nu[1]})")
        plt.xlabel("x1"), plt.ylabel("x2")

        # plt.subplot(1, 2, 2)
        plt.scatter(t_gen[:, 1], t_gen[:, 0], s=0.5, alpha=0.5, color='orange')
        # plt.title("Generated Samples from Model")
        # plt.xlabel("x1"), plt.ylabel("x2")

        plt.tight_layout() 
        plt.xlim((-500,500))
        plt.ylim((-10,10))
        plt.savefig("Student.png")

        # visualize_samples(t_samples, title="t-EDM (ν=4)")

    if 'frechet' in distribution:
        # Generate samples (t-EDM with ν=4)
        f_edm_model = Denoiser().to(device)
        if train:
            train_denoiser(f_edm_model, dataloader, epochs=num_epochs, tail="frechet",alpha=alpha,scale=scale)
            torch.save(f_edm_model.state_dict(),"frechet.pt")
        else:
            f_edm_model.load_state_dict(torch.load("frechet.pt"))

        
        f_latents = sample_latents(num_generated_samples, tail='frechet', alpha=alpha,scale=scale).to(device)
        f_samples = f_edm_model(f_latents, sigma).to(device)
        f_gen = f_samples.detach().cpu().numpy()
        f_gen = f_gen*std+mean


        # plt.subplot(1, 2, 1)
        plt.figure()

        plt.scatter(raw_data_np[:, 1], raw_data_np[:, 0], s=0.5, alpha=0.5)
        plt.title("Frechet Neal’s Funnel Samples")
        plt.xlabel("x1"), plt.ylabel("x2")

        # plt.subplot(1, 2, 2)
        plt.scatter(f_gen[:, 1], f_gen[:, 0], s=0.5, alpha=0.5, color='orange')
        # plt.title("Generated Samples from Model")
        # plt.xlabel("x1"), plt.ylabel("x2")

        plt.tight_layout()
        plt.xlim((-500,500))
        plt.ylim((-10,10))
        plt.savefig("Frechet.png")

    if 'gamma' in distribution:
        # Generate samples (t-EDM with ν=4)
        f_edm_model = Denoiser().to(device)
        if train:
            train_denoiser(f_edm_model, dataloader, epochs=num_epochs, tail="gamma",gamma_shape=gamma_shape, gamma_scale=gamma_scale)
            torch.save(f_edm_model.state_dict(),"gamma.pt")
        else:
            f_edm_model.load_state_dict(torch.load("gamma.pt"))

        
        f_latents = sample_latents(num_generated_samples, tail='gamma', gamma_shape=gamma_shape, gamma_scale=gamma_scale).to(device)
        f_samples = f_edm_model(f_latents, sigma).to(device)
        f_gen = f_samples.detach().cpu().numpy()
        f_gen = f_gen*std+mean


        # plt.subplot(1, 2, 1)
        plt.figure()

        plt.scatter(raw_data_np[:, 1], raw_data_np[:, 0], s=0.5, alpha=0.5)
        plt.title("Gamma Neal’s Funnel Samples")
        plt.xlabel("x1"), plt.ylabel("x2")

        # plt.subplot(1, 2, 2)
        plt.scatter(f_gen[:, 1], f_gen[:, 0], s=0.5, alpha=0.5, color='orange')
        # plt.title("Generated Samples from Model")
        # plt.xlabel("x1"), plt.ylabel("x2")

        plt.tight_layout()
        plt.xlim((-500,500))
        plt.ylim((-10,10))
        plt.savefig("Gamma.png")
        # visualize_samples(f_samples, title="f-EDM ")

