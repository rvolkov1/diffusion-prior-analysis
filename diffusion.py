import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils import student_t_from_inv_gamma_nll

class GaussianDiffusion:
    def __init__(self, 
                 model, 
                 timesteps=1000, 
                 beta_start=1e-4, 
                 beta_end=0.02,
                 class_embed_size = 8, 
                 num_classes = 100,
                 device = None):
        self.device = torch.device("cpu") if device is None else device
        self.model = model
        self.timesteps = timesteps
        self.embed_class_labels = nn.Embedding(num_classes,class_embed_size).to(device)


        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_prod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[t]) ** 0.5
        return sqrt_alpha_prod[:, None, None, None] * x_start + sqrt_one_minus_alpha_prod[:, None, None, None] * noise

    def p_losses(self, x_start, t, y):
        bs, ch, w ,h = x_start.shape
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)

        class_emb = self.embed_class_labels(y)
        class_emb = class_emb.view(bs, class_emb.shape[1],1,1).expand(bs,class_emb.shape[1],w,h)

        model_input = torch.cat((x_noisy,class_emb),dim=1)

        predicted_noise = self.model(
            model_input,
            timestep=t
        ).sample
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def sample(self, class_label, shape=(3,32,32), num_steps=1000):
        bs, ch, w ,h = shape
        x = torch.randn(shape).to(self.device)

        if isinstance(class_label, int):
            class_label = torch.tensor([class_label], device=self.device)
        elif isinstance(class_label, list):
            class_label = torch.tensor(class_label, device=self.device)
        else:
            class_label = class_label.to(self.device)

        class_emb = self.embed_class_labels(class_label)
        class_emb = class_emb.view(bs, class_emb.shape[1],1,1).expand(bs,class_emb.shape[1],w,h)

        for t in reversed(range(num_steps)):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)

            # Predict noise
            model_input = torch.cat((x,class_emb),dim=1)
            output = self.model(model_input, timestep=t_tensor)
            eps = output.sample  # predicted noise

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]

            # Compute the mean of p(x_{t-1} | x_t)
            mean = (1 / alpha_t**0.5) * (x - (beta_t / (1 - alpha_bar_t)**0.5) * eps)

            if t > 0:
                noise = torch.randn_like(x, device=self.device)
                x = mean + beta_t**0.5 * noise
            else:
                x = mean

        return x

def sample_student_t(shape, df, device):
    # Sample from Student-t using: t = Z / sqrt(V / df), where
    # Z ~ N(0,1), V ~ chi2(df)
    normal = torch.randn(shape, device=device)
    chi2 = torch.distributions.Chi2(df).sample((shape[0],)).to(device)
    chi2 = chi2.view(-1, *([1] * (len(shape) - 1)))  # shape broadcast
    return normal / torch.sqrt(chi2 / df)

class StudentTDiffusion:


    def __init__(self, 
                 model, 
                 timesteps=1000, 
                 beta_start=1e-4, 
                 beta_end=0.02,
                 class_embed_size = 8, 
                 num_classes = 100,
                 device = None):
        self.device = torch.device("cpu") if device==None else device
        self.model = model
        self.timesteps = timesteps
        self.embed_class_labels = nn.Embedding(num_classes,class_embed_size).to(device)


        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_prod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[t]) ** 0.5
        return sqrt_alpha_prod[:, None, None, None] * x_start + sqrt_one_minus_alpha_prod[:, None, None, None] * noise

    def p_losses(self, x_start, t, y, df=3):
        bs, ch, w ,h = x_start.shape
        noise = sample_student_t(x_start.shape, df=df, device=self.device)
        x_noisy = self.q_sample(x_start, t, noise)

        class_emb = self.embed_class_labels(y)
        class_emb = class_emb.view(bs, class_emb.shape[1],1,1).expand(bs,class_emb.shape[1],w,h)

        model_input = torch.cat((x_noisy,class_emb),dim=1)

        predicted_noise = self.model(
            model_input,
            timestep=t
        ).sample
        return F.mse_loss(predicted_noise, noise)
        # return student_t_from_inv_gamma_nll(predicted_noise,noise) #Student-t nll to see if it improves performance
        # # Student-t loss (e.g., NLL)
        # scale = 1.0  # assume scale=1, or output it from model
        # resid = (predicted_noise - noise) / scale
        # nll = torch.lgamma(torch.tensor((df + 1) / 2)) - torch.lgamma(torch.tensor(df / 2)) \
        #     - 0.5 * torch.log(torch.tensor(df * math.pi * scale ** 2)) \
        #     - ((df + 1) / 2) * torch.log(torch.tensor(1 + (resid ** 2) / df))
        # return -nll.mean()
    
    @torch.no_grad()
    def sample(self, class_label, shape=(3,32,32), num_steps=1000, df=3):
        bs, ch, w ,h = shape
        x = torch.randn(shape).to(self.device)

        if isinstance(class_label, int):
            class_label = torch.tensor([class_label], device=self.device)
        elif isinstance(class_label, list):
            class_label = torch.tensor(class_label, device=self.device)
        else:
            class_label = class_label.to(self.device)

        class_emb = self.embed_class_labels(y)
        class_emb = class_emb.view(bs, class_emb.shape[1],1,1).expand(bs,class_emb.shape[1],w,h)

        
        # return F.mse_loss(predicted_noise, noise)

        for t in reversed(range(num_steps)):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)

            # Predict noise
            model_input = torch.cat((x,class_emb),dim=1)
            output = self.model(model_input, timestep=t_tensor)
            eps = output.sample  # predicted noise

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]

            # Compute the mean of p(x_{t-1} | x_t)
            mean = (1 / alpha_t**0.5) * (x - (beta_t / (1 - alpha_bar_t)**0.5) * eps)

            if t > 0:
                noise = sample_student_t(x.shape, df=df, device=self.device)
                x = mean + beta_t**0.5 * noise
            else:
                x = mean  # final denoised image

        return x

class GammaDiffusion:
    def __init__(self, 
                 model, 
                 timesteps=1000, 
                 beta_start=1e-4, 
                 beta_end=0.02, 
                 g_alpha=2.0, 
                 g_beta=1.0, 
                 class_embed_size = 8, 
                 num_classes = 100, 
                 device = None):
        self.device = torch.device("cpu") if device==None else device
        self.model = model
        self.timesteps = timesteps
        self.embed_class_labels = nn.Embedding(num_classes,class_embed_size).to(device)
        self.g_alpha = g_alpha
        self.g_beta = g_beta

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def sample_gamma(self, shape, device="cpu"):
        gamma1 = torch.distributions.Gamma(self.g_alpha,self.g_beta).sample(shape).to(device)
        
        # Standardize per sample in the batch
        dims = tuple(range(1, len(shape)))  # exclude batch dim
        mean = gamma1.mean(dim=dims, keepdim=True)
        std = gamma1.std(dim=dims, keepdim=True) + 1e-8  # avoid div by zero
        
        return (gamma1 - mean) / std

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_prod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[t]) ** 0.5
        return sqrt_alpha_prod[:, None, None, None] * x_start + sqrt_one_minus_alpha_prod[:, None, None, None] * noise

    def p_losses(self, x_start, t, y, df=3):
        bs, ch, w ,h = x_start.shape

        noise = self.sample_gamma(x_start.shape, device=self.device)
        x_noisy = self.q_sample(x_start, t, noise)

        class_emb = self.embed_class_labels(y)
        class_emb = class_emb.view(bs, class_emb.shape[1],1,1).expand(bs,class_emb.shape[1],w,h)

        model_input = torch.cat((x_noisy,class_emb),dim=1)

        predicted_noise = self.model(
            model_input,
            timestep=t
        ).sample
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def sample(self, class_label, shape=(3,32,32), num_steps=1000, df=3):
        bs, ch, w ,h = shape
        x = torch.randn(shape).to(self.device)

        if isinstance(class_label, int):
            class_label = torch.tensor([class_label], device=self.device)
        elif isinstance(class_label, list):
            class_label = torch.tensor(class_label, device=self.device)
        else:
            class_label = class_label.to(self.device)

        class_emb = self.embed_class_labels(class_label)
        class_emb = class_emb.view(bs, class_emb.shape[1],1,1).expand(bs,class_emb.shape[1],w,h)

        

        for t in reversed(range(num_steps)):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)

            # Predict noise
            model_input = torch.cat((x,class_emb),dim=1)
            output = self.model(model_input, timestep=t_tensor)
            eps = output.sample  # predicted noise

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]

            # Compute the mean of p(x_{t-1} | x_t)
            mean = (1 / alpha_t**0.5) * (x - (beta_t / (1 - alpha_bar_t)**0.5) * eps)

            if t > 0:
                noise = self.sample_gamma(x.shape, device=self.device)
                x = mean + beta_t**0.5 * noise
            else:
                x = mean  # final denoised image

        return x
    
class FrechetDiffusion:
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02, f_alpha=4.0, f_beta=1.0, device = None):
        self.device = torch.device("cpu") if device==None else device
        self.model = model
        self.timesteps = timesteps
        self.embed_class_labels = nn.Embedding(100,1280).to(device)
        self.f_alpha = f_alpha
        self.f_beta = f_beta

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

    def sample_frechet(self, shape, device="cpu"):
        U = torch.rand(shape, device=device).clamp(min=1e-6)
        
        frechet = self.f_beta * (-torch.log(U))**(-1.0 / self.f_alpha)
        # Standardize per sample in the batch
        dims = tuple(range(1, len(shape)))  # exclude batch dim
        mean = frechet.mean(dim=dims, keepdim=True)
        std = frechet.std(dim=dims, keepdim=True) + 1e-8  # avoid div by zero
        
        return (frechet - mean) / std

    def q_sample(self, x_start, t, noise):
        sqrt_alpha_prod = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[t]) ** 0.5
        return sqrt_alpha_prod[:, None, None, None] * x_start + sqrt_one_minus_alpha_prod[:, None, None, None] * noise

    def p_losses(self, x_start, t, y, df=3):
        bs, ch, w ,h = x_start.shape
        noise = self.sample_frechet(x_start.shape, device=self.device)
        x_noisy = self.q_sample(x_start, t, noise)

        class_emb = self.embed_class_labels(y)
        class_emb = class_emb.view(bs, class_emb.shape[1],1,1).expand(bs,class_emb.shape[1],w,h)

        model_input = torch.cat((x_noisy,class_emb),dim=1)

        predicted_noise = self.model(
            model_input,
            timestep=t
        ).sample
        return F.mse_loss(predicted_noise, noise)
    
    @torch.no_grad()
    def sample(self, class_label, shape=(3,32,32), num_steps=1000, df=3):
        bs, ch, w ,h = shape
        x = torch.randn(shape).to(self.device)

        if isinstance(class_label, int):
            class_label = torch.tensor([class_label], device=self.device)
        elif isinstance(class_label, list):
            class_label = torch.tensor(class_label, device=self.device)
        else:
            class_label = class_label.to(self.device)

        class_emb = self.embed_class_labels(y)
        class_emb = class_emb.view(bs, class_emb.shape[1],1,1).expand(bs,class_emb.shape[1],w,h)

        for t in reversed(range(num_steps)):
            t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)

            # Predict noise
            model_input = torch.cat((x,class_emb),dim=1)
            output = self.model(model_input, timestep=t_tensor)
            eps = output.sample  # predicted noise

            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]

            # Compute the mean of p(x_{t-1} | x_t)
            mean = (1 / alpha_t**0.5) * (x - (beta_t / (1 - alpha_bar_t)**0.5) * eps)

            if t > 0:
                noise = self.sample_frechet(x.shape, device=self.device)
                x = mean + beta_t**0.5 * noise
            else:
                x = mean  # final denoised image

        return x