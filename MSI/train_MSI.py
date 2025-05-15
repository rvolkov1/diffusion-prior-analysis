import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from torch.utils.data import Dataset
import torch.multiprocessing as mp
from torch.distributions.studentT import StudentT
import torch.distributed as dist
import os, torch, torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import numpy as np
import os
import pickle
from diffusers import UNet2DConditionModel

from models import UNet2
from GammaDDPM import GammaDDPM, FrechetDDPM, StudentTDDPM, DDPM

class CIFAR100LongTail(Dataset):
    def __init__(self, root, phase='train', imbalance_factor=0.01, transform=None):
        self.root = root
        self.phase = phase
        self.transform = transform
        self.num_classes = 100
        self.imgs, self.labels = self._make_longtail(imbalance_factor)

    def _make_longtail(self, imbalance_factor):
        print("Making longtail dataset")
        cifar = CIFAR100(self.root, train=(self.phase == 'train'), download=False)
        data, targets = cifar.data, np.array(cifar.targets)
        cls_num = self.num_classes

        # Long tail class distribution
        cls_counts = []
        img_per_cls_max = len(targets) // cls_num
        for cls_idx in range(cls_num):
            num = img_per_cls_max * (imbalance_factor ** (cls_idx / (cls_num - 1)))
            cls_counts.append(int(num))

        new_data, new_targets = [], []
        for cls_idx, cls_count in enumerate(cls_counts):
            idx = np.where(targets == cls_idx)[0]
            np.random.shuffle(idx)
            sel = idx[:cls_count]
            new_data.append(data[sel])
            new_targets.extend([cls_idx] * cls_count)

        new_data = np.concatenate(new_data)
        return new_data, new_targets

    def __getitem__(self, index):
        img, target = self.imgs[index], self.labels[index]
        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.labels)

# ================================
#         CIFAR10 Dataset
# ================================
class ToMinusOneToOne:
    def __call__(self, x):
        return x * 2. - 1.
    
# ================================
#       Linear Beta Schedule
# ================================
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta (float): Minimum improvement to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

# ================================
#     Beefier UNet-like Model
# ================================
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb).to(t.device)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        print(in_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(4,in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(4,out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        time_emb = self.time_mlp(t).view(t.shape[0], -1, 1, 1)
        h = h + time_emb
        h = self.block2(h)
        return h + self.residual_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_classes = 100, time_emb_dim=256):
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        self.label_emb = nn.Embedding(num_classes, time_emb_dim)


        # Down
        self.down1 = ResidualBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Middle
        self.middle = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)

        # Up
        self.up1 = ResidualBlock(base_channels * 2 + base_channels * 2, base_channels, time_emb_dim)
        self.up2 = ResidualBlock(base_channels + base_channels, in_channels, time_emb_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, label, t):
        label_emb = self.label_emb(label)
        t_emb = self.time_embedding(t)
        t_emb = t_emb + label_emb


        # Down
        x1 = self.down1(x, t_emb)

        x2 = self.pool(x1)
        x2 = self.down2(x2, t_emb)

        x3 = self.pool(x2)

        # Middle
        x3 = self.middle(x3, t_emb)


        # Up
        x = self.upsample(x3)
        x = self.up1(torch.cat([x, x2], dim=1), t_emb)
        x = self.upsample(x)
        x = self.up2(torch.cat([x, x1], dim=1), t_emb)

        return x


def main(rank, world_size):


    # === DDP Init ===
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # === Dataset ===
  # === Dataset ===
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    dataset = CIFAR100LongTail(root='./data', imbalance_factor=0.01, transform=transform)
    val_dataset = CIFAR100LongTail(root='./data', phase='test', imbalance_factor=0.01, transform=transform)
    num_classes = dataset.num_classes

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True
    )

    print(len(dataloader))
    print(len(val_dataloader))

    # === Model ===
    # model = UNet2(in_channels=3, base_channels=192,num_classes=num_classes)
    model = UNet2DConditionModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        class_embed_type="timestep",  # Important for conditional generation
        num_class_embeds=100,  # CIFAR-100
    )
    model = model.to(rank)
    betas = linear_beta_schedule(timesteps=400)

    ddpm = DDPM(model,betas)  # Your custom scheduler
    model = DDP(model, device_ids=[rank])

    # === Training ===
    num_epochs = 250
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                               mode='min',
                                               factor=0.1,
                                               patience=10)
    early_stopper = EarlyStopping(patience=20, min_delta=1e-4)

    

    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        #Train
        sampler.set_epoch(epoch)
        model.train()
        for batch in dataloader:
            x = batch[0].to(ddpm.device)
            label = batch[1].to(ddpm.device)
            t = torch.randint(0, ddpm.timesteps, (x.size(0),), device=ddpm.device)
            loss = ddpm.p_losses(x, label, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        #Validation
        val_sampler.set_epoch(epoch)
        model.eval()
        for batch in val_dataloader:
            x = batch[0].to(ddpm.device)
            label = batch[1].to(ddpm.device)
            t = torch.randint(0, ddpm.timesteps, (x.size(0),), device=ddpm.device)
            loss = ddpm.p_losses(x, label, t)

            val_loss += loss.item()
        if rank == 0:
            early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}. Best val loss: {early_stopper.best_loss:.4f}")
            break


        if epoch%50==0 and rank==0:
            model_save_file = f"model_saves/ddpm_gamma_conditional_epoch{epoch}.pth"
            torch.save(model.state_dict(),model_save_file)
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} / Val Loss = {val_loss:.4f}")
        print(f"Current learning rate: {scheduler.get_last_lr()}",flush=True)
        scheduler.step(val_loss)

    if rank==0:
        model_save_file = f"model_saves/ddpm_gamma_conditional_epoch_final.pth"
        torch.save(model.state_dict(),model_save_file)
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)
