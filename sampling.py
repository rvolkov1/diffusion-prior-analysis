import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


from model import create_model
from diffusion import GaussianDiffusion, StudentTDiffusion, GammaDiffusion, FrechetDiffusion
from dataset import LongTailCIFAR100

def show_image(tensor_img,label=None):
    # Normalize to [0, 1] for visualization
    img = tensor_img.squeeze(0).detach().cpu()
    img = (img - img.min()) / (img.max() - img.min())
    img = T.ToPILImage()(img)
    plt.imshow(img)
    plt.axis("off")
    if label!=None:
        plt.title(label)
    plt.show()

def visualize_one_sample_per_class(ddpm,label_dict):
    ddpm.model.eval()
    with torch.no_grad():
        # class_labels = torch.arange(0,100,10).to(ddpm.device)
        class_labels = [58]*10
        class_labels = torch.tensor(class_labels,device=ddpm.device)
        samples = ddpm.sample(class_labels, shape = (10, 3, 32, 32)).cpu()
        samples = (samples + 1) / 2  # De-normalize to [0, 1]

        print("Class labels:", class_labels)
        print("Label titles:", [label_dict[i.item()] for i in class_labels])

        fig, axes = plt.subplots(1, 10, figsize=(15, 2))
        for i, ax in enumerate(axes):
            img = samples[i]
            img = (img - img.min()) / (img.max() - img.min())
            # img = transforms.ToPILImage()(img)
            ax.imshow(img.permute(1, 2, 0))
            ax.axis("off")
            ax.set_title(label_dict[class_labels[i].item()], fontsize=8)
        plt.tight_layout()
        plt.show()

if __name__=="__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    dataset = LongTailCIFAR100(root="./data",transform=transform,imb_factor=0.1)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    model = create_model("weights.pth").to(device)
    diffusion = GaussianDiffusion(model,
                                  device = device)

    diffusion.embed_class_labels.load_state_dict(torch.load("embeddings.pth"))

    model.eval()
    fid = FrechetInceptionDistance().to(device)
    inception = InceptionScore().to(device)
    batch_size = 5
    for real_imgs,label in dataloader:
        real_imgs = real_imgs.to(device)
        real_imgs = (real_imgs.clamp(0,1) * 255).to(torch.uint8)
        fid.update(real_imgs,real=True)
    
    for class_label in range(2):
        print(class_label)
        gen_imgs = diffusion.sample([class_label]*batch_size, shape=(batch_size, 3, 32, 32))
        gen_imgs = (gen_imgs.clamp(0,1) * 255).to(torch.uint8)
        fid.update(gen_imgs.to(device), real=False)
        inception.update(gen_imgs)

    # Compute FID
    score = fid.compute()
    inception_score = inception.compute()
    print("FID:", score.item())
    print(f"IS: {inception_score}")
    label_dict = dataset.idx_to_class
    # visualize_one_sample_per_class(diffusion,label_dict)
     # class_label = 1
    # sample = diffusion.sample(class_label=class_label)
    # print(f"Label: {dataset.idx_to_class[class_label]}")
    # show_image(sample)