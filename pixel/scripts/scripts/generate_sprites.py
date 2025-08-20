import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from PIL import Image

# -------------------------------
# Dataset
# -------------------------------
class SpriteDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith((".png", ".jpg"))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = SpriteDataset("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------------
# GAN Models
# -------------------------------
class Generator(nn.Module):
    def __init__(self, z_dim=100, ngf=64, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),

            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z.view(z.size(0), z.size(1), 1, 1))

class Discriminator(nn.Module):
    def __init__(self, ndf=64, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x):
        return self.net(x).view(-1, 1)

# -------------------------------
# Training setup
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 100

G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCEWithLogitsLoss() 
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 200
for epoch in range(epochs):
    for imgs in loader:
        imgs = imgs.to(device)
        batch_size = imgs.size(0)

        # Labels suavizados
        real_labels = torch.full((batch_size,1), 0.9, device=device)
        fake_labels = torch.full((batch_size,1), 0.0, device=device)

        # --- Train Discriminator ---
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)

        D_loss_real = criterion(D(imgs), real_labels)
        D_loss_fake = criterion(D(fake_imgs.detach()), fake_labels)
        D_loss = (D_loss_real + D_loss_fake)/2

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # --- Train Generator ---
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)
        G_loss = criterion(D(fake_imgs), real_labels)  # tenta enganar o D

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")

    if (epoch+1) % 20 == 0:
        os.makedirs("generated", exist_ok=True)
        save_image(fake_imgs.data[:16], f"generated/epoch_{epoch+1}.png", nrow=4, normalize=True)

torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")

print("Training finished and models saved.")