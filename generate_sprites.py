import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from PIL import Image
import glob

# -------------------------------
# 1️⃣ Dataset de sprites simples
# -------------------------------
class SpriteDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.files = glob.glob(f"{folder}/*.png")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

# Transformação: 32x32 e tensor normalizado [-1,1]
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = SpriteDataset("dataset", transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# -------------------------------
# 2️⃣ Modelo GAN simples
# -------------------------------

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, channels*32*32),
            nn.Tanh()
        )
    def forward(self, z):
        img = self.model(z)
        return img.view(z.size(0), 3, 32, 32)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(channels*32*32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# -------------------------------
# 3️⃣ Instanciar modelos e otimizadores
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 100
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

# -------------------------------
# 4️⃣ Treinamento simplificado
# -------------------------------
epochs = 50
for epoch in range(epochs):
    for imgs in loader:
        imgs = imgs.to(device)
        batch_size = imgs.size(0)

        # --- Treinar Discriminator ---
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)

        real_labels = torch.ones(batch_size,1).to(device)
        fake_labels = torch.zeros(batch_size,1).to(device)

        D_loss_real = criterion(D(imgs), real_labels)
        D_loss_fake = criterion(D(fake_imgs.detach()), fake_labels)
        D_loss = (D_loss_real + D_loss_fake)/2

        optimizer_D.zero_grad()
        D_loss.backward()
        optimizer_D.step()

        # --- Treinar Generator ---
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        G_loss = criterion(D(fake_imgs), real_labels)  # quer enganar o discriminator

        optimizer_G.zero_grad()
        G_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D_loss: {D_loss.item():.4f} | G_loss: {G_loss.item():.4f}")

    # Salvar algumas imagens geradas
    if (epoch+1) % 10 == 0:
        os.makedirs("generated", exist_ok=True)
        save_image(fake_imgs.data[:16], f"generated/epoch_{epoch+1}.png", nrow=4, normalize=True)
