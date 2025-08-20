import torch
from torchvision.utils import save_image
from generate_sprites import Generator  # só precisa do Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 100

# Carregar modelo treinado
G = Generator(latent_dim).to(device)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.eval()  # modo inferência

# Gerar sprites
z = torch.randn(16, latent_dim).to(device)
fake_imgs = G(z)

# Salvar
import os
os.makedirs("inference", exist_ok=True)
save_image(fake_imgs, "inference/sprites.png", nrow=4, normalize=True)

print("Sprites gerados ✅")
