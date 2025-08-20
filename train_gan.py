import torch
from torchvision.utils import save_image

# Importa sua classe Generator
from .generate_sprites import Generator

device = "cuda" if torch.cuda.is_available() else "cpu"
latent_dim = 100

# Carregar modelo treinado
G = Generator(latent_dim)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.to(device)
G.eval()

# Gerar sprites
z = torch.randn(16, latent_dim).to(device)
fake_imgs = G(z)
save_image(fake_imgs.data, "generated/new_sprites.png", nrow=4, normalize=True)
