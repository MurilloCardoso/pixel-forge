from fastapi import FastAPI
from fastapi.responses import FileResponse
import torch
from torchvision.utils import save_image
from train_gan import Generator
import os

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 100
G = Generator(latent_dim)
G.load_state_dict(torch.load("generator.pth", map_location=device))
G.to(device)
G.eval()

@app.get("/generate")
def generate_sprites():
    z = torch.randn(16, latent_dim).to(device)
    fake_imgs = G(z)
    os.makedirs("generated", exist_ok=True)
    save_path = "generated/web_sprites.png"
    save_image(fake_imgs.data, save_path, nrow=4, normalize=True)
    return FileResponse(save_path)
