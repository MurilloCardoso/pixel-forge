🎮 PixelSpriteGAN (Early Stage)

PixelSpriteGAN is an early-stage AI project that generates 32x32 pixel art character sprites using a simple GAN trained with PyTorch.
The goal is to build an interactive tool for creating character sprites for portfolios, game prototypes, and generative AI demos.

⚡ Features

Generate 32x32 pixel art sprites.

Supports multiple character classes (fighter, mage, archer, etc.).

Simple web interface (optional).

Fully implemented in PyTorch.

🚀 Usage

Add 32x32 sprites to dataset/.

Train the GAN:

python train_gan.py


Generate new sprites:

python generate_sprites.py


(Optional) Run the web interface with FastAPI.

📌 Notes

Project is for study and experimentation, not production.

Sprite quality depends on dataset size and diversity.

🔗 References

PyTorch GAN Tutorial

OpenGameArt – Pixel Art Dataset
