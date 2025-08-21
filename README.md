# 🎮 PixelSpriteGAN (Early Stage)

PixelSpriteGAN is an early-stage AI project that generates **32x32 pixel art character sprites** using a simple GAN trained with PyTorch.  
The goal is to build an interactive tool for creating character sprites for portfolios, game prototypes, and generative AI demos.

![This is an alt text.](https://github.com/MurilloCardoso/pixel-forge/blob/main/pixel/scripts/generated/epoch_500.png?raw=true "This is a sample image.")

## ⚡ Features

- Generate **32x32 pixel art sprites**.
- Supports multiple character classes (e.g., fighter, mage, archer).
- Simple web interface (optional).
- Fully implemented in **PyTorch**.

## 🚀 Usage

1. Add **32x32 sprites** to the `dataset/` directory.
2. Train the GAN:
    ```bash
    python train_gan.py
    ```
3. Generate new sprites:
    ```bash
    python generate_sprites.py
    ```
4. (Optional) Run the web interface with **FastAPI**.

## 📌 Notes

- This project is for **study and experimentation**, not production.
- Sprite quality depends on **dataset size and diversity**.

## 🔗 References

- [PyTorch GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [OpenGameArt – Pixel Art Dataset](https://opengameart.org/)
