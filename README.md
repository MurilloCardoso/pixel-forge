# 🎮 PixelSpriteGAN

PixelSpriteGAN é um projeto de inteligência artificial que gera sprites animados em pixel art (32x32) de personagens de jogos, usando um GAN simples treinado em PyTorch.  
O objetivo é criar uma ferramenta interativa para geração de sprites de personagens variados, útil para portfólio, protótipos de jogos e demonstração de IA generativa.

# Primeiros Testes - Sem CLIP - SEM GIF

Epoch = 500 , Dataset = 1000
![This is an alt text.](https://github.com/MurilloCardoso/pixel-forge/blob/main/pixel/scripts/generated/epoch_500.png?raw=true  "This is a sample image.")

## ⚡ Funcionalidades

- Geração de sprites 32x32 em estilo pixel art.
- Possibilidade de criar sprite sheets animados (vários frames).
- Suporte a diferentes classes de personagens (ex.: lutador, mago, arqueiro).
- Fácil integração com interface web, permitindo gerar sprites via navegador.
- Código totalmente em PyTorch, ideal para estudo de GANs e IA generativa.

## 🛠️ Tecnologias utilizadas

- **Python 3.10+**
- **PyTorch** (modelo Generator/Discriminator)
- **Torchvision** (pré-processamento e salvamento de imagens)
- **FastAPI** (opcional, para API web)
- **HTML/JS** (frontend simples para gerar sprites)

## 📂 Estrutura do projeto

```plaintext
PixelSpriteGAN/
│
├─ dataset/           # Sprites de treino (32x32 PNG)
├─ generated/         # Sprites gerados durante testes
├─ train_gan.py       # Script de treino do GAN
├─ generate_sprites.py # Script para gerar sprites a partir do modelo treinado
├─ web_interface/     # Frontend simples (HTML + JS)
├─ models/            # Modelos treinados (.pth)
└─ README.md
```

## 🚀 Como usar

### 1️⃣ Preparar o dataset

- Crie a pasta `dataset/` e coloque sprites 32x32 PNG.
- Estruture por classes, se quiser condicionamento:

```plaintext
dataset/lutador/
dataset/mago/
dataset/arqueiro/
```

### 2️⃣ Treinar o GAN

```bash
python train_gan.py
```

- O script treina o Generator e Discriminator.
- A cada 10 epochs, salva sprites de teste em `generated/`.
- Ao final do treino, salve o modelo:

```python
torch.save(G.state_dict(), "models/generator.pth")
```

### 3️⃣ Gerar sprites novos

```bash
python generate_sprites.py
```

- Gera 16 sprites 32x32 e salva em `generated/new_sprites.png`.
- Pode ser integrado a frontend web ou usado diretamente.

### 4️⃣ Executar interface web (opcional)

- Rodar FastAPI:

```bash
uvicorn web_app:app --reload
```

- Abrir [http://localhost:8000](http://localhost:8000) e clicar em **Gerar Sprite** para ver resultados.

## 🎨 Personalização

- Adicione novas classes de personagens no dataset para o modelo aprender novas profissões ou tipos.
- Ajuste o tamanho do latent vector ou arquitetura para gerar sprites mais detalhados.
- Experimente usar ConvGAN em vez de MLP para resultados mais consistentes em pixel art.

## 📌 Observações

- Projeto voltado para portfólio e estudo, não para produção.
- A qualidade dos sprites depende da quantidade e diversidade do dataset.
- Sprites gerados podem ser usados em protótipos de jogos ou animações simples.

## 🔗 Referências

- [PyTorch GAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)
- [Pixel Art Dataset (OpenGameArt)](https://opengameart.org/)
- [Denoising Diffusion / DDPM](https://arxiv.org/abs/2006.11239) – referência se você quiser evoluir para difusão.
