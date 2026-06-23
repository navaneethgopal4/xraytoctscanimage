# SynthoCT: X-ray to CT Image Translation with CycleGAN

SynthoCT is a PyTorch-based CycleGAN project that trains a neural network to translate 2D chest X-ray textures into 3D axial CT-like slices. This unpaired image translation is extremely useful for generating synthetic data, transfer learning, and prototyping in medical imaging research.

---

## 🚀 Getting Started

### 1. Install Dependencies
Install all required libraries using pip:
```bash
pip install -r requirements.txt
```

### 2. Generate or Acquire Datasets
To quickly test and verify the entire pipeline, you can generate a detailed, anatomically matching synthetic medical image dataset of 1000 training and 200 testing images:
```bash
# Full dataset (1000 pairs - fast, complete in seconds):
python download_data.py

# Quick testing dataset (100 train, 20 test):
python download_data.py --quick
```
This creates the following structure:
```
data/
├── xray/
│   ├── train/     ← 2D chest X-ray phantoms (lungs, ribs, spine, heart)
│   └── test/      ← 2D chest X-ray phantoms
└── ct/
    ├── train/     ← Axial CT slice phantoms (vertebra, outer fat, lungs, vessels)
    └── test/      ← Axial CT slice phantoms
```

---

## 🏋️ Training the Model

To train the CycleGAN model on your dataset:

### On CPU (For quick structure testing):
```bash
python train.py --epochs 5 --decay_epoch 2 --batch_size 2 --sample_interval 10 --checkpoint_interval 2
```

### On GPU (Colab / Kaggle / Local CUDA - Recommended):
```bash
python train.py --epochs 100 --decay_epoch 50 --batch_size 1 --checkpoint_interval 10
```

*Notes:*
- Checkpoints will be saved in `checkpoints/` (e.g. `checkpoints/checkpoint_epoch_10.pt`).
- Progress image grids will be saved in `samples/` showing Real X-ray, Fake CT, Reconstructed X-ray, Real CT, Fake X-ray, and Reconstructed CT.

---

## 📈 Evaluation

Evaluate translation quality using **SSIM** (Structural Similarity Index) and **PSNR** (Peak Signal-to-Noise Ratio) over the test dataset:
```bash
python evaluate.py --checkpoint checkpoints/checkpoint_epoch_10.pt --data_dir data/
```
This prints the metrics and saves side-by-side comparison images into the `results/` directory.

---

## 💻 Run Web Demo (Gradio UI)

SynthoCT comes with a premium web UI containing custom CSS styling, dynamic tabs, and sample templates:

```bash
# Run with a trained model:
python app.py --checkpoint checkpoints/checkpoint_epoch_10.pt

# Run in Simulated Demo mode (Instant testing - no model training required!):
python app.py
```

Options:
- `--share`: Generates a public Gradio URL (perfect for sharing Google Colab runs).
- `--port`: Port to host the app on (default is 7860).

---

## 🛠️ Project Structure
- `download_data.py`: Multi-domain dataset downloader and synthetic medical chest phantom generator.
- `models/generator.py`: Standard 9-block ResNet Generator.
- `models/discriminator.py`: 70x70 PatchGAN Discriminator.
- `utils/dataset.py`: PyTorch custom dataset with data normalization and medical-safe augmentations.
- `train.py`: LSGAN and Cycle-consistency training pipeline with linear LR schedulers and image buffer.
- `test.py`: Fast translation inference script.
- `evaluate.py`: SSIM/PSNR calculation and image grid generation script.
- `app.py`: Gradio web UI.
- `requirements.txt`: Python package requirements.
