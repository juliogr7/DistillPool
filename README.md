# DistillPool: Knowledge Distillation Techniques for Single-Pixel Imaging

Reconstructing images from compressed measurements is a main inverse problem in computational imaging, with Single-Pixel Imaging (SPI) as a flagship example. Although deep learning methods have shown strong performance in this setting, acquiring high-fidelity measurements remains a physical constraint due to the trade-off between the undersampling ratio and the optical sensor cost limits. Knowledge Distillation (KD) has emerged as a paradigm to improve performance under these constraints, by leveraging the supervision of models trained with simulated low-constrained acquisition (teachers) to guide the training of models with inputs from high-constrained and feasible optical encoders (students). This paper evaluates KD cross-techniques for SPI  distilling the knowledge between the teachers and students network. This study explores training under various undersampling ratios, leveraging response-, feature-, and relation-based distillation strategies. Our results demonstrate that specific combinations of distillation schemes significantly improve reconstruction quality for the student model, achieving up to 1.49 dB gain in PSNR metric over non-distilled training, highlighting KD as an effective framework to enhance reconstruction in physically constrained imaging systems without increasing sensing complexity.

## Key Features
- Feature distillation (encoder, latent, decoder, output combinations)
- Supervised (ground-truth) or teacher-supervised (pseudo-label) training
- Optional additive white Gaussian noise (AWGN) robustness
- Feature visualization (first-channel early feature snapshot)
- WandB experiment tracking (PSNR / SSIM / loss curves)

## Repository Structure
Root scripts:
- `main_unet.py`            : Train a UNet at a specified compression ratio (used for teacher pretraining and baseline)
- `main_unet_dist.py`       : Standard knowledge distillation (student with single teacher)
- `progressive_distill.py`  : Progressive distillation (supports redistillation and advanced modes)
- `test_UNet.py`            : Evaluate a trained model and save reconstructions + first encoder feature (channel 0)
- `visualize_mask_spc.py`   : Inspect Single-Pixel Camera sensing masks

Models:
- `model_unet.py`           : Base UNet
- `model_unet_dist.py`      : UNet variant with feature extraction for distillation
- `model_hybrid_unet.py`    : Hybrid UNet (progressive distillation scenarios)

Utilities / Algorithms:
- `data_loader.py`          : Dataset and DataLoader builders (MNIST / CelebA)
- `algo_functions.py`       : Physics factory, meters, noise injection
- `progressive_distill.py`  : Includes matrix-based relational distillation loss

Other:
- `deepinv/`                : External library

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/juliogr7/DistillPool.git
   cd DistillPool
   ```
2. Create environment and Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The file `requirements.txt` is derived from `env_distillpool.yml` and pins versions (includes CUDA 12.4 wheels for PyTorch). If you need CPU-only PyTorch, override the first torch/torchvision/torchaudio lines with CPU packages.

## Train a Teacher
Use `main_unet.py` with desired undersampling ratio `--cr` (ideally higher than student). Output weights saved under `results/trained_models`.

## Evaluation & Feature Visualization
Use `test_UNet.py` to:
- Reconstruct test samples
- Save: measurement backprojection (x_0), reconstruction, original, and first encoder feature (channel 0 only) as a grayscale PNG
- Log PSNR / SSIM into a CSV

## Distillation Configuration
### Feature Selection Modes (type_dist)
Pass an integer that indexes the ordered list:
- 0: latent
- 1: all (every stored feature scale including latent)
- 2: first_latent (first encoder + latent)
- 3: output (reconstruction only)
- 4: output_latent (output + latent)
- 5: encoder (first encoder block only)
- 6: encoder_latent (first encoder + latent)
- 7: latent_decoder (latent + decoder pathway features)
- 8: output_latent_decoder (output + latent + decoder)
- 9: output_all (output + all features)
- 10: first (first encoder only)
- 11: output_first_latent (output + first encoder + latent)
- 12: output_first (output + first encoder)

### Loss Functions
Argument: `--loss_distill`
- mse / mae / l1 : pointwise regression losses
- cs            : cosine similarity (converted to distance 1 - cos)
- l2            : vector L2 norm of difference
- linf          : max-absolute (L-infinity) difference
- comb_matrix   : relational Frobenius distance between RBF similarity matrices of feature sets (`--gamma` controls kernel width via exp(-gamma * ||xi - xj||^2))

### Noise Augmentation
Enable AWGN on measurements with `--awgn_snr <SNR_dB>` to test robustness. Adds noise to student (and optionally teacher) measurement streams before backprojection.

## Physics: Single-Pixel Camera
The physics module builds a Single-Pixel Camera forward operator using binary masks. For a target compression ratio cr, the number of measurements m ≈ cr * n^2 (validated at runtime and printed). The adjoint operator provides an initial backprojection x_0 used as network input.

## Experiment Logging
Weights & Biases (WandB) is integrated.

## Results Directory Layout
results/
  trained_models/
    UNET_cr_<ratio>.pth                      (teacher checkpoints)
    <date>_student_weights_... .pth          (student distilled weights)
  test/
    (Evaluations from `test_UNet.py`)

## Citation
If this work contributes to your research, please cite:

```bibtex
@INPROCEEDINGS{11156574,
  author    = {Gutierrez-Rengifo, Julio and Diaz-Delgado, Laura C. and Arguello, Paula and Jacome, Roman and Arguello, Henry},
  booktitle = {2025 XXV Symposium of Image, Signal Processing, and Artificial Vision (STSIVA)},
  title     = {DistillPool: Knowledge Distillation Techniques for Single-Pixel Imaging},
  year      = {2025},
  volume	= {},
  number	= {},
  pages     = {1-5},
  keywords  = {Training; Deep learning; Magnetic resonance imaging; Imaging; Signal processing; Optical variables measurement; Optical imaging; Optical sensors; Image reconstruction; Signal to noise ratio; Single-Pixel Imaging; Deep Learning; Knowledge Distillation; Computational Imaging},
  doi       = {10.1109/STSIVA66383.2025.11156574}
}
```



## License

This repository is released under the MIT License.






