from model_unet_dist import UNet
# from model_unet import UNet as UNet_basic
import torch
import argparse
import deepinv as dinv
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim

from tqdm import tqdm
from algo_functions import AverageMeter, AddGaussianNoiseSNR
from data_loader import get_dataloaders

import os
from datetime import datetime
import csv
import numpy as np

import matplotlib.pyplot as plt

torch.manual_seed(0)

def save_psnr_to_csv(args, psnr_global, ssim_global, output_dir, saved_batches=0):
    """Append global PSNR / SSIM metrics to CSV log (creates file with header if needed)."""
    csv_filename = os.path.join(output_dir, "psnr_results.csv")
    file_exists = os.path.isfile(csv_filename)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Data to be written
    row = [timestamp, args.n, args.dataset, args.batch_size, args.cr, args.c, args.physics, psnr_global, ssim_global, saved_batches]

    # Writing to CSV
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Date/Time", "Image Dim", "Dataset", "Batch Size", "CR", "Channels", "Physics", "PSNR Global", "SSIM Global", "Saved Batches"])
        writer.writerow(row)

    print(f"Global PSNR saved to {csv_filename}")

def main(args):
    """Evaluate trained UNet on test set and optionally save intermediate visualizations."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Config the cuda ID to use
    if args.device >= 0:
        args.device = f"cuda:{args.device}"
    else:  # If there is no cuda, use cpu
        args.device = "cpu"

    decoder = UNet(
        in_channels=args.c,
        out_channels=args.c,
        scales=2,  # Reduced depth
        residual=True,
        circular_padding=True,
        cat=True,
        bias=True,
        batch_norm=False,
    ).to(args.device)

    name = 'UNET_cr_09_scales_2_dataset_mnist_2025_05_04_21_38.pth'
    #name = 'UNET_cr_09_scales_2_dataset_celeba_2025_05_04_22_09.pth'
    path = os.path.join('results','trained_models',name)
    print(f"Loading weights from --> {path} \n")

    decoder.load_weights(path)

    measurements = int(args.n**2*args.cr)
            
    # Sensing mask are fixed and not trainable #SUGERENCIA APRENDER LA MASCARA 
    physics_s = dinv.physics.SinglePixelCamera(m=measurements,img_shape=(1,args.n,args.n),device=args.device, fast=True)    
    psnr = PeakSignalNoiseRatio().to(args.device)
    ssim_value = ssim(data_range=1.0).to(args.device)

    physics_t = dinv.physics.SinglePixelCamera(m=int(args.n**2*0.9),img_shape=(1,args.n,args.n),device=args.device, fast=True)

    if args.awgn_snr is not None:
        AWGN = AddGaussianNoiseSNR(snr = args.awgn_snr)

    # Loading the data and the rest of the pipeline
    _, testloader = get_dataloaders(args)

    date = datetime.today().strftime('%Y_%m_%d_%H_%M')
    final = f'Test_results_{name}'
    output_dir_o = os.path.join("results", "test", final, "originals")
    os.makedirs(output_dir_o, exist_ok=True)
    output_dir_m = os.path.join("results", "test", final, "measurements_backprojection")
    os.makedirs(output_dir_m, exist_ok=True)
    if args.awgn_snr is not None:
        output_dir_m_noisy = os.path.join("results", "test", final, "measurements_noise")
        os.makedirs(output_dir_m_noisy, exist_ok=True)
    output_dir_r = os.path.join("results", "test", final, "reconstructions")
    os.makedirs(output_dir_r, exist_ok=True)
    output_dir_f = os.path.join("results", "test", final, "features")
    os.makedirs(output_dir_f, exist_ok=True)

    output_dir_csv = os.path.join("results", "test", final)
    os.makedirs(output_dir_csv, exist_ok=True)

    with torch.no_grad():
        decoder.eval()
        
        psnr_meter_test = AverageMeter()
        ssim_meter_test = AverageMeter()
        
        # Counter for saving samples (to avoid saving too many)
        saved_samples = 0
        max_samples_to_save = args.save_batches  # Use command line argument

        data_loop_test = tqdm(enumerate(testloader), total=len(testloader), colour='BLUE')

        for data in data_loop_test:

            if args.dataset == "mnist":
                batch_idx, images = data
                images = images[0]
            elif args.dataset == "celeba":
                batch_idx, batch = data
                images = batch[0]
            images = images.to(args.device)
            
            y_s = physics_s(images).to(args.device)

            y_s_original = y_s.clone()
            if args.awgn_snr is not None:
                y_s = AWGN(y_s).to(args.device)

            x_0 = physics_s.A_adjoint(y_s)
            output, features = decoder(x_0, return_features=True)
            psnr_value = psnr(output, images)
            ssim_v = ssim_value(output, images)


            # Save limited visual diagnostics
            if saved_samples < max_samples_to_save:
                first_layer_features = features[0]  # x1 is the first element in features list

                for i in range(first_layer_features.size(0)):
                    # Save feature map (first channel of first layer)
                    feature_map = first_layer_features[i].cpu()
                    channel_data = feature_map[0].numpy()
                    channel_normalized = ((channel_data - channel_data.min()) /
                                         (channel_data.max() - channel_data.min() + 1e-8) * 255).astype('uint8')
                    feature_img_filename = os.path.join(output_dir_f, f"first_layer_ch0_batch_{batch_idx}_sample_{i}.png")
                    plt.imsave(feature_img_filename, channel_normalized, format='png')

                    # Save x_0 (backprojection)
                    x0_img = x_0[i].detach().cpu().numpy()
                    x0_img = np.transpose(x0_img, (1, 2, 0)) if x0_img.shape[0] > 1 else x0_img[0]
                    x0_img = ((x0_img - x0_img.min()) / (x0_img.max() - x0_img.min() + 1e-8) * 255).astype('uint8')
                    plt.imsave(os.path.join(output_dir_m, f"x0_batch_{batch_idx}_sample_{i}.png"), x0_img, format='png')

                    # Save output (reconstruction)
                    out_img = output[i].detach().cpu().numpy()
                    out_img = np.transpose(out_img, (1, 2, 0)) if out_img.shape[0] > 1 else out_img[0]
                    out_img = ((out_img - out_img.min()) / (out_img.max() - out_img.min() + 1e-8) * 255).astype('uint8')
                    plt.imsave(os.path.join(output_dir_r, f"recon_batch_{batch_idx}_sample_{i}.png"), out_img, format='png')

                    # Save original image
                    orig_img = images[i].detach().cpu().numpy()
                    orig_img = np.transpose(orig_img, (1, 2, 0)) if orig_img.shape[0] > 1 else orig_img[0]
                    orig_img = ((orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8) * 255).astype('uint8')
                    plt.imsave(os.path.join(output_dir_o, f"orig_batch_{batch_idx}_sample_{i}.png"), orig_img, format='png')

                saved_samples += 1
                if saved_samples == 1:
                    print(f"Saving first channel of first decoder layer, x_0, reconstructions and originals for the first {max_samples_to_save} batches...")

            psnr_meter_test.update(psnr_value.item(), images.size(0))
            ssim_meter_test.update(ssim_v.item(), images.size(0))

            # data_loop_test.set_description(f'Test iter [{iter + 1}/{args.iters}]')
            data_loop_test.set_postfix(psnr=psnr_meter_test.avg,
                                       ssim=ssim_meter_test.avg)

            data_loop_test.set_postfix(psnr=psnr_meter_test.avg,
                                       ssim=ssim_meter_test.avg)
        
        save_psnr_to_csv(args, psnr_meter_test.avg, ssim_meter_test.avg, output_dir_csv, saved_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--n', type=int, default=32, help='Dimension of image')
    parser.add_argument('--batch_size', type=int, default=50, help='Number of batches')
    parser.add_argument("--awgn_snr", type=float, default=None, help="Additive White Gaussian Noise SNR")
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset')
    parser.add_argument('--device', type=int, default=0, help='Device') # n < 0 == cpu
    parser.add_argument('--cr', type=float, default=0.9, help='Measurements Ratio')
    parser.add_argument('--c', type=int, default=1, help='Number of channels')
    parser.add_argument('--physics', type=str, default='cs', help='Physics')
    parser.add_argument('--save_batches', type=int, default=10, help='Number of batches to save features for')
    args = parser.parse_args()
    print(args)
    main(args)
