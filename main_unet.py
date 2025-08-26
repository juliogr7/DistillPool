import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim
from data_loader import get_dataloaders
from algo_functions import AverageMeter, AddGaussianNoiseSNR, get_physics
from model_unet import UNet

import argparse
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm
import wandb as wandb
import deepinv as dinv
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)

def get_physics(args):

    if args.physics == "cs":  # single-pixel
        ordering = "sequency"
        if args.cr == 1:
            physics_arr = []
            for cr in args.crs_expert:
                m = int(args.n**2 * cr)
                physics = dinv.physics.SinglePixelCamera(
                    m=m,
                    img_shape=(1, args.n, args.n),
                    # noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
                    device=args.device,
                    ordering=ordering,
                )
                physics_arr.append(physics)
        else:
            m = int(args.n * args.n * args.cr) # number of measurements
            # noise_level_img = 0.0

            physics = dinv.physics.SinglePixelCamera(
                m=m,
                img_shape=(1, args.n, args.n),
                # noise_model=dinv.physics.GaussianNoise(sigma=noise_level_img),
                device=args.device,
                ordering=ordering,
            )

    if args.cr == 1:
        return physics_arr
    else:
        return physics


def main(args):

    # Escoger Device (GPU o CPU)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    if args.device >= 0:
        args.device = f"cuda:{args.device}"
    else:
        args.device = "cpu"

    expert_train = args.cr == 1
    
    if expert_train:
        crs_str = "_".join(str(num).replace('.', '') for num in args.crs_expert)

    awgn_str = f"_awgn_{str(args.awgn_snr).split('.')[0]}" if args.awgn_snr is not None else ""
    config = (
        f"{'EXPERT_' if expert_train else ''}UNET"
        f"_cr_{crs_str if expert_train else str(args.cr).replace('.', '')}_scales_{args.scales}_"
        f"{awgn_str}"
        f"dataset_{args.dataset}_"
        f"{datetime.today().strftime('%Y_%m_%d_%H_%M')}"
    )


    # WandB Logging
    entity_name = "your_entity"  # WandB entity name
    wandb.login(key="Put_your_key_here")
    wandb.Api()
    wandb.init(project="name", entity=entity_name, name=config, config=args)

    # Load pickle file for algo params
    trainloader, testloader = get_dataloaders(args)

    # UNet model as decoder
    decoder = UNet(
        in_channels=args.c,
        out_channels=args.c,
        scales=args.scales,  # Network deepness
        residual=True,  # Residual conections
        circular_padding=True,
        cat=True,  # Skip connections
        bias=True,  # This should be False when batch_norm is True
        batch_norm=args.batch_norm,  # True -> helps training time and performance
    ).to(args.device)

    if args.train_again:
        os.environ["PRETRAINED_MODEL_PATH"] = "UNET_cr_02_scales_2_2025_04_22_14_39.pth"
        filename_model = os.getenv("PRETRAINED_MODEL_PATH", None)
    
    if args.train_again:
        dir_model = os.path.join(
            "results",
            "trained_models",
            filename_model
        )
        decoder.load_weights(dir_model, device = args.device) # Load pretrained student weights

        print(f"Training pretrained decoder model ----> {filename_model}")


    # count and print the number of parameters in thousands
    num_params = sum(p.numel() for p in decoder.parameters()) / 1e3
    print(f"Number of parameters: {num_params:.2f}k")

    if expert_train:
        physics_arr = get_physics(args)
        idx_options = np.arange(len(physics_arr))
        print('Training with different CRs!!')
    else:
        physics = get_physics(args)

    undersampling_rate = physics.mask.mean()
    # print(f"Image Size: {x.shape}")
    m = int(args.n * args.n * args.cr)
    print(f"Expected SPC measurement: {m}")
    print(f"Real SPC measurement: {physics.mask.sum()}")
    print(f"Expected CR: {args.cr}")
    print(f"Real undersampling rate: {undersampling_rate:.2f}")

    # print("Real compressing ratio: ", torch.mean(physics.mask.mean()).item())

    psnr = PeakSignalNoiseRatio(data_range=1.0).to(args.device)

    if args.awgn_snr is not None:
        AWGN = AddGaussianNoiseSNR(snr = args.awgn_snr)

    # Optimizer to use during training
    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adam":  # Best
        optimizer = torch.optim.Adam(
            decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    output_dir = os.path.join("results", "trained_models")
    os.makedirs(output_dir, exist_ok=True)

    

    # Training loop
    for iter in range(args.iters):
        loss_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        decoder.train()
        data_loop_train = tqdm(
            enumerate(trainloader), total=len(trainloader), colour="cyan"
        )

        for data in data_loop_train:

            if args.dataset == "mnist":
                _, images = data
                images = images[0]
            elif args.dataset == "celeba":
                _, batch = data
                images = batch[0]
            images = images.to(args.device)

            if expert_train:
                j = np.random.choice(idx_options)
                physics = physics_arr[j]
            # Encoder part
            y_s = physics(images).to(args.device)

            # AWGN
            if args.awgn_snr is not None:
                y_s = AWGN(y_s).to(args.device)

            # Aprox to x (transposing y) because y = Hx --> x_o = H^T y
            x_0 = physics.A_adjoint(y_s).to(args.device)

            # Direct forward
            output = decoder(x_0)

            # Loss function
            if args.loss == "l2":
                loss = torch.norm(output - images).to(args.device)
            elif args.loss == "mse":
                loss = torch.nn.functional.mse_loss(output, images).to(args.device)
            psnr_value = psnr(output, images)
            ssim_value = ssim(output, images, data_range=1.0)

            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            loss_meter.update(loss.item(), images.size(0))
            psnr_meter.update(psnr_value.item(), images.size(0))
            ssim_meter.update(ssim_value.item(), images.size(0))

            data_loop_train.set_description(f"Train iter [{iter + 1}/{args.iters}]")
            data_loop_train.set_postfix(loss=loss_meter.avg, 
                                        psnr=psnr_meter.avg, 
                                        cr=args.crs_expert[j] if expert_train else args.cr)

        # Testing
        with torch.no_grad():
            decoder.eval()

            psnr_meter_test = AverageMeter()
            ssim_meter_test = AverageMeter()
            data_loop_test = tqdm(
                enumerate(testloader), total=len(testloader), colour="green"
            )

            if expert_train:
                j = np.random.choice(idx_options)
                physics = physics_arr[j]
            
            for data in data_loop_test:

                if args.dataset == "mnist":
                    _, images = data
                    images = images[0]
                elif args.dataset == "celeba":
                    _, batch = data
                    images = batch[0]
                images = images.to(args.device)

                images = images.to(args.device)

                y_s = physics(images).to(args.device)

                # AWGN
                if args.awgn_snr is not None:
                    y_s = AWGN(y_s).to(args.device)

                x_0 = physics.A_adjoint(y_s)
                output = decoder(x_0)

                psnr_value = psnr(output, images)
                ssim_value = ssim(output, images, data_range=1.0)
                psnr_meter_test.update(psnr_value.item(), images.size(0))
                ssim_meter_test.update(ssim_value.item(), images.size(0))

                data_loop_test.set_description(f"Test iter [{iter + 1}/{args.iters}]")
                postfix_dict = {
                    "psnr": psnr_meter_test.avg,
                }
                if expert_train:
                    postfix_dict["cr"] = args.crs_expert[j]

                data_loop_test.set_postfix(**postfix_dict)

        # Saving metrics in WandB
        wandb.log(
            {
                "Train loss": loss_meter.avg,
                "Train PSNR": psnr_meter.avg,
                "Test PSNR": psnr_meter_test.avg,
                "Train SSIM": ssim_meter.avg,
                "Test SSIM": ssim_meter_test.avg,
            }
        )

    print("Training complete!")

    save_path = os.path.join(output_dir, config + ".pth")

    # Saving the model
    save_path = os.path.join(output_dir, config + ".pth")
    torch.save(decoder.state_dict(), save_path)
    print(f"Student model weights saved at {config}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--algo", type=str, default="UNET", help="Algorithm to use")
    parser.add_argument(
        "--batch_norm", type=bool, default=False, help="Batch normalization for UNet"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate")
    parser.add_argument("--loss", 
                        type=str, 
                        default="mse", 
                        help="Loss function to use")
    parser.add_argument("--scales", 
                        type=int, 
                        default=2, 
                        help="UNet scales")
    parser.add_argument("--n", type=int, default=32, help="Dimension of image")
    parser.add_argument(
        "--cr", type=float, default=0.1, help="Measurements compress ratio"
    )
    parser.add_argument(
        "--crs_expert", nargs="+", 
        type=float,
        default = None, # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
        help="Lista de compress ratios para el expert")
    parser.add_argument(
        "--iters", type=int, default=50, help="Number of iterations"
    )
    parser.add_argument("--awgn_snr", type=float, default=None, help="Additive White Gaussian Noise SNR")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset"
    )
    parser.add_argument("--train_again",
                        type=bool, 
                        default=False, 
                        help="True --> Retrain student model by loading pretrained student weights")
    parser.add_argument("--device", type=int, default=0, help="Device")  # n < 0 == cpu
    parser.add_argument("--c", type=int, default=1, help="Number of channels")
    parser.add_argument(
        "--physics", type=str, default="cs", help="Physics"
    )

    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer: adam, adamw or sgd"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="Weight decay coefficient"
    )  # 1e-2

    args = parser.parse_args()
    print(args)
    main(args)