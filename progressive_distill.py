import os
import torch
import numpy as np
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from data_loader import get_dataloaders
from algo_functions import AverageMeter, get_physics, AddGaussianNoiseSNR
from model_unet_dist import UNet # Import modified UNet
from model_hybrid_unet import HybridUNet

import argparse
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm
import wandb
from datetime import datetime

torch.manual_seed(42)
np.random.seed(42)


def matrix_combinatorial(A, gamma = 1e-6):
    """Compute combinatorial similarity matrix for feature tensor A with RBF kernel (per channel)."""
    total = 0.0

    for i in range(A.shape[1]):
        A = A[:, i]
        A = A.permute(1, 0, 2, 3)
        A_flat = A.reshape(A.shape[3], -1)

        matrix_comb = torch.exp(-gamma * torch.cdist(A_flat, A_flat, p=2) )# gamma = 1 / 2*(sigma**2)
        total +=  matrix_comb

    return total

def matrix_combinatorial_generalized(student_features, teacher_features, gamma):
    """RBF Frobenius distance between student and teacher combinatorial feature descriptors."""

    student_features = torch.permute(torch.stack(student_features), [1, 0, 2, 3, 4])
    teacher_features = torch.permute(torch.stack(teacher_features), [1, 0, 2, 3, 4])

    cc_s = matrix_combinatorial(student_features, gamma)
    cc_t = matrix_combinatorial(teacher_features, gamma)

    return torch.norm(cc_s - cc_t, p="fro")

def distillation_loss(teacher_feats, student_feats, teacher_output = None, student_output = None, mode="cs", type_dist="all", gamma = None):
    """
    Compute distillation loss between teacher and student features
    teacher_feats: List of teacher features, list of tensors. eg. [feat1, feat2, ..., featN] N = number of scales
    student_feats: List of student features, list of tensors. eg. [feat1, feat2, ..., featM] M = number of scales
    mode: 'mse' or 'cs', 'cs' for cosine similarity
    type_dist: 'latent' or 'all', 'latent' for distillation only in latent space, 'all' for distillation in all scales
    """

    if type_dist == "latent":

        teacher_feats = [teacher_feats[1].clone()]
        student_feats = [student_feats[1].clone()]

        if args.teacher_scales != args.student_scales:# If the scales are different, we need to adjust the student features
            factor = args.teacher_scales - args.student_scales
            
            Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            conv_part = nn.Conv2d(in_channels=student_feats[0].shape[1], 
                                    out_channels=teacher_feats[0].shape[1], 
                                    kernel_size=1, stride=1, bias=True).to(args.device)
            for _ in range(factor):
                student_feats[0] = Maxpool(student_feats[0])

            if student_feats[0].shape[1] != teacher_feats[0].shape[1]: # I think this is always True
                student_feats[0] = conv_part(student_feats[0])
    elif type_dist == "first_latent":
        teacher_feats = [teacher_feats[0], teacher_feats[1]]
        student_feats = [student_feats[0], student_feats[1]]
    
    elif type_dist == "first":
        teacher_feats = [teacher_feats[0]]
        student_feats = [student_feats[0]]

    elif type_dist == "output":
        teacher_feats = [teacher_output]
        student_feats = [student_output]

    elif type_dist == "output_latent":
        teacher_feats = [teacher_output, teacher_feats[1]]
        student_feats = [student_output, student_feats[1]]

    elif type_dist == "encoder":
        teacher_feats = [teacher_feats[0]]
        student_feats = [student_feats[0]]
    
    elif type_dist == "encoder_latent":
        teacher_feats = teacher_feats[0:2].copy()
        student_feats = student_feats[0:2].copy()

    elif type_dist == "latent_decoder":
        teacher_feats = teacher_feats[1:].copy()
        student_feats = student_feats[1:].copy()

    elif type_dist == "output_latent_decoder":
        teacher_feats = [teacher_output] + teacher_feats[1:].copy()
        student_feats = [student_output] + student_feats[1:].copy()

    elif type_dist == "output_all":
        teacher_feats = [teacher_output] + teacher_feats.copy()
        student_feats = [student_output] + student_feats.copy()

    elif type_dist == "output_first_latent":
        teacher_feats = [teacher_output]+ [teacher_feats[0], teacher_feats[1]]
        student_feats = [student_output] + [student_feats[0], student_feats[1]]
    elif type_dist == "output_first":
        teacher_feats = [teacher_output, teacher_feats[0]]
        student_feats = [student_output, student_feats[0]]

    fn_loss = None
    if mode == "mse":
        fn_loss = torch.nn.functional.mse_loss
    elif mode == "mae" or mode == "l1":
        fn_loss = torch.nn.functional.l1_loss
    elif mode == "l2":
        fn_loss = torch.norm
    elif mode == "cs":
        fn_loss = torch.nn.functional.cosine_similarity
    elif mode == "linf":
        fn_loss = torch.max
    elif mode == "comb_matrix":
        fn_loss = matrix_combinatorial_generalized
    else:
        raise ValueError(f"Invalid mode: {mode}")

    loss = 0.0

    for i, (t_feat, s_feat) in enumerate(zip(teacher_feats, student_feats)):

        if mode != "comb_matrix":
            s_feat = s_feat.view(s_feat.size(0), -1).to(args.device)
            t_feat = t_feat.view(t_feat.size(0), -1).to(args.device)

        if mode == "l2":
            curr_loss = fn_loss(s_feat - t_feat)
        elif mode == "mae":
            curr_loss = fn_loss(s_feat, t_feat, reduction = 'mean')
        elif mode == "linf":
            curr_loss = fn_loss(torch.abs(s_feat - t_feat))
        elif mode == "comb_matrix":
            curr_loss = fn_loss([s_feat], [t_feat], gamma)
        elif mode == "cs":
            curr_loss = fn_loss(s_feat, t_feat)
            curr_loss = (
                1 - curr_loss
            )  # Cosine similarity is in [-1, 1], so we need to invert it
            curr_loss = curr_loss.mean().to(args.device)  # Average over batch
        elif mode == "mse":
            curr_loss = fn_loss(s_feat, t_feat)
        else:
            curr_loss = fn_loss(s_feat, t_feat)

        loss += curr_loss

    return loss


def main(args):
    """Main training loop for progressive / hybrid distillation."""
    # Set up device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # Config the cuda ID to use
    if args.device >= 0:
        args.device = f"cuda:{args.device}"
    else:  # If there is no cuda, use cpu
        args.device = "cpu"

    print(args)

    date = datetime.today().strftime('%Y_%m_%d_%H_%M')

    awgn_str = f"_awgn_{str(args.awgn_snr).split('.')[0]}" if args.awgn_snr is not None else ""

    type_dist = ["latent", "all", "first_latent", "output", "output_latent", "encoder",
                  "encoder_latent", "latent_decoder", "output_latent_decoder","output_all", "first", "output_first_latent", "output_first"]
    type_dist = type_dist[args.type_dist]
    awgn_str = f"_awgn_{str(args.awgn_snr).split('.')[0]}" if args.awgn_snr is not None else ""

    if args.distill_student_again:
        os.environ["PRETRAINED_STUDENT_PATH"] = "student_weights_crs_01_crt_03_scales_2_alpha_100_beta_1_type_output_latent_decoder_loss_dist_mse_loss_gt_mse_iters_200_2025_04_22_06_50.pth"
        filename_student = os.getenv("PRETRAINED_STUDENT_PATH", None)
        filename_student_str = filename_student.split(".")[0]
        predate = filename_student_str.split('2025_')[1].split('.pth')[0]
        config = (f"Hybrid_Distillation_UNET_crs_{str(args.cr_student).replace('.', '')}_crt_{str(args.cr_teacher).replace('.', '')}" +
                f"_scales_{args.student_scales}_type_{type_dist}_pre_{predate}"
                f"{awgn_str}"
                f"date_{date}")
    else:
        config = (f"Distillation_Progressive_UNET_crs_{str(args.cr_student).replace('.', '')}_crt_{str(args.cr_teacher).replace('.', '')}" +
                f"_scales_{args.student_scales}_type_{type_dist}_{date}")
        
    entity_name = "laumila-universidad-industrial-de-santander"

    # WandB Logging
    entity_name = "your_entity"  # WandB entity name

    # WandB Logging
    wandb.login(key="Put_your_key_here")
    wandb.Api()
    wandb.init(project="name", entity=entity_name, name=config, config=args)
    
    expert_t = args.cr_teacher == 1

    # Load Data
    trainloader, testloader = get_dataloaders(args)
    
    if expert_t:
        physics_s, physics_t_arr = get_physics(args)
        idx_options_t = np.arange(len(physics_t_arr))
    else:
        physics_s, physics_t = get_physics(args)

    student = UNet(
        in_channels=args.c,
        out_channels=args.c,
        scales=args.student_scales,  # Smaller student model
        residual=True,
        circular_padding=True,
        cat=True,
        bias=True,
        batch_norm=args.batch_norm,
    ).to(args.device)

    # Define Teacher and Student Models
    teacher = UNet(
        in_channels=args.c,
        out_channels=args.c,
        scales=args.teacher_scales,  # Full-sized teacher model
        residual=True,
        circular_padding=True,
        cat=True,
        bias=True,
        batch_norm=args.batch_norm,
    ).to(args.device)
    
    filename = f"UNET_cr_{str(args.cr_teacher).replace('.','')}.pth"

    folder_teacher = "teachers_notnoisy" if args.dataset == "mnist" else "teachers_celeba"

    dir_teacher = os.path.join(
            "results",
            "trained_models",
            folder_teacher,
            filename
        )
    
    teacher.load_weights(dir_teacher, device = args.device) # The teacher is automatically set to eval mode
    print(f"Pretrained teacher model ----> {filename}")

    if args.distill_student_again:

        filename_student = "UNET_cr_01_scales_2_dataset_celeba_2025_05_02_15_08.pth"

        dir_student = os.path.join(
            "results",
            "trained_models",
            filename_student
        )
        student.load_weights(dir_student, device = args.device) # Load pretrained student weights
        hybrid_model = HybridUNet(student, student).to(args.device)

        print(f"Distilling pretrained student model ----> {filename_student}")
    else:
        second_model = UNet(
            in_channels=args.c,
            out_channels=args.c,
            scales=args.teacher_scales,  # Full-sized teacher model
            residual=True,
            circular_padding=True,
            cat=True,
            bias=True,
            batch_norm=args.batch_norm,
        ).to(args.device)

        filename_second = "student_weights_crs_03_crt_08_scales_2_alpha_2000_beta_1_type_output_latent_loss_dist_mse_loss_gt_mse_iters_100__dataset_mnist_2025_05_03_19_49.pth"
        dir_second = os.path.join(
                "results",
                "trained_models",
                #"teachers_notnoisy",
                filename_second
            )
        
        second_model.load_weights(dir_second, device = args.device) # The teacher is automatically set to eval mode
        print(f"Full weight initialization from ----> {filename_second}")

        hybrid_model = HybridUNet(second_model, second_model).to(args.device)

    hybrid_model.train()

    # Compare scales of teacher and student
    print(f"Hybrid scales: {hybrid_model.compact}")

    # Count and print number of parameters
    num_params_student = sum(p.numel() for p in hybrid_model.parameters()) / 1e3
    print(f"Hybrid Parameters: {num_params_student:.2f}k")

    # Loss and Optimizer
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, hybrid_model.parameters()),
        # hybrid_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    print("Parámetros que serán optimizados (requires_grad=True):")
    for name, param in hybrid_model.named_parameters():
        if param.requires_grad:
            print(f"  ✅ {name}")

    if args.awgn_snr is not None:
        AWGN = AddGaussianNoiseSNR(snr = args.awgn_snr)

    output_dir = os.path.join("results", "trained_models")
    os.makedirs(output_dir, exist_ok=True)

    alpha = args.alpha

    for iter in range(args.iters):
        loss_meter = AverageMeter()
        loss_gt_meter = AverageMeter()
        psnr_meter = AverageMeter()
        ssim_meter = AverageMeter()

        hybrid_model.train() #Student setted to train mode
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
            # Apply physics transformation
            y_s = physics_s(images).to(args.device)
            y_t = physics_t(images).to(args.device)

            # AWGN
            if args.awgn_snr is not None:
                y_s = AWGN(y_s).to(args.device)

            x_0_s = physics_s.A_adjoint(y_s).to(args.device)
            x_0_t = physics_t.A_adjoint(y_t).to(args.device)

            # student_output = hybrid_model(x_0_s)
            student_output, student_feats = hybrid_model(x_0_s, return_features=True)

            with torch.no_grad():
                teacher_output, teacher_feats = teacher(
                    x_0_t, return_features=True
                )
            
            # Compute losses
            target = images

            if args.loss_gt == "l2":
                loss_gt = torch.norm(student_output - target).to(args.device)
            elif args.loss_gt == "mse":
                loss_gt = torch.nn.functional.mse_loss(student_output, target).to(args.device)
            elif args.loss_gt == "cs":
                loss_gt = (1 - torch.nn.functional.cosine_similarity(student_output, target)).mean().to(args.device)

            distill_loss = distillation_loss(
                teacher_feats,
                student_feats, 
                teacher_output if args.type_dist >= 3 else None,
                student_output if args.type_dist >= 3 else None,
                mode=args.loss_distill, type_dist=type_dist,
                gamma=args.gamma if args.loss_distill == "comb_matrix" else None,
            ).to(args.device)
            
            total_loss = loss_gt + (alpha * distill_loss)

            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Metrics
            psnr_value = psnr(student_output, images)
            ssim_value = ssim(student_output, images, data_range=1.0)
            loss_gt_meter.update(loss_gt.item(), images.size(0))
            loss_meter.update(total_loss.item(), images.size(0))
            psnr_meter.update(psnr_value.item(), images.size(0))
            ssim_meter.update(ssim_value.item(), images.size(0))

            data_loop_train.set_description(f"Train iter [{iter + 1}/{args.iters}]")
            postfix_dict = {
                "loss": loss_meter.avg,
                "loss_gt": loss_gt_meter.avg,
                "psnr": psnr_meter.avg,
            }

            data_loop_train.set_postfix(**postfix_dict)


        # Validation Loop
        with torch.no_grad():
            hybrid_model.eval()
            psnr_meter_test = AverageMeter()
            ssim_meter_test = AverageMeter()
            data_loop_test = tqdm(
                enumerate(testloader), total=len(testloader), colour="magenta"
            )

            for data in data_loop_test:

                if args.dataset == "mnist":
                    _, images = data
                    images = images[0]
                elif args.dataset == "celeba":
                    _, batch = data
                    images = batch[0]
                images = images.to(args.device)


                y_s = physics_s(images).to(args.device)

                # AWGN
                if args.awgn_snr is not None:
                    y_s = AWGN(y_s).to(args.device)
                
                x_0 = physics_s.A_adjoint(y_s)

                output = hybrid_model(x_0)

                psnr_value = psnr(output, images)
                ssim_value = ssim(output, images, data_range=1.0)
                psnr_meter_test.update(psnr_value.item(), images.size(0))
                ssim_meter_test.update(ssim_value.item(), images.size(0))

                data_loop_test.set_description(f"Test iter [{iter + 1}/{args.iters}]")
                data_loop_test.set_postfix(psnr=psnr_meter_test.avg)

        # Logging to WandB
        wandb.log(
            {
                "Student/Train Loss": loss_meter.avg,
                "Student/GT Loss": loss_gt_meter.avg,
                "Student/Train PSNR": psnr_meter.avg,
                "Student/Test PSNR": psnr_meter_test.avg,
                "Student/Train SSIM": ssim_meter.avg,
                "Student/Test SSIM": ssim_meter_test.avg,
                }
            )

    # Save Student Model
    filename = (f"{date}_Progressive_student_weights_crs_{str(args.cr_student).replace('.', '')}_crt_{str(args.cr_teacher).replace('.', '')}"
                f"_scales_{args.student_scales}_"
                f"loss_gt_{args.loss_gt}_iters_{args.iters}"
                f"{awgn_str}"
                + ("redistilled_" if args.distill_student_again else "") + ".pth")
    
    save_path = os.path.join(output_dir, filename)
    torch.save(hybrid_model.state_dict(), save_path)
    print(f"Student model weights saved at {filename}")

    print("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distillation Training for UNet")

    parser.add_argument("--algo", type=str, default="UNET", help="Algorithm")
    parser.add_argument(
        "--batch_norm", type=bool, default=False, help="Batch normalization"
    )
    parser.add_argument(
        "--cr_teacher",
        type=float,
        default=0.8,
        help="Teacher compress ratio",
    )
    parser.add_argument(
        "--cr_student", 
        type=float, 
        default=0.1, 
        help="Student compress ratio"
    )
    parser.add_argument(
        "--crs_expert_teacher", nargs="+", 
        type=float,
        default = None, #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        help="Lista de compress ratios para el expert teacher")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate")
    parser.add_argument("--n", type=int, default=32, help="Image size")
    parser.add_argument("--distill_student_again",
                        type=bool, 
                        default=False, 
                        help="True --> Retrain student model by loading pretrained student weights")
    parser.add_argument(
        "--type_dist",
        type=int,
        default=4,
        help="0 for latent distillation, 1 for full distillation, 2 for first layer and latent distillation",
    )
    parser.add_argument("--alpha", type=float, default=0, help="Weight of distillation loss. Between [0,1]")
    parser.add_argument("--awgn_snr", type=float, default=None, help="Additive White Gaussian Noise SNR")
    parser.add_argument("--teacher_scales", type=int, default=2, help="Teacher scales")
    parser.add_argument("--student_scales", type=int, default=2, help="Student scales")
    parser.add_argument("--iters", type=int, default=1, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset")
    parser.add_argument("--device", type=int, default=0, help="Device (GPU or CPU) ---> < 0 = cpu")
    parser.add_argument("--c", type=int, default=1, help="Number of channels")
    parser.add_argument("--physics", type=str, default="cs", help="Physics model")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer") 
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--loss_distill", 
                        type=str, 
                        default="mse", 
                        help="Loss function to use for distillation ---> mse / mae / cs / l2_cs")
    parser.add_argument("--loss_gt", 
                        type=str, 
                        default="mse", 
                        help="Ground truth loss function ---> mse / l2 ")

    args = parser.parse_args()
    
    main(args)