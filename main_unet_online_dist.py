import os
import torch
import numpy as np
import torch.nn as nn
from torchmetrics.functional import structural_similarity_index_measure as ssim
from data_loader import get_dataloaders
from algo_functions import AverageMeter, get_physics, AddGaussianNoiseSNR
from model_unet_dist import UNet # Import modified UNet

import argparse
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm
import wandb
from datetime import datetime

import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

def matrix_combinatorial(A, gamma = 1e-6):

    total = 0.0

    for i in range(A.shape[1]):
        A = A[:, i]
        A = A.permute(1, 0, 2, 3)
        A_flat = A.reshape(A.shape[3], -1)

        matrix_comb = torch.exp(-gamma * torch.cdist(A_flat, A_flat, p=2) )# gamma = 1 / 2*(sigma**2)
        total +=  matrix_comb

    return total

def matrix_combinatorial_generalized(student_features, teacher_features, gamma):

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
        teacher_feats = teacher_feats[0].copy()
        student_feats = student_feats[0].copy()
    
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
        teacher_feats = [teacher_output] + teacher_feats[0:2].copy()
        student_feats = [student_output] + student_feats[0:2].copy()
    
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
    # Set up device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

    # Config the cuda ID to use
    if args.device >= 0:
        args.device = f"cuda:{args.device}"
    else:  # If there is no cuda, use cpu
        args.device = "cpu"

    expert_t = args.cr_teacher == 1

    if expert_t:
        crs_str = "_".join(str(num).replace('.', '') for num in args.crs_expert_teacher)
    
    type_dist = ["latent", "all", "first_latent", "output", "output_latent", "encoder",
                  "encoder_latent", "latent_decoder", "output_latent_decoder","output_all", "first", "output_first_latent", "output_first"]
    type_dist = type_dist[args.type_dist]
    print(f"Distillation type: {type_dist}")
    
    awgn_str = f"_awgn_{str(args.awgn_snr).split('.')[0]}" if args.awgn_snr is not None else ""
    date = datetime.today().strftime('%Y_%m_%d_%H_%M')
    config = (f"Distillation_online_UNET_crs_{str(args.cr_student).replace('.', '')}" +
              f"_crt_{crs_str if expert_t else str(args.cr_teacher).replace('.', '')}" +
              f"_alpha_{str(args.alpha).replace('.', '')}"
              f"{awgn_str}"
              f"_dataset_{args.dataset}_"
              f"_scales_{args.student_scales}_type_{type_dist}_{date}")
    entity_name = "laumila-universidad-industrial-de-santander"
    
    # WandB Logging
    entity_name = "your_entity"  # WandB entity name

    # WandB Logging
    wandb.login(key="Put_your_key_here")
    wandb.Api()
    wandb.init(project="name", entity=entity_name, name=config, config=args)


    # Load Data
    trainloader, testloader = get_dataloaders(args)
    
    if expert_t:
        physics_s, physics_t_arr = get_physics(args)
        idx_options_t = np.arange(len(physics_t_arr))
    else:
        physics_s, physics_t = get_physics(args)

    undersampling_rate_s = physics_s.mask.mean()
    undersampling_rate_t = physics_s.mask.mean()
    # print(f"Image Size: {x.shape}")
    m = int(args.n * args.n * args.cr_student)
    print(f"Expected SPC measurement: {m}")
    print(f"Real SPC measurement: {physics_s.mask.sum()}")
    print(f"Expected CR student: {args.cr_student}")
    print(f"Real undersampling rate student: {undersampling_rate_s:.2f}")
    print(f"Expected CR student: {args.cr_teacher}")
    print(f"Real undersampling rate student: {undersampling_rate_t:.2f}")

    # Define Teacher and Student Models
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

    # Compare scales of teacher and student
    print(f"Teacher scales: {teacher.compact}")
    print(f"Student scales: {student.compact}")

    if teacher.compact < student.compact:
        raise ValueError(
            "Student scales must be greater than or equal to teacher scales"
        )

    # Count and print number of parameters
    num_params_teacher = sum(p.numel() for p in teacher.parameters()) / 1e3
    num_params_student = sum(p.numel() for p in student.parameters()) / 1e3
    print(f"Teacher Parameters: {num_params_teacher:.2f}k")
    print(f"Student Parameters: {num_params_student:.2f}k")



    # Loss and Optimizer
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(args.device)
    optimizer_s = torch.optim.Adam(
        student.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optimizer_t = torch.optim.Adam(
        teacher.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.awgn_snr is not None:
        AWGN = AddGaussianNoiseSNR(snr = args.awgn_snr)

    output_dir = os.path.join("results", "trained_models")
    os.makedirs(output_dir, exist_ok=True)

    # Training - Distillation Loop 
    alpha = args.alpha  # Weight for distillation loss
    beta = args.beta  # Weight for MSE loss

    for iter in range(args.iters):
        loss_meter_s = AverageMeter()
        loss_gt_meter_s = AverageMeter()
        loss_distill_metter = AverageMeter()
        psnr_meter_s = AverageMeter()
        ssim_meter = AverageMeter()

        loss_meter_t = AverageMeter()
        # loss_gt_meter_t = AverageMeter()
        psnr_meter_t = AverageMeter()
        

        student.train() #Student setted to train mode
        teacher.train()
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

            if expert_t:
                j = np.random.choice(idx_options_t)
                physics_t = physics_t_arr[j]

            # Apply physics transformation
            y_s = physics_s(images).to(args.device)
            y_t = physics_t(images).to(args.device)

            # AWGN
            if args.awgn_snr is not None:
                y_s = AWGN(y_s).to(args.device)

            x_0_s = physics_s.A_adjoint(y_s).to(args.device)
            x_0_t = physics_t.A_adjoint(y_t).to(args.device)

            # Get Teacher and Student outputs
            teacher_output, teacher_feats = teacher(x_0_t, return_features=True)
            student_output, student_feats = student(x_0_s, return_features=True)

            # Compute losses
            if args.loss_gt == "l2":
                loss_gt_s = torch.norm(student_output - images).to(args.device)
                total_loss_t = torch.norm(teacher_output - images).to(args.device)
            elif args.loss_gt == "mse":
                loss_gt_s = torch.nn.functional.mse_loss(student_output, images).to(args.device)
                total_loss_t = torch.nn.functional.mse_loss(teacher_output, images).to(args.device)
            elif args.loss_gt == "cs":
                loss_gt_s = (1 - torch.nn.functional.cosine_similarity(student_output, images)).mean().to(args.device)
                total_loss_t = (1 - torch.nn.functional.cosine_similarity(teacher_output, images)).mean().to(args.device)

            distill_loss = distillation_loss(
                teacher_feats, 
                student_feats,
                teacher_output if args.type_dist >= 3 else None,
                student_output if args.type_dist >= 3 else None,
                mode=args.loss_distill, 
                type_dist=type_dist
            ).to(args.device)

            total_loss_s = beta * loss_gt_s + alpha * distill_loss

            # Optimize
            if True: # i > 0
                optimizer_s.zero_grad()
                total_loss_s.backward(retain_graph=True)
                optimizer_s.step()
                
            optimizer_t.zero_grad()
            total_loss_t.backward()
            optimizer_t.step()

            # Metrics
            psnr_value_s = psnr(student_output, images)
            ssim_value = ssim(student_output, images, data_range=1.0)
            loss_gt_meter_s.update(loss_gt_s.item(), images.size(0))
            loss_distill_metter.update(distill_loss.item(), images.size(0))
            loss_meter_s.update(total_loss_s.item(), images.size(0))
            psnr_meter_s.update(psnr_value_s.item(), images.size(0))
            ssim_meter.update(ssim_value.item(), images.size(0))

            psnr_value_t = psnr(teacher_output, images)
            loss_meter_t.update(total_loss_t.item(), images.size(0))
            psnr_meter_t.update(psnr_value_t.item(), images.size(0))

            data_loop_train.set_description(f"Train iter [{iter + 1}/{args.iters}]")
            data_loop_train.set_postfix(loss_s=loss_meter_s.avg,
                                        loss_gt_s = loss_gt_meter_s.avg,
                                        loss_distill = loss_distill_metter.avg, 
                                        loss_t = loss_meter_t.avg,
                                        psnr_s=psnr_meter_s.avg, 
                                        psnr_t=psnr_meter_t.avg,
                                        cr_t=args.crs_expert_teacher[j] if expert_t else args.cr_teacher)

        # Validation Loop
        with torch.no_grad():
            student.eval()
            teacher.eval()
            psnr_meter_test_s = AverageMeter()
            psnr_meter_test_t = AverageMeter()
            ssim_meter_test = AverageMeter()
            data_loop_test = tqdm(
                enumerate(testloader), total=len(testloader), colour="magenta"
            )

            if expert_t:
                j = np.random.choice(idx_options_t)
                physics_t = physics_t_arr[j]

            for data in data_loop_test:

                if args.dataset == "mnist":
                    _, images = data
                    images = images[0]
                elif args.dataset == "celeba":
                    _, batch = data
                    images = batch[0]
                images = images.to(args.device)
                y_s = physics_s(images).to(args.device)
                y_t = physics_t(images).to(args.device)

                # AWGN
                if args.awgn_snr is not None:
                    y_s = AWGN(y_s).to(args.device)

                x_0_s = physics_s.A_adjoint(y_s).to(args.device)
                x_0_t = physics_t.A_adjoint(y_t).to(args.device)

                student_output = student(x_0_s)
                teacher_output = teacher(x_0_t)

                psnr_value_s = psnr(student_output, images)
                ssim_value = ssim(student_output, images, data_range=1.0)
                psnr_meter_test_s.update(psnr_value_s.item(), images.size(0))
                ssim_meter_test.update(ssim_value.item(), images.size(0))

                psnr_value_t = psnr(teacher_output, images)
                psnr_meter_test_t.update(psnr_value_t.item(), images.size(0))


                data_loop_test.set_description(f"Test iter [{iter + 1}/{args.iters}]")
                postfix_dict = {
                    "psnr_s": psnr_meter_test_s.avg,
                    "psnr_t": psnr_meter_test_t.avg
                }
                if expert_t:
                    postfix_dict["cr_t"] = args.crs_expert_teacher[j]

                data_loop_test.set_postfix(**postfix_dict)


        # Logging to WandB
        wandb.log(
            {
                "Student/Train Loss": loss_meter_s.avg,
                "Student/GT Loss": loss_gt_meter_s.avg,
                "Student/Distillation Loss": loss_distill_metter.avg,
                "Student/Train PSNR": psnr_meter_s.avg,
                "Student/Test PSNR": psnr_meter_test_s.avg,
                "Student/Train SSIM": ssim_meter.avg,
                "Student/Test SSIM": ssim_meter_test.avg,
                "Teacher/Train Loss": loss_meter_t.avg,
                "Teacher/Train PSNR": psnr_meter_t.avg,
                "Teacher/Test PSNR": psnr_meter_test_t.avg,
                }
            )

    # Save Student Model
    alpha_str = str(args.alpha).replace('.', '')
    beta_str = str(args.beta).replace('.', '')
    filename = (f"student_weights_crs_{str(args.cr_student).replace('.', '')}_"
                f"crt_{crs_str if expert_t else str(args.cr_teacher).replace('.', '')}_"
                f"scales_{args.student_scales}_alpha_{alpha_str}_beta_{beta_str}_type_{type_dist}_"
                f"loss_dist_{args.loss_distill}_loss_gt_{args.loss_gt}_iters_{args.iters}_"
                f"{awgn_str}"
                f"_dataset_{args.dataset}_" + f"{date}.pth")
    
    save_path = os.path.join(output_dir, filename)
    torch.save(student.state_dict(), save_path)
    print(f"Student model weights saved at {filename}")

    filename = (
        f"{'EXPERT_' if expert_t else ''}UNET"
        f"_cr_{crs_str if expert_t else str(args.cr_teacher).replace('.', '')}_scales_{args.teacher_scales}_"
        f"{datetime.today().strftime('%Y_%m_%d_%H_%M')}"
    )
    
    save_path = os.path.join(output_dir, filename)
    torch.save(teacher.state_dict(), save_path)
    print(f"Teacher model weights saved at {filename}")


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
        default=0.4,
        help="Teacher compress ratio",
    )
    parser.add_argument(
        "--cr_student", 
        type=float, 
        default=0.2, 
        help="Student compress ratio"
    )
    parser.add_argument(
        "--crs_expert_teacher", nargs="+", 
        type=float,
        default = None, # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
        help="Lista de compress ratios para el expert teacher")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate")
    parser.add_argument("--n", type=int, default=32, help="Image size")
    parser.add_argument(
        "--type_dist",
        type=int,
        default=4,
        help="0 for latent distillation, 1 for full distillation, 2 for first layer and latent distillation",
    )
    parser.add_argument("--alpha", type=float, default=200, help="Weight of distillation loss. Between [0,1]")
    parser.add_argument("--beta", type=float, default=1, help="Weight of MSE Loss. Between [0,1]")
    parser.add_argument("--gamma", type=float, default=1e-6, help=" 1 / 2*(sigma**2), sigma --> variance")
    parser.add_argument("--awgn_snr", type=float, default=None, help="Additive White Gaussian Noise SNR")
    parser.add_argument("--teacher_scales", type=int, default=2, help="Teacher scales")
    parser.add_argument("--student_scales", type=int, default=2, help="Student scales")
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
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
    print(args)
    main(args)
