import torch
from algorithms import PreconditionedFISTAIteration
from data_loader import get_dataloaders
from algo_functions import get_physics, AverageMeter
from deepinv.optim.prior import PnP
from deepinv.models import DnCNN
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import optim_builder
import argparse
from pathlib import Path
from torchmetrics.image import PeakSignalNoiseRatio
from tqdm import tqdm
import os

torch.manual_seed(0)


def main(args):

    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"
    if args.device >= 0:
        args.device = f"cuda:{args.device}"
    else:
        args.device = "cpu"

    BASE_DIR = Path(".")
    ORIGINAL_DATA_DIR = BASE_DIR / "datasets"

    if args.max_iter == 20:
        str_max_iter = ""
    else:
        str_max_iter = f"maxIter_{args.max_iter}_"

    if args.stepsize_t == 0.7:
        str_stepsize_t = ""
    else:
        str_stepsize_t = f"fstepT_{args.stepsize_t}_"

    if args.stepsize_s == 0.4:
        str_stepsize_s = ""
    else:
        str_stepsize_s = f"fstepS_{args.stepsize_s}_"

    if args.physics == "fmri":
        exp_config = f"results_{args.physics}_acc_s_{args.acceleration_s}_acc_t_{args.acceleration_t}_{str_max_iter}{str_stepsize_t}{str_stepsize_s}mask_s_{args.mask_s}_mask_t_{args.mask_t}_opt_{args.optimizer}_decay_{args.weight_decay}_n_{args.n}_gamma_{args.gamma}_lr_{args.lr}_init_{args.init}_reg_convergence_{args.convergence_reg}_reg_param_{args.reg_param}_supp_loss_{args.sup_loss}"
    elif args.physics == "cs":
        exp_config = f"results_{args.physics}_cr_student_{args.cr_student}_cr_teacher_{args.cr_teacher}_{str_max_iter}{str_stepsize_t}{str_stepsize_s}opt_{args.optimizer}_decay_{args.weight_decay}_n_{args.n}_gamma_{args.gamma}_lr_{args.lr}_init_{args.init}_reg_convergence_{args.convergence_reg}_reg_param_{args.reg_param}_supp_loss_{args.sup_loss}"
    elif args.physics == "sr":
        exp_config = f"results_{args.physics}_srf_student_{args.srf_student}_srf_teacher_{args.srf_teacher}_{str_max_iter}{str_stepsize_t}{str_stepsize_s}opt_{args.optimizer}_decay_{args.weight_decay}_n_{args.n}_gamma_{args.gamma}_lr_{args.lr}_init_{args.init}_reg_convergence_{args.convergence_reg}_reg_param_{args.reg_param}_supp_loss_{args.sup_loss}"

    args.dataset_path = ORIGINAL_DATA_DIR

    physics_s, physics_t = get_physics(args)

    trainloader, testloader = get_dataloaders(args)

    path = f"results_2/{exp_config}"

    D = torch.load(path + "/D_final.pth", map_location=args.device)
    D.requires_grad = False

    def preconditioner(x):
        xr = x.reshape(x.shape[0], -1)
        xp = xr @ D
        return xp.reshape(x.shape)

    denoiser = DnCNN(
        in_channels=args.c,
        out_channels=args.c,
        pretrained="download",  # automatically downloads the pretrained weights, set to a path to use custom weights.
        device=args.device,
    )
    denoiser.eval()
    sigma = 0.1
    params_algo_teacher = {"stepsize": args.stepsize_t, "g_param": sigma}
    early_stop = False
    data_fidelity = L2()

    prior = PnP(denoiser=denoiser)

    iterator = PreconditionedFISTAIteration(P=None, a=3)
    model_no_prec = optim_builder(
        iteration=iterator,
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=args.max_iter,
        verbose=True,
        params_algo=params_algo_teacher,
        custom_init=lambda y, physics: {
            "est": (physics.A_adjoint(y), physics.A_adjoint(y))
        },
        backtracking=True,
    )
    params_algo_base = {"stepsize": args.stepsize_s, "g_param": sigma}

    model_base = optim_builder(
        iteration=iterator,
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=args.max_iter,
        verbose=True,
        params_algo=params_algo_base,
        custom_init=lambda y, physics: {
            "est": (physics.A_adjoint(y), physics.A_adjoint(y))
        },
        backtracking=True,
    )

    params_algo_student = {"stepsize": args.stepsize_s, "g_param": sigma}
    iterator_s = PreconditionedFISTAIteration(P=preconditioner, a=3)
    model_s = optim_builder(
        iteration=iterator_s,
        prior=prior,
        data_fidelity=data_fidelity,
        early_stop=early_stop,
        max_iter=args.max_iter,
        verbose=True,
        params_algo=params_algo_student,
        custom_init=lambda y, physics: {
            "est": (physics.A_adjoint(y), physics.A_adjoint(y))
        },
        backtracking=True,
    )

    if args.n_batches_train == 0:
        args.n_batches_train = len(trainloader)
    if args.n_batches_test == 0:
        args.n_batches_test = len(testloader)

    model_no_prec.eval()
    model_s.eval()
    model_base.eval()
    psnr = PeakSignalNoiseRatio().to(args.device)
    torch.autograd.set_detect_anomaly(True)

    with torch.no_grad():
        data_loop_test = tqdm(
            enumerate(testloader), total=args.n_batches_test, colour="green"
        )
        psnr_s_v_test = AverageMeter()
        psnr_t_v_test = AverageMeter()
        psnr_b_v_test = AverageMeter()
        for data in data_loop_test:
            i, images = data
            if args.dataset == "mnist":
                images = images[0]
            images = images.to(args.device)
            y_s = physics_s(images)
            y_t = physics_t(images)
            y_s = y_s.to(args.device)
            y_t = y_t.to(args.device)

            x_t = model_no_prec(y_t, physics_t, compute_metrics=False)
            x_s = model_s(y_s, physics_s, compute_metrics=False)
            x_b = model_base(y_s, physics_s, compute_metrics=False)
            psnr_s = psnr(x_s, images)
            psnr_t = psnr(x_t, images)
            psnr_b = psnr(x_b, images)

            psnr_s_v_test.update(psnr_s.item(), images.size(0))
            psnr_t_v_test.update(psnr_t.item(), images.size(0))
            psnr_b_v_test.update(psnr_b.item(), images.size(0))

            dict_metrics_test = {
                "psnr_s_test": psnr_s_v_test.avg,
                "psnr_t_test": psnr_t_v_test.avg,
                "psnr_b_test": psnr_b_v_test.avg,
            }
            data_loop_test.set_postfix(**dict_metrics_test)
            if i == args.n_batches_test:
                break

    print(
        f"psnr_b_test: {psnr_b_v_test.avg}, psnr_s_test: {psnr_s_v_test.avg}, psnr_t_test: {psnr_t_v_test.avg}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Processing")
    parser.add_argument("--algo", type=str, default="admm", help="Algorithm to use")
    parser.add_argument("--gamma", type=float, default=1, help="Gamma")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning Rate")
    parser.add_argument("--init", type=str, default="eye", help="Initialization")
    parser.add_argument("--n", type=int, default=32, help="Dimension of image")
    parser.add_argument(
        "--cr_student", type=float, default=0.4, help="Compression Ratio"
    )
    parser.add_argument(
        "--cr_teacher", type=float, default=0.8, help="Compression Ratio"
    )
    parser.add_argument("--iters", type=int, default=50, help="Number of iterations")
    parser.add_argument(
        "--n_batches_train", type=int, default=35, help="Number of batches"
    )  # 0
    parser.add_argument(
        "--n_batches_test", type=int, default=2, help="Number of test batches"
    )  # 0
    parser.add_argument("--batch_size", type=int, default=30, help="Batch size")
    parser.add_argument(
        "--convergence_reg",
        type=str,
        default="False",
        help="Convergence regularization",
    )  ### False
    parser.add_argument(
        "--reg_param",
        type=float,
        default=1e-3,
        help="Convergence Regularization parameter",
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist", help="Dataset"
    )  ########### CHECK, FOR MRI -> dataset='fmri', FOR SR, CS -> dataset='mnist' for now
    parser.add_argument(
        "--sup_loss", type=str, default="False", help="Supervised loss"
    )  ###### True
    parser.add_argument("--device", type=int, default=0, help="Device")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/fastmri_knee_singlecoil",
        help="Path to dataset",
    )
    parser.add_argument("--srf_student", type=int, default=4, help="Student SRF")
    parser.add_argument("--srf_teacher", type=int, default=1, help="Teacher SRF")
    parser.add_argument("--c", type=int, default=1, help="Number of channels")
    parser.add_argument(
        "--physics", type=str, default="cs", help="Physics"
    )  ########### CHECK, FOR MRI -> physics='mri', FOR SR, CS -> physics='cs' for now
    parser.add_argument(
        "--acceleration_s", type=int, default=4, help="Acceleration for student"
    )
    parser.add_argument(
        "--acceleration_t", type=int, default=3, help="Acceleration for teacher"
    )
    parser.add_argument(
        "--mask_s",
        type=str,
        default="gaussian",
        help="Mask for student [gaussian, uniform]",
    )
    parser.add_argument(
        "--mask_t",
        type=str,
        default="gaussian",
        help="Mask for teacher [gaussian, uniform]",
    )

    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="Optimizer: adam or adamw"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Weight decay coefficient"
    )
    parser.add_argument("--max_iter", type=int, default=20, help="Algorithm iterations")
    parser.add_argument(
        "--stepsize_t", type=float, default=0.7, help="f_step param for teacher"
    )
    parser.add_argument(
        "--stepsize_s", type=float, default=0.4, help="f_step param for student"
    )

    args = parser.parse_args()
    # print(args)
    main(args)
