import deepinv as dinv
import matplotlib.pyplot as plt
import torch

def get_physics(args):
    if args.physics == "cs":  # Compressed Sensing (Single-Pixel)
        measurements = int(args.n**2 * args.cr)
        physics = dinv.physics.SinglePixelCamera(
            m=measurements, 
            img_shape=(1, args.n, args.n), 
            fast=True, 
            device=args.device
        )
    return physics

class Args:
    def __init__(self):
        self.physics = "cs"
        self.n = 32
        self.cr = 0.1
        self.device = "cpu"

args = Args()

physics = get_physics(args)

mask = physics.mask

print(mask.shape)
print("One ",torch.mean(physics.mask.mean()))

undersampling_rate = physics.mask.mean()
print(f"SPC measurement: {physics.mask.sum()}")
print(f"Undersampling rate: {undersampling_rate:.2f}")

plt.imshow(mask[0, 0, :, :],cmap='gray')
plt.show()