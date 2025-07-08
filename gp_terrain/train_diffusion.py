import os
import argparse
from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

from algo.vae import VAE
from algo.flow_matching import FlowMatching
from dataset import build_dataset

print("Train script is starting...")


def generate_output(algo, vae, batch_size, step_size):
    with torch.no_grad():
        z_generated = algo.sample(batch_size, step_size)
        x_generated = vae.decode(z_generated)

    x_generated = x_generated.detach().cpu().numpy()
    x_generated = x_generated.transpose((0,3,2,1))
    x_generated = x_generated.reshape((-1, x_generated.shape[2], x_generated.shape[3]))

    return x_generated

def train(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)

    ckpt_dir = os.path.join(cfg.save_dir, "ckpts")
    output_dir = os.path.join(cfg.save_dir, "outputs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("building network ...")
    vae = VAE()
    vae_ckpt = torch.load(
        os.path.join(cfg.vae_dir, "ckpts/vae.pt"),
        weights_only=True,
    )
    vae.load_state_dict(vae_ckpt)
    vae.to(cfg.device)

    cfg.tokenize_scale = vae.spatial_compression
    cfg.tokenize_dim = vae.latent_dim

    algo = FlowMatching(cfg)

    print("building dataset ...")
    train_dataset, __ = build_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    # generate pre-trained model outputs
    x_concat = generate_output(algo, vae, cfg.eval_batch_size, cfg.eval_dt)
    output_path = os.path.join(output_dir, f"update_0_output.png")
    plt.imshow(x_concat, cmap='terrain', vmin=-1.5, vmax=1.5)
    plt.savefig(output_path)
    plt.close()

    update_count = 0
    loss_sum = 0.0
    start_time = time.time()

    print("training ...")
    while update_count < cfg.num_updates:
        for __, x_batch in enumerate(train_loader):
            with torch.no_grad():
                # print(f"x_batch shape: {x_batch.shape}")
                if x_batch.shape[2] == 360:
                    padding = (0, 0, 4, 4)  # (left, right, top, bottom)
                    x_batch = torch.nn.functional.pad(x_batch, padding, mode='constant', value=0)
                # print(f"x_batch shape after padding: {x_batch.shape}")
                # print(f"x_data type: {x_batch.dtype}")
                z_batch = vae.encode(x_batch)
                # Add padding to make the size 360 -> 368 for batch
                

            loss = algo.update(z_batch)
            loss_sum += loss

            update_count += 1

            if cfg.debug and (update_count % cfg.log_every == 0):
                print(
                    f"Update count: {update_count} | Loss: {loss_sum / cfg.log_every:.6f} | Time: {time.time()-start_time:.6f}"
                )
                start_time = time.time()
                loss_sum = 0.0

            if update_count % cfg.eval_every == 0:
                # generate outputs
                x_concat = generate_output(algo, vae, cfg.eval_batch_size, cfg.eval_dt)
                output_path = os.path.join(
                    output_dir, f"update_{update_count}_output.png"
                )
                plt.imshow(x_concat, cmap='terrain', vmin=-1.5, vmax=1.5)
                plt.savefig(output_path)
                plt.close()
                torch.save(algo.model.state_dict(), os.path.join(ckpt_dir, f"diffusion_{update_count}.pt"))
            
            if update_count > cfg.num_updates:
                break

    torch.save(algo.model.state_dict(), os.path.join(ckpt_dir, f"diffusion_{update_count}.pt"))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug mode")
    parser.set_defaults(debug=True)
    parser.add_argument(
        "--dataset_path", type=str, default="/home/yoonho/Workspace/RLLab/Multiverse/extreme-parkour/legged_gym/legged_gym/utils/heightmap_dataset/"
    )
    parser.add_argument("--vae_dir", type=str, default="/home/yoonho/Workspace/RLLab/Multiverse/gp_terrain/results/vae/")
    parser.add_argument("--input_h", type=int, default=368)
    parser.add_argument("--input_w", type=int, default=80)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--denoising_step", type=int, default=128)

    parser.add_argument("--energy_lr", type=float, default=1e-4)
    parser.add_argument("--energy_weight_decay", type=float, default=1e-4)
    parser.add_argument("--guidance_lr", type=float, default=1e-4)
    parser.add_argument("--guidance_weight_decay", type=float, default=1e-4)

    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_dt", type=int, default=4)

    parser.add_argument("--num_updates", type=int, default=150000)
    parser.add_argument("--save_dir", type=str, default="results/diffusion/")
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500)

    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    args_dict = vars(args)
    cfg: DictConfig = OmegaConf.create(args_dict)
    train(args)

# if __name__ == "__main__":
#     # Load configuration
#     args_dict = {
#         "vae_dir": "/home/yoonho/Workspace/RLLab/Multiverse/gp_terrain/results/vae/",
#         "save_dir": "results/diffusion/",
#         "eval_batch_size": 16,
#         "eval_dt": 4,
#         "device": "cuda",
#         "debug": True,
#         "input_h": 368,
#         "input_w": 80,
#         "hidden_size": 256,
#         "depth": 12,
#         "num_heads": 8,
#         "tokenize_scale": 8,
#         "tokenize_dim": 32,
#         "ema_decay": 0.99,
#         "denoising_step": 128,
#         "learning_rate": 1e-4,
#         "weight_decay": 1e-4,

#     }
#     cfg = OmegaConf.create(args_dict)

#     # Test the diffusion model
#     test_diffusion_model(cfg)

# def test_diffusion_model(cfg):
#     # Load the trained VAE
#     vae = VAE()
#     vae_ckpt = torch.load(
#         os.path.join(cfg.vae_dir, "ckpts/vae.pt"),
#         weights_only=True,
#     )
#     vae.load_state_dict(vae_ckpt)
#     vae.to(cfg.device)
#     vae.eval()

#     # Load the trained diffusion model
#     algo = FlowMatching(cfg)
#     diffusion_ckpt = torch.load(os.path.join(cfg.save_dir, "ckpts/diffusion.pt"))
#     algo.model.load_state_dict(diffusion_ckpt)
#     algo.model.to(cfg.device)
#     algo.model.eval()

#     # Generate samples
#     print("Generating samples...")
#     x_concat = generate_output(algo, vae, cfg.eval_batch_size, cfg.eval_dt)

#     # Save the generated samples
#     output_path = os.path.join(cfg.save_dir, "test_output.png")
#     plt.imshow(x_concat, cmap='terrain', vmin=-1.5, vmax=1.5)
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Generated samples saved to {output_path}")