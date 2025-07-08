import os
import argparse
from omegaconf import OmegaConf, DictConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

from algo.vae import VAE
from dataset import build_dataset

print("Train script is starting...")


def generate_output(vae, val_loader, val_iterator):
    vae.eval()

    try:
        x_batch = next(val_iterator)
    except StopIteration:
        val_iterator = iter(val_loader)
        x_batch = next(val_iterator)

    with torch.no_grad():
        x_recon, *_ = vae(x_batch)
        x_recon = x_recon.detach().cpu().numpy()
        x_gt = x_batch.detach().cpu().numpy()

    x_recon = x_recon.transpose((0,3,2,1))
    x_gt = x_gt.transpose((0,3,2,1))

    x_recon = x_recon.reshape((-1, x_recon.shape[2], x_recon.shape[3]))
    x_gt = x_gt.reshape((-1, x_gt.shape[2], x_gt.shape[3]))

    return np.concatenate([x_gt, x_recon], axis=1)


def train(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)

    ckpt_dir = os.path.join(cfg.save_dir, "ckpts")
    output_dir = os.path.join(cfg.save_dir, "outputs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print("building network ...")
    vae = VAE()
    vae.to(cfg.device)

    optimizer = torch.optim.AdamW(
        vae.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    loss_fn = torch.nn.MSELoss()

    print("building dataset ...")
    train_dataset, val_dataset = build_dataset(cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    val_iterator = iter(val_loader)

    # generate pre-trained model outputs
    x_concat = generate_output(vae, val_loader, val_iterator)
    output_path = os.path.join(output_dir, f"update_0_output.png")
    plt.imshow(x_concat, cmap='terrain', vmin=-1.5, vmax=1.5)
    plt.savefig(output_path)
    plt.close()

    recon_loss_sum = 0.0
    kl_loss_sum = 0.0
    update_count = 0
    start_time = time.time()

    print("training ...")
    while update_count < cfg.num_updates:
        for __, x_batch in enumerate(train_loader):
            vae.train()
            x_recon, mu, logvar = vae(x_batch)

            recon_loss = loss_fn(x_batch, x_recon)
            kl_div = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1)

            # use cyclic kl loss annealing
            # Usually, smaller kl loss weight is used to train vae used for latent diffusion for better reconstruction quality
            # But here we use 0.001, maybe you can change it to a smaller value
            kl_loss_weight = 0.001
            if update_count % 10000  < 5000:
                kl_loss_weight *= (update_count % 10000) / 5000

            loss = recon_loss + kl_div * kl_loss_weight

            recon_loss_sum += recon_loss.item()
            kl_loss_sum += kl_div.item()


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_count += 1

            if cfg.debug and (update_count % cfg.log_every == 0):
                print(
                    f"Update count: {update_count} | Recon Loss: {recon_loss_sum / cfg.log_every:.6f} | KL Loss: {kl_loss_sum / cfg.log_every:.6f} | Time: {time.time()-start_time:.6f}"
                )
                start_time = time.time()
                recon_loss_sum = 0.0
                kl_loss_sum = 0.0

            if update_count % cfg.eval_every == 0:
                # generate outputs
                x_concat = generate_output(vae, val_loader, val_iterator)
                output_path = os.path.join(
                    output_dir, f"update_{update_count}_output.png"
                )
                plt.imshow(x_concat, cmap='terrain', vmin=-1.5, vmax=1.5)
                plt.savefig(output_path)
                plt.close()
            
            if update_count > cfg.num_updates:
                break

    torch.save(vae.state_dict(), os.path.join(ckpt_dir, "vae.pt"))

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug mode")
    parser.set_defaults(debug=True)
    parser.add_argument(
        "--dataset_path", type=str, default="/home/yoonho/Workspace/RLLab/Multiverse/extreme-parkour/legged_gym/legged_gym/utils/heightmap_dataset/"
    )
    parser.add_argument("--save_dir", type=str, default="results/vae/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_updates", type=int, default=50000)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    args_dict = vars(args)
    cfg: DictConfig = OmegaConf.create(args_dict)
    train(args)