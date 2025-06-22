import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import EmoTalk
from dataset import EmoTalkDataset
from loss import compute_total_loss
from tqdm import tqdm

class Args:
    feature_dim = 832
    bs_dim = 52
    period = 30
    max_seq_len = 5000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    num_workers = 2
    epochs = 1
    learning_rate = 1e-4
    dataset_root = './3D-ETF'
    dataset_name = 'RAVDESS'  # 또는 "HDTF"
    save_path = './checkpoints'
    resume = False
    resume_path = './checkpoints/last.pth'

args = Args()

def train():
    os.makedirs(args.save_path, exist_ok=True)

    # 1. Dataset & DataLoader
    dataset = EmoTalkDataset(root_dir=args.dataset_root, dataset_name=args.dataset_name, max_len=64000)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # 2. Model
    model = EmoTalk(args).to(args.device)

    if args.resume and os.path.exists(args.resume_path):
        print(f"🔄 Resume from {args.resume_path}")
        model.load_state_dict(torch.load(args.resume_path))

    # 3. Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 4. Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 40}")

        for batch in pbar:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(args.device)

            bs_output11, bs_output12, emo_logits = model(batch)

            loss = compute_total_loss(bs_output11, bs_output12, emo_logits, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(loader)
        print(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.4f}")

        # Save checkpoint
        # torch.save(model.state_dict(), os.path.join(args.save_path, f"epoch_{epoch:03d}.pth"))
        torch.save(model.state_dict(), os.path.join(args.save_path, "last.pth"))


if __name__ == "__main__":
    train()