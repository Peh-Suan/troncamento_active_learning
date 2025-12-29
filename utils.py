import torch
import pickle
import os
import torch.nn as nn
from troncamento_datasets import SegmentPairDataset
import random
import tqdm
from model import uncertainty
import pandas as pd

from torch.utils.data import random_split, DataLoader

def make_train_val_loaders(dataset, batch_size=4, val_ratio=0.0002, seed=42):
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, phone_ids, y in tqdm.tqdm(loader, leave=False):

        x = x.to(device)
        phone_ids = phone_ids.to(device)
        y = y.to(device)

        logits = model(x, phone_ids)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        preds = (logits > 0).long()
        total_correct += (preds == y).sum().item()
        total_seen += batch_size

    return {
        "loss": total_loss / total_seen,
        "acc": total_correct / total_seen
    }


import tqdm
import matplotlib.pyplot as plt

def pretrain_one_epoch(
    dataset,
    model,
    optimizer,
    criterion,
    device,
    pretrain_ckpt,
    batch_size=4
):
    train_loader, val_loader = make_train_val_loaders(
        dataset,
        batch_size=batch_size
    )

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    # ---- logging buffers (VAL only) ----
    log_steps = []
    val_losses = []
    val_accs = []

    pbar = tqdm.tqdm(train_loader)
    step = -1

    for x, phone_ids, y in pbar:
        step += 1
        x = x.to(device)
        phone_ids = phone_ids.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x, phone_ids)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # ---- train stats (for progress bar only) ----
        batch_size = y.size(0)
        loss += loss.item() * batch_size
        preds = (logits > 0).long()
        correct = (preds == y).sum().item()
        seen = batch_size

        pbar.set_description(
            f"train loss={loss/seen:.4f}, "
            f"train acc={correct/seen:.4f}"
        )

        # ---- validation every 1000 steps ----
        if step % 10 == 0 and step > 0:
            val_stats = evaluate(model, val_loader, criterion, device)

            log_steps.append(step)
            val_losses.append(val_stats["loss"])
            val_accs.append(val_stats["acc"])

            model.train()  # switch back

            # ---- plot validation curves ----
            fig, ax1 = plt.subplots(figsize=(6, 4))

            ax1.plot(log_steps, val_losses, label="Val Loss")
            ax1.set_xlabel("Training step")
            ax1.set_ylabel("Loss")

            ax2 = ax1.twinx()
            ax2.plot(log_steps, val_accs, color="tab:orange", label="Val Accuracy")
            ax2.set_ylabel("Accuracy")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(pretrain_ckpt), "pretrain_val_curve.png"))
            plt.close("all")
        if step % 100 == 0 and step > 0:
            torch.save(model.state_dict(), pretrain_ckpt)
            
    print(f"Training completed.")
    print("Checkpoint saved at ", pretrain_ckpt)
    torch.save(model.state_dict(), pretrain_ckpt)

def train_on_gold_dataset(model_class, model_folder, base_dataset, device="cpu", pretained_ckpt=None):
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    ckpt_path = os.path.join(model_folder, "model.pt")
    train_history_path = os.path.join(model_folder, "train_history.pkl")

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
        train_history = {
            "iteration": [],
            "train_loss": [],
            "trained_samples": []
        }
        if pretained_ckpt is not None:
            model.load_state_dict(torch.load(pretained_ckpt))
    else:
        ckpt_path = os.path.join(model_folder, "model.pt")
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path))
            with open(train_history_path, "rb") as f:
                train_history = pickle.load(f)
        else:
            train_history = {
                "iteration": [],
                "train_loss": [],
                "trained_samples": []
            }
            if pretained_ckpt is not None:
                model.load_state_dict(torch.load(pretained_ckpt))

    gold_dataset, unique_ids = base_dataset.return_gold_dataset(trained_samples=train_history["trained_samples"])

    if len(gold_dataset) == 0:
        print(f"All gold samples have been trained on.")
        return


    print(f"Training on {len(gold_dataset)} gold samples.")
    segment_pair_dataset = SegmentPairDataset(gold_dataset)

    loader = torch.utils.data.DataLoader(
        segment_pair_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

    model.train()
    total_loss = 0

    for (x, phone_ids, y), unique_id in tqdm.tqdm(zip(loader, unique_ids)):
        x = x.to(device)
        phone_ids = phone_ids.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x, phone_ids)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_history["iteration"].append(len(train_history["iteration"]) + 1)
        train_history["train_loss"].append(loss.item())
        train_history["trained_samples"].append(unique_id)

    print(f"Trained on {len(gold_dataset)} gold samples: loss = {total_loss / len(loader):.4f}")

    # save model checkpoint
    ckpt_path = os.path.join(model_folder, "model.pt")
    torch.save(model.state_dict(), ckpt_path)
    print("Model checkpoint saved at ", ckpt_path)
    # save training history
    with open(train_history_path, "wb") as f:
        pickle.dump(train_history, f)

def select_uncertain_pos_neg(logits, k):
    """
    logits: Tensor of shape (N,) or (N, 1)
    k: number of samples per group

    Returns:
        pos_idx: indices of top-k uncertain predicted-positive samples
        neg_idx: indices of top-k uncertain predicted-negative samples
    """
    logits = logits.view(-1)

    # probabilities
    p = torch.sigmoid(logits)

    # predicted labels
    preds = (p >= 0.5)

    # uncertainty (Bernoulli entropy)
    eps = 1e-8
    uncertainty = -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)

    # split indices
    pos_indices = torch.where(preds)[0]
    neg_indices = torch.where(~preds)[0]

    # uncertainties per group
    pos_uncert = uncertainty[pos_indices]
    neg_uncert = uncertainty[neg_indices]

    # top-k uncertain within each group
    k_pos = min(k, len(pos_indices))
    k_neg = min(k, len(neg_indices))

    _, pos_topk_idx = torch.topk(pos_uncert, k_pos)
    _, neg_topk_idx = torch.topk(neg_uncert, k_neg)

    pos_selected = pos_indices[pos_topk_idx]
    neg_selected = neg_indices[neg_topk_idx]

    return pos_selected, neg_selected

def select_uncertain_samples(
    model,
    target_basedataset,
    random_sample=100,
    k=50,
    device="cpu"
):
    model.eval()

    target_basedataset._refresh_gold_labels()

    excluded_ids = {
        idx
        for idx in range(len(target_basedataset))
        if target_basedataset.get_unique_id(idx)
        in target_basedataset.gold_labels["label"]
    }

    target_dataset = SegmentPairDataset(target_basedataset)

    if random_sample is not None and random_sample < len(target_basedataset):
        pool_indices = random.sample(range(len(target_basedataset)), random_sample)
    else:
        pool_indices = list(range(len(target_basedataset)))

    all_logits = []
    all_indices = []

    with torch.no_grad():
        for idx in tqdm.tqdm(pool_indices):
            if idx in excluded_ids:
                continue

            x, phone_ids, y = target_dataset[idx]

            logit = model(
                x.unsqueeze(0).to(device),
                phone_ids.unsqueeze(0).to(device)
            ).squeeze(0)

            all_logits.append(logit)
            all_indices.append(idx)

    logits = torch.stack(all_logits)

    pos_idx, neg_idx = select_uncertain_pos_neg(logits, k=k)

    selected_dataset_indices = (
        [all_indices[i] for i in pos_idx.tolist()] +
        [all_indices[i] for i in neg_idx.tolist()]
    )

    for idx in selected_dataset_indices:
        target_basedataset.add_data_to_gold(idx, label=None)

def put_files_to_folder(target_basedataset, folder_path="selected_samples", tgrd_fs_folder="it_vxc_textgrids17_acoustic17"):
    target_basedataset._refresh_gold_labels()
    for unique_id, label in target_basedataset.gold_labels["label"].items():
        if pd.isna(label):
            target_basedataset.put_file_to_folder(unique_id, folder_path=folder_path, tgrd_fs_folder=tgrd_fs_folder)

def delete_annotated_files(target_basedataset, folder_path="selected_samples"):
    target_basedataset._refresh_gold_labels()
    for unique_id, label in target_basedataset.gold_labels["label"].items():
        if pd.notna(label):
            target_basedataset.delete_file_from_folder(unique_id, folder_path=folder_path)