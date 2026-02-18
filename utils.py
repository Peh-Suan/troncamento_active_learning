"""
Rewritten training + evaluation + active-learning utilities for:
- Variable-length wav2vec2 hidden states (T x 1024)
- Attention pooling
- Two-head model: vowel (/e/) + bad/misaligned

Assumptions:
- Your SegmentPairDataset returns:
    (hidden, phone_ids, silver_label, vowel_y, vowel_mask, bad_y)
  where:
    hidden: (T, 1024) float tensor
    phone_ids: (2,) long tensor
    silver_label: scalar float tensor
    vowel_y: scalar float tensor (dummy 0.0 when vowel_mask=0)
    vowel_mask: scalar float tensor (1.0 valid, 0.0 bad)
    bad_y: scalar float tensor (1.0 bad, 0.0 valid)

- Your BaseDataset.__getitem__ stores hidden in ex["hidden"] already.
"""

import os
import random
import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from troncamento_datasets import SegmentPairDataset  # uses your dataset code


# -----------------------------
# Collate (pad hidden sequences)
# -----------------------------
def collate_segment_batch(batch):
    """
    batch: list of tuples:
      hidden (T,1024), phone_ids (2,), silver (), vowel_y (), vowel_mask (), bad_y ()
    returns:
      hidden_padded: (B, T_max, 1024)
      seq_mask:      (B, T_max) bool
      phone_ids:     (B, 2) long
      silver:        (B,) float
      vowel_y:       (B,) float
      vowel_mask:    (B,) float
      bad_y:         (B,) float
    """
    hiddens, phone_ids_list, silver_list, vowel_y_list, vowel_mask_list, bad_y_list = zip(*batch)

    lengths = torch.tensor([h.size(0) for h in hiddens], dtype=torch.long)
    B = len(hiddens)
    T_max = int(lengths.max().item())
    D = int(hiddens[0].size(1))

    hidden_padded = torch.zeros(B, T_max, D, dtype=torch.float32)
    seq_mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, h in enumerate(hiddens):
        T = h.size(0)
        hidden_padded[i, :T] = h
        seq_mask[i, :T] = True

    phone_ids = torch.stack(phone_ids_list, dim=0)          # (B, 2)
    silver = torch.stack(silver_list, dim=0).view(B)        # (B,)

    vowel_y = torch.stack(vowel_y_list, dim=0).view(B)      # (B,)
    vowel_mask = torch.stack(vowel_mask_list, dim=0).view(B)  # (B,)
    bad_y = torch.stack(bad_y_list, dim=0).view(B)          # (B,)

    return hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y


# -----------------------------
# Optional: zero silver feature
# -----------------------------
def clamp_silver_column(model):
    """
    If you want to prevent the model from using silver_label as an input feature,
    you can zero-out the corresponding column in the first Linear layer.

    IMPORTANT: This assumes your model has model.shared[0] as the first Linear.
    If your architecture differs, update accordingly.
    """
    if not hasattr(model, "shared"):
        return
    if not isinstance(model.shared[0], nn.Linear):
        return

    first = model.shared[0]
    silver_col = first.weight.shape[1] - 1  # last concat feature is silver_label
    with torch.no_grad():
        first.weight[:, silver_col].zero_()


# -----------------------------
# Train/val split loaders
# -----------------------------
def make_train_val_loaders(dataset, batch_size=4, val_ratio=0.05, seed=42):
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_ratio))
    n_train = max(1, n_total - n_val)

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_segment_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_segment_batch,
    )

    return train_loader, val_loader


# -----------------------------
# Evaluation (two-head metrics)
# -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, pbar, thr_bad=0.5, thr_e=0.5, lambda_e=1.0):
    """
    Returns:
      loss (combined), bad_loss, vowel_loss,
      bad_* metrics on all samples,
      vowel_* metrics on valid only (vowel_mask==1),
      counts.
    """
    model.eval()
    crit_none = nn.BCEWithLogitsLoss(reduction="none")

    total_n = 0
    total_loss = 0.0

    total_bad_loss = 0.0
    total_vowel_loss = 0.0
    total_vowel_n = 0

    # Bad metrics (all)
    TPb = FPb = FNb = TNb = 0

    # Vowel metrics (valid only)
    TPe = FPe = FNe = TNe = 0
    total_valid = 0
    subpbar = tqdm.tqdm(loader, desc="Evaluating", leave=False)

    for hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y in subpbar:
        hidden_padded = hidden_padded.to(device)
        seq_mask = seq_mask.to(device)
        phone_ids = phone_ids.to(device)
        silver = silver.to(device)

        vowel_y = vowel_y.to(device)
        vowel_mask = vowel_mask.to(device)
        bad_y = bad_y.to(device)

        logit_e, logit_bad, _ = model(hidden_padded, seq_mask, phone_ids, silver)

        B = bad_y.size(0)

        # losses
        loss_bad_vec = crit_none(logit_bad, bad_y)  # (B,)
        loss_bad = loss_bad_vec.mean()

        valid = (vowel_mask > 0.5)
        if valid.any():
            loss_e_vec = crit_none(logit_e, vowel_y)  # (B,)
            loss_e = loss_e_vec[valid].mean()
            total_vowel_loss += loss_e.item() * int(valid.sum().item())
            total_vowel_n += int(valid.sum().item())
        else:
            loss_e = 0.0 * loss_bad

        loss_all = loss_bad + lambda_e * loss_e

        total_loss += loss_all.item() * B
        total_bad_loss += loss_bad.item() * B
        total_n += B

        # bad metrics
        prob_bad = torch.sigmoid(logit_bad)
        pred_bad = (prob_bad >= thr_bad).long()
        yb = bad_y.long()

        TPb += ((pred_bad == 1) & (yb == 1)).sum().item()
        FPb += ((pred_bad == 1) & (yb == 0)).sum().item()
        FNb += ((pred_bad == 0) & (yb == 1)).sum().item()
        TNb += ((pred_bad == 0) & (yb == 0)).sum().item()

        # vowel metrics on valid only
        if valid.any():
            prob_e = torch.sigmoid(logit_e[valid])
            pred_e = (prob_e >= thr_e).long()
            ye = vowel_y[valid].long()

            TPe += ((pred_e == 1) & (ye == 1)).sum().item()
            FPe += ((pred_e == 1) & (ye == 0)).sum().item()
            FNe += ((pred_e == 0) & (ye == 1)).sum().item()
            TNe += ((pred_e == 0) & (ye == 0)).sum().item()
            total_valid += int(valid.sum().item())
        
        pbar.set_postfix({"progress": subpbar})

    def prf(TP, FP, FN, TN):
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        return acc, precision, recall, f1

    acc_bad, prec_bad, rec_bad, f1_bad = prf(TPb, FPb, FNb, TNb)
    if total_valid > 0:
        acc_e, prec_e, rec_e, f1_e = prf(TPe, FPe, FNe, TNe)
    else:
        acc_e = prec_e = rec_e = f1_e = float("nan")

    results = {
        "loss": total_loss / max(total_n, 1),
        "bad_loss": total_bad_loss / max(total_n, 1),
        "vowel_loss": total_vowel_loss / max(total_vowel_n, 1),

        "bad_acc": acc_bad,
        "bad_f1": f1_bad,
        "bad_precision": prec_bad,
        "bad_recall": rec_bad,

        # "vowel_acc": acc_e,
        # "vowel_f1": f1_e,
        # "vowel_precision": prec_e,
        # "vowel_recall": rec_e,

        "n_total": total_n,
        "n_valid_for_e": total_valid,
    }

    # Only add vowel metrics if there were any valid samples
    if total_valid > 0:
        results.update({
            "vowel_acc": acc_e,
            "vowel_f1": f1_e,
            "vowel_precision": prec_e,
            "vowel_recall": rec_e,
        })
    else:
        # Optional: explicitly mark as missing (useful for logging)
        results.update({
            "vowel_acc": None,
            "vowel_f1": None,
            "vowel_precision": None,
            "vowel_recall": None,
        })
    
    return results



# -----------------------------
# Pretrain one epoch (with val)
# -----------------------------
import matplotlib.pyplot as plt

def pretrain_one_epoch(
    dataset,
    model,
    optimizer,
    device,
    pretrain_ckpt="model_ckpt/prepretrain.pt",
    batch_size=8,
    val_ratio=0.05,
    lambda_e=1.0,
    eval_every=5,
    save_every=500,
    patience=20,
    clamp_silver=False,
):

    train_loader, val_loader = make_train_val_loaders(dataset, batch_size=batch_size, val_ratio=val_ratio)

    crit_none = nn.BCEWithLogitsLoss(reduction="none")

    model.train()

    log_steps = []
    val_losses_bad = []
    val_accs_bad = []
    val_losses_vowel = []
    val_accs_vowel = []
    val_losses_all = []

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    step = 0
    pbar = tqdm.tqdm(train_loader, desc="pretrain")

    for batch in pbar:
        step += 1
        hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y = batch

        hidden_padded = hidden_padded.to(device)
        seq_mask = seq_mask.to(device)
        phone_ids = phone_ids.to(device)
        silver = silver.to(device)

        vowel_y = vowel_y.to(device)
        vowel_mask = vowel_mask.to(device)
        bad_y = bad_y.to(device)

        logit_e, logit_bad, _ = model(hidden_padded, seq_mask, phone_ids, silver)

        # losses
        loss_bad_vec = crit_none(logit_bad, bad_y)
        loss_bad = loss_bad_vec.mean()

        valid = (vowel_mask > 0.5)
        if valid.any():
            loss_e_vec = crit_none(logit_e, vowel_y)
            loss_e = loss_e_vec[valid].mean()
        else:
            loss_e = 0.0 * loss_bad

        loss_all = loss_bad + lambda_e * loss_e

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        if clamp_silver:
            clamp_silver_column(model)

        pbar.set_postfix(loss=float(loss_all.item()), bad=float(loss_bad.item()),
                         vowel=float(loss_e.item()) if valid.any() else 0.0)

        # validation
        if step % eval_every == 0:
            val_stats = evaluate(model, val_loader, device, pbar, lambda_e=lambda_e)
            log_steps.append(step)

            val_losses_all.append(val_stats["loss"])
            val_losses_bad.append(val_stats["bad_loss"])
            val_accs_bad.append(val_stats["bad_acc"])
            val_losses_vowel.append(val_stats["vowel_loss"])
            val_accs_vowel.append(val_stats["vowel_acc"])

            model.train()

            # plot
            fig, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(log_steps, val_losses_bad, label="Val Loss (Bad)", ls="-.")
            ax1.plot(log_steps, val_losses_vowel, label="Val Loss (Vowel)", ls="--")
            ax1.plot(log_steps, val_losses_all, label="Val Loss (Total)")
            ax1.set_xlabel("Training step")
            ax1.set_ylabel("Loss")

            ax2 = ax1.twinx()
            ax2.plot(log_steps, val_accs_bad, label="Val Acc (Bad)", ls="-.")
            ax2.plot(log_steps, val_accs_vowel, label="Val Acc (Vowel)", ls="--")
            ax2.set_ylabel("Accuracy")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(pretrain_ckpt), "pretrain_val_curve.png"), dpi=200)
            plt.close("all")

            # early stopping
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    if best_state is not None:
                        model.load_state_dict(best_state)
                    break

        if step % save_every == 0:
            torch.save(model.state_dict(), pretrain_ckpt)

    torch.save(model.state_dict(), pretrain_ckpt)
    print("Checkpoint saved at", pretrain_ckpt)


# -----------------------------
# Score distribution plotting
# -----------------------------
@torch.no_grad()
def collect_probs_twohead(model, loader, device):
    model.eval()
    p_bad_all, y_bad_all = [], []
    p_e_all, y_e_all = [], []

    for hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y in loader:
        hidden_padded = hidden_padded.to(device)
        seq_mask = seq_mask.to(device)
        phone_ids = phone_ids.to(device)
        silver = silver.to(device)
        vowel_y = vowel_y.to(device)
        vowel_mask = vowel_mask.to(device)
        bad_y = bad_y.to(device)

        logit_e, logit_bad, _ = model(hidden_padded, seq_mask, phone_ids, silver)

        p_bad = torch.sigmoid(logit_bad)
        p_bad_all.append(p_bad.detach().cpu().numpy())
        y_bad_all.append(bad_y.detach().cpu().numpy())

        valid = (vowel_mask > 0.5)
        if valid.any():
            p_e = torch.sigmoid(logit_e[valid])
            y_e = vowel_y[valid]
            p_e_all.append(p_e.detach().cpu().numpy())
            y_e_all.append(y_e.detach().cpu().numpy())

    return {
        "p_bad": np.concatenate(p_bad_all) if len(p_bad_all) else np.array([]),
        "y_bad": np.concatenate(y_bad_all) if len(y_bad_all) else np.array([]),
        "p_e": np.concatenate(p_e_all) if len(p_e_all) else np.array([]),
        "y_e": np.concatenate(y_e_all) if len(y_e_all) else np.array([]),
    }


def plot_score_distributions(probs, ys, threshold=0.5, bins=40, title=None, savepath=None,
                             xlabel="Predicted probability P(y=1)"):
    probs = np.asarray(probs)
    ys = np.asarray(ys)
    p_pos = probs[ys == 1]
    p_neg = probs[ys == 0]

    plt.figure(figsize=(6, 4))
    plt.hist(p_neg, bins=bins, range=(0, 1), alpha=0.5, density=True, label="True negatives (y=0)")
    plt.hist(p_pos, bins=bins, range=(0, 1), alpha=0.5, density=True, label="True positives (y=1)")
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.2f}")

    plt.xlim(0, 1)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()


# -----------------------------
# Uncertainty utilities
# -----------------------------
def entropy_from_prob(p):
    eps = 1e-8
    return -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)


def select_topk(scores, k, largest=True):
    scores = scores.view(-1)
    k = min(k, scores.numel())
    vals, idx = torch.topk(scores, k, largest=largest)
    return idx, vals


# -----------------------------
# Active learning selection
# -----------------------------
def select_uncertain_samples(
    model,
    target_basedataset,
    random_sample=200,
    k=50,
    device="cpu",
    log_csv_path="uncertainty_log.csv",
    most_certain=False,
    alpha_valid=1.0,   # how strongly to gate by (1 - p_bad)
    max_uncertainty=None,
):
    """
    Picks uncertain samples among unlabeled pool using:
      score = entropy(p_e) * (1 - p_bad)^alpha_valid

    Adds selected items to current_gold_labels.csv with label=None via add_data_to_gold().
    """
    model.eval()
    target_basedataset._refresh_gold_labels()

    excluded_ids = {
        idx for idx in range(len(target_basedataset))
        if target_basedataset.get_unique_id(idx) in target_basedataset.gold_labels["label"]
    }

    target_dataset = SegmentPairDataset(target_basedataset)

    if random_sample is not None and random_sample < len(target_basedataset):
        pool_indices = random.sample(range(len(target_basedataset)), random_sample)
    else:
        pool_indices = list(range(len(target_basedataset)))

    all_scores = []
    all_indices = []
    all_p_e = []
    all_p_bad = []

    with torch.no_grad():
        for idx in tqdm.tqdm(pool_indices, desc="scoring pool"):
            if idx in excluded_ids:
                continue

            hidden, phone_ids, silver_label, vowel_y, vowel_mask, bad_y = target_dataset[idx]

            hidden_b = hidden.unsqueeze(0).to(device)  # (1,T,1024)
            mask_b = torch.ones(1, hidden.size(0), dtype=torch.bool, device=device)  # (1,T)

            phone_b = phone_ids.unsqueeze(0).to(device)  # (1,2)
            silver_b = silver_label.view(1).to(device)   # (1,)

            logit_e, logit_bad, _ = model(hidden_b, mask_b, phone_b, silver_b)

            p_e = torch.sigmoid(logit_e).squeeze(0)
            p_bad = torch.sigmoid(logit_bad).squeeze(0)

            score = entropy_from_prob(p_e) + entropy_from_prob(p_bad) * alpha_valid

            all_scores.append(score)
            all_indices.append(idx)
            all_p_e.append(p_e.detach().cpu())
            all_p_bad.append(p_bad.detach().cpu())

    if len(all_scores) == 0:
        print("No available unlabeled samples in pool.")
        return

    scores = torch.stack(all_scores)  # (N,)
    probs_e = torch.stack(all_p_e).cpu().numpy()
    probs_bad = torch.stack(all_p_bad).cpu().numpy()

    # pick top-k uncertain (or most certain if requested)
    sel_idx, sel_vals = select_topk(scores, k=k, largest=not most_certain)
    selected_dataset_indices = [all_indices[i] for i in sel_idx.tolist()]

    for idx in selected_dataset_indices:
        target_basedataset.add_data_to_gold(idx, label=None)

    # logging
    if os.path.exists(log_csv_path):
        log_df = pd.read_csv(log_csv_path)
        run_idx = len(log_df)
    else:
        log_df = pd.DataFrame(columns=["idx", "mean_score", "selected_mean_score"])
        run_idx = 0

    log_df = pd.concat([log_df, pd.DataFrame([{
        "idx": run_idx,
        "mean_score": float(scores.mean().item()),
        "selected_mean_score": float(scores[sel_idx].mean().item()),
    }])], ignore_index=True)
    log_df.to_csv(log_csv_path, index=False)

    # quick plots
    os.makedirs("log_plots", exist_ok=True)
    import glob
    import matplotlib.pyplot as plt

    plot_idx = len(glob.glob("log_plots/uncertainty_*.png"))
    order = torch.argsort(scores)
    scores_sorted = scores[order].detach().cpu()
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(order.numel(), device=order.device)
    sel_x = inv_order[sel_idx].detach().cpu()

    if not max_uncertainty: max_uncertainty = scores.max().item()
    plt.figure()
    plt.plot(scores_sorted.numpy())
    plt.scatter(sel_x.numpy(), scores_sorted[sel_x].numpy(), label="selected")
    plt.ylim(0, max_uncertainty * 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"log_plots/uncertainty_{plot_idx}.png", dpi=200)
    plt.close()

    import numpy as np
    import matplotlib.pyplot as plt
    plot_idx = len(glob.glob("log_plots/ternary_bad_no_has_*.png"))

    # all_p_e / all_p_bad are lists of 0-dim tensors (or floats)
    p_e = np.array([float(x) for x in all_p_e])
    p_bad = np.array([float(x) for x in all_p_bad])

    # 3-way distribution on the simplex
    p_BAD = p_bad
    p_HAS = (1.0 - p_bad) * p_e
    p_NO  = (1.0 - p_bad) * (1.0 - p_e)

    # (optional) sanity check
    S = p_BAD + p_HAS + p_NO
    assert np.allclose(S, 1.0, atol=1e-6), (S.min(), S.max())

    # barycentric -> 2D coords
    # vertices: A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    # weights:  a at A, b at B, c at C  (a+b+c=1)
    a = p_BAD
    b = p_NO
    c = p_HAS

    x = b + 0.5 * c
    y = (np.sqrt(3) / 2.0) * c

    # plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # triangle boundary
    tri_x = [0, 1, 0.5, 0]
    tri_y = [0, 0, np.sqrt(3)/2, 0]
    ax.plot(tri_x, tri_y, lw=1)

    # scatter
    ax.scatter(x, y, s=10, alpha=0.6)

    # corner labels
    ax.text(-0.03, -0.03, "Bad", ha="right", va="top")
    ax.text(1.03, -0.03, "No vowel (no /e/) [good]", ha="left", va="top")
    ax.text(0.5, np.sqrt(3)/2 + 0.03, "Has vowel (yes /e/) [good]", ha="center", va="bottom")

    ax.set_aspect("equal", "box")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, np.sqrt(3)/2 + 0.08)
    ax.axis("off")
    ax.set_title("Ternary plot: Bad vs No-/e/ vs Has-/e/ (conditioned on not-bad)")

    plt.tight_layout()
    plt.savefig(f"log_plots/ternary_bad_no_has_{plot_idx}.png", dpi=250, bbox_inches="tight")
    plt.close()


    # hist_idx = len(glob.glob("log_plots/prob_hist_vowel_*.png"))
    # plt.figure()
    # plt.hist(probs_e, bins=30, density=True, edgecolor="black")
    # plt.xlabel("Predicted probability P(/e/=1)")
    # plt.ylabel("Density")
    # plt.title("Predicted probability distribution (vowel head)")
    # plt.tight_layout()
    # plt.savefig(f"log_plots/prob_hist_vowel_{hist_idx}.png", dpi=200)
    # plt.close()

    # # Optional: also plot P(bad)
    # histb_idx = len(glob.glob("log_plots/prob_hist_bad_*.png"))
    # plt.figure()
    # plt.hist(probs_bad, bins=30, density=True, edgecolor="black")
    # plt.xlabel("Predicted probability P(bad=1)")
    # plt.ylabel("Density")
    # plt.title("Predicted probability distribution (bad head)")
    # plt.tight_layout()
    # plt.savefig(f"log_plots/prob_hist_bad_{histb_idx}.png", dpi=200)
    # plt.close()

    return max_uncertainty


# -----------------------------
# File operations for annotation
# -----------------------------
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

import os
import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_on_gold_dataset(
    model_class,
    model_folder,
    base_dataset,
    batch_size=8,
    device="cpu",
    pretained_ckpt="model_ckpt/prepretrain.pt",
    do_eval=True,
    train_type="pretrain",
    upsample=True,
    upsample_ratio=1,
    save_ckpt="train.pt",
    lr=1e-4,
    lambda_e=1.0,
    patience=3,
    max_epochs=200,
    thr_bad=0.5,
    thr_e=0.5,
):
    """
    Trains the two-head model on gold labels:
      - bad_y (1 if misaligned, else 0) trains on all
      - vowel_y trains only where vowel_mask==1

    Expects SegmentPairDataset to return:
      hidden, phone_ids, silver, vowel_y, vowel_mask, bad_y
    and collate_segment_batch to return:
      hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y
    """
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    import seaborn as sns

    os.makedirs(model_folder, exist_ok=True)

    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    crit_none = nn.BCEWithLogitsLoss(reduction="none")

    if pretained_ckpt is not None and os.path.isfile(pretained_ckpt):
        model.load_state_dict(torch.load(pretained_ckpt, map_location=device))
        print(f"Loaded pretrained model from {pretained_ckpt}")

    # --- datasets ---
    train_gold_dataset, train_unique_ids = base_dataset.return_gold_dataset(
        dataset_type=train_type,
        upsample=upsample,
        upsample_ratio=upsample_ratio
    )
    if len(train_gold_dataset) == 0:
        print("No gold samples found in the training set. Exiting.")
        return

    print(f"Training on {len(train_gold_dataset)} gold samples ({train_type}).")

    if do_eval:
        val_gold_dataset, val_unique_ids = base_dataset.return_gold_dataset(dataset_type="val")
        print(f"Validation on {len(val_gold_dataset)} gold samples.")
        if len(val_gold_dataset) == 0:
            print("Warning: val set is empty, disabling eval.")
            do_eval = False

    train_ds = SegmentPairDataset(train_gold_dataset, dataset_type="gold")
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_segment_batch
    )

    if do_eval:
        val_loader = torch.utils.data.DataLoader(
            SegmentPairDataset(val_gold_dataset, dataset_type="gold"),
            batch_size=max(1, batch_size),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_segment_batch
        )

    # --- training loop ---
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_counter = 0

    train_losses = []
    val_losses = []

    val_bad_accs, val_bad_f1s = [], []
    val_vowel_accs, val_vowel_f1s = [], []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        running_n = 0

        pbar = tqdm.tqdm(train_loader, desc=f"gold epoch {epoch+1}/{max_epochs}")
        for hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y in pbar:
            hidden_padded = hidden_padded.to(device)
            seq_mask = seq_mask.to(device)
            phone_ids = phone_ids.to(device)
            silver = silver.to(device)

            vowel_y = vowel_y.to(device)
            vowel_mask = vowel_mask.to(device)
            bad_y = bad_y.to(device)

            logit_e, logit_bad, _ = model(hidden_padded, seq_mask, phone_ids, silver)

            # bad loss (all)
            loss_bad = crit_none(logit_bad, bad_y).mean()

            # vowel loss (valid only)
            valid = (vowel_mask > 0.5)
            if valid.any():
                loss_e = crit_none(logit_e, vowel_y)[valid].mean()
            else:
                loss_e = 0.0 * loss_bad

            loss = loss_bad + lambda_e * loss_e

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            B = bad_y.size(0)
            running_loss += loss.item() * B
            running_n += B
            pbar.set_postfix(loss=float(loss.item()), bad=float(loss_bad.item()),
                             vowel=float(loss_e.item()) if valid.any() else 0.0)

        train_loss = running_loss / max(running_n, 1)
        train_losses.append(train_loss)

        # --- eval ---
        if do_eval:
            stats = evaluate(model, val_loader, device, pbar=pbar, thr_bad=thr_bad, thr_e=thr_e, lambda_e=lambda_e)
            val_losses.append(stats["loss"])
            val_bad_accs.append(stats["bad_acc"])
            val_bad_f1s.append(stats["bad_f1"])
            val_vowel_accs.append(stats["vowel_acc"])
            val_vowel_f1s.append(stats["vowel_f1"])

            # early stop
            if stats["loss"] < best_val_loss:
                best_val_loss = stats["loss"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
                    break

    # restore best
    if do_eval and best_state is not None:
        model.load_state_dict(best_state)

    # save checkpoint
    ckpt_path = os.path.join(model_folder, save_ckpt)
    torch.save(model.state_dict(), ckpt_path)
    print("Model checkpoint saved at", ckpt_path)

    # -------------------------
    # Plot curves
    # -------------------------
    if do_eval:
        colors = sns.color_palette("Set2", n_colors=10)
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(range(len(train_losses)), train_losses, label="Train Loss", color=colors[0])
        ax1.plot(range(len(val_losses)), val_losses, label="Val Loss", color=colors[1])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")

        ax2 = ax1.twinx()
        ax2.plot(range(len(val_bad_accs)), val_bad_accs, label="Val Bad Acc", ls="-.", color=colors[2])
        ax2.plot(range(len(val_vowel_accs)), val_vowel_accs, label="Val Vowel Acc", ls="--", color=colors[2])
        ax2.plot(range(len(val_bad_f1s)), val_bad_f1s, label="Val Bad F1", ls="-.", color=colors[3])
        ax2.plot(range(len(val_vowel_f1s)), val_vowel_f1s, label="Val Vowel F1", ls="--", color=colors[3])
        ax2.set_ylabel("Accuracy/F1")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(model_folder, "val_curve.png"), dpi=250, bbox_inches="tight")
        plt.close("all")

    # -------------------------
    # Diagnostics on val set
    # -------------------------
    if do_eval:

        # collect probabilities
        out = collect_probs_twohead(model, val_loader, device)
        from sklearn.metrics import precision_recall_curve, average_precision_score

        # --- ROC for bad head ---
        if len(out["y_bad"]) > 0 and len(np.unique(out["y_bad"])) > 1:
            fpr, tpr, _ = roc_curve(out["y_bad"], out["p_bad"])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, lw=2, label=f"Bad ROC (AUC={roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC — Bad Head")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(model_folder, "roc_bad.png"), dpi=250, bbox_inches="tight")
            plt.close()

        # --- PR for bad head ---
        if len(out["y_bad"]) > 0 and len(np.unique(out["y_bad"])) > 1:
            precision, recall, _ = precision_recall_curve(out["y_bad"], out["p_bad"])
            ap = average_precision_score(out["y_bad"], out["p_bad"])  # AUC-PR

            plt.figure()
            plt.plot(recall, precision, lw=2, label=f"Bad PR (AP={ap:.2f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall — Bad Head")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(model_folder, "pr_bad.png"), dpi=250, bbox_inches="tight")
            plt.close()


        # --- ROC for vowel head (valid only) ---
        if len(out["y_e"]) > 0 and len(np.unique(out["y_e"])) > 1:
            fpr, tpr, _ = roc_curve(out["y_e"], out["p_e"])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, lw=2, label=f"Vowel ROC (AUC={roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC — Vowel Head (valid only)")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(model_folder, "roc_vowel.png"), dpi=250, bbox_inches="tight")
            plt.close()

        # --- PR for vowel head (valid only) ---
        if len(out["y_e"]) > 0 and len(np.unique(out["y_e"])) > 1:
            precision, recall, _ = precision_recall_curve(out["y_e"], out["p_e"])
            ap = average_precision_score(out["y_e"], out["p_e"])

            plt.figure()
            plt.plot(recall, precision, lw=2, label=f"Vowel PR (AP={ap:.2f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision–Recall — Vowel Head (valid only)")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(os.path.join(model_folder, "pr_vowel.png"), dpi=250, bbox_inches="tight")
            plt.close()

        # --- Score distributions ---
        plot_score_distributions(
            out["p_bad"], out["y_bad"],
            threshold=thr_bad,
            title="P(bad=1) distributions",
            xlabel="Predicted probability P(bad=1)",
            savepath=os.path.join(model_folder, "score_dist_bad.png")
        )
        plot_score_distributions(
            out["p_e"], out["y_e"],
            threshold=thr_e,
            title="P(/e/=1) distributions (valid only)",
            xlabel="Predicted probability P(/e/=1)",
            savepath=os.path.join(model_folder, "score_dist_vowel.png")
        )

        # --- Confusion matrices ---
        # bad cm on all
        if len(out["y_bad"]) > 0:
            y = out["y_bad"].astype(int)
            pred = (out["p_bad"] >= thr_bad).astype(int)
            cm = confusion_matrix(y, pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Pred bad")
            plt.ylabel("True bad")
            plt.title("Confusion Matrix — Bad Head")
            plt.tight_layout()
            plt.savefig(os.path.join(model_folder, "cm_bad.png"), dpi=250, bbox_inches="tight")
            plt.close()

        # vowel cm on valid only
        if len(out["y_e"]) > 0:
            y = out["y_e"].astype(int)
            pred = (out["p_e"] >= thr_e).astype(int)
            cm = confusion_matrix(y, pred)
            plt.figure(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.xlabel("Pred /e/")
            plt.ylabel("True /e/")
            plt.title("Confusion Matrix — Vowel Head (valid only)")
            plt.tight_layout()
            plt.savefig(os.path.join(model_folder, "cm_vowel.png"), dpi=250, bbox_inches="tight")
            plt.close()

    return model
