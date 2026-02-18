import torch
import pickle
import os
import torch.nn as nn
from troncamento_datasets import SegmentPairDataset
import random
import tqdm
import pandas as pd

from torch.utils.data import random_split, DataLoader

def collate_segment_batch(batch):
    hiddens, phone_ids_list, silver_list, vowel_y_list, vowel_mask_list, bad_y_list = zip(*batch)

    lengths = torch.tensor([h.size(0) for h in hiddens], dtype=torch.long)
    B = len(hiddens)
    T_max = int(lengths.max().item())
    D = int(hiddens[0].size(1))

    hidden_padded = torch.zeros(B, T_max, D, dtype=torch.float32)
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i, h in enumerate(hiddens):
        T = h.size(0)
        hidden_padded[i, :T] = h
        mask[i, :T] = True

    phone_ids = torch.stack(phone_ids_list, dim=0)  # (B,2)
    silver = torch.stack(silver_list, dim=0).view(B)

    vowel_y = torch.stack(vowel_y_list, dim=0).view(B)
    vowel_mask = torch.stack(vowel_mask_list, dim=0).view(B)
    bad_y = torch.stack(bad_y_list, dim=0).view(B)

    return hidden_padded, mask, phone_ids, silver, vowel_y, vowel_mask, bad_y

def clamp_silver_column(model):
    first = model.net[0]
    silver_col = first.weight.shape[1] - 1
    with torch.no_grad():
        first.weight[:, silver_col].zero_()

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
        num_workers=0,
        collate_fn=collate_segment_batch
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_segment_batch
    )

    return train_loader, val_loader


import torch

@torch.no_grad()
def evaluate(model, loader, crit_none, device, thr_bad=0.5, thr_e=0.5, lambda_e=1.0):
    """
    criterion: BCEWithLogitsLoss(reduction='none') or BCEWithLogitsLoss().

    We compute:
      - loss_bad on all
      - loss_e only where vowel_mask == 1
      - metrics_bad (acc/precision/recall/f1) on all
      - metrics_e   (acc/precision/recall/f1) on valid only
    """
    model.eval()

    total_n = 0
    total_loss = 0.0

    # Bad metrics (all samples)
    TPb = FPb = FNb = TNb = 0

    # Vowel metrics (valid only)
    TPe = FPe = FNe = TNe = 0
    total_valid = 0

    # Ensure we can mask per-example losses

    for hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y in loader:
        hidden_padded = hidden_padded.to(device)
        seq_mask = seq_mask.to(device)
        phone_ids = phone_ids.to(device)
        silver = silver.to(device)
        vowel_y = vowel_y.to(device)
        vowel_mask = vowel_mask.to(device)
        bad_y = bad_y.to(device)

        logit_e, logit_bad, _ = model(hidden_padded, seq_mask, phone_ids, silver)

        # ----- losses -----
        loss_bad_vec = crit_none(logit_bad, bad_y)             # (B,)
        loss_bad = loss_bad_vec.mean()

        # vowel loss only for valid
        valid = (vowel_mask > 0.5)  # bool mask
        if valid.any():
            loss_e_vec = crit_none(logit_e, vowel_y)           # (B,)
            loss_e = (loss_e_vec[valid]).mean()
        else:
            loss_e = 0.0 * loss_bad

        loss_all = loss_bad + lambda_e * loss_e

        B = bad_y.size(0)
        total_loss += loss.item() * B
        total_n += B

        # ----- bad metrics -----
        prob_bad = torch.sigmoid(logit_bad)
        pred_bad = (prob_bad >= thr_bad).long()
        yb = bad_y.long()

        TPb += ((pred_bad == 1) & (yb == 1)).sum().item()
        FPb += ((pred_bad == 1) & (yb == 0)).sum().item()
        FNb += ((pred_bad == 0) & (yb == 1)).sum().item()
        TNb += ((pred_bad == 0) & (yb == 0)).sum().item()

        # ----- vowel metrics (valid only) -----
        if valid.any():
            prob_e = torch.sigmoid(logit_e[valid])
            pred_e = (prob_e >= thr_e).long()
            ye = vowel_y[valid].long()

            TPe += ((pred_e == 1) & (ye == 1)).sum().item()
            FPe += ((pred_e == 1) & (ye == 0)).sum().item()
            FNe += ((pred_e == 0) & (ye == 1)).sum().item()
            TNe += ((pred_e == 0) & (ye == 0)).sum().item()
            total_valid += int(valid.sum().item())

    def prf(TP, FP, FN, TN):
        precision = TP / (TP + FP + 1e-8)
        recall    = TP / (TP + FN + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        acc       = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        return acc, precision, recall, f1

    acc_bad, prec_bad, rec_bad, f1_bad = prf(TPb, FPb, FNb, TNb)
    if total_valid > 0:
        acc_e, prec_e, rec_e, f1_e = prf(TPe, FPe, FNe, TNe)
    else:
        acc_e = prec_e = rec_e = f1_e = float("nan")

    return {
        "loss": total_loss / max(total_n, 1),

        "bad_acc": acc_bad,
        "bad_f1": f1_bad,
        "bad_precision": prec_bad,
        "bad_recall": rec_bad,

        "vowel_acc": acc_e,
        "vowel_f1": f1_e,
        "vowel_precision": prec_e,
        "vowel_recall": rec_e,

        "n_total": total_n,
        "n_valid_for_e": total_valid,
    }


import tqdm
import matplotlib.pyplot as plt

def pretrain_one_epoch(
    dataset,
    model,
    optimizer,
    crit_none,
    device,
    pretrain_ckpt,
    batch_size=4,
    lambda_e=1.0
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
    val_losses_bad = []
    val_accs_bad = []
    val_losses_vowel = []
    val_accs_vowel = []

    pbar = tqdm.tqdm(train_loader)
    step = -1
    best_val_loss = float("inf")
    patience_counter = 0

    for hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y in pbar:
        step += 1
        hidden_padded = hidden_padded.to(device)
        seq_mask = seq_mask.to(device)
        phone_ids = phone_ids.to(device)
        silver = silver.to(device)
        vowel_y = vowel_y.to(device)
        vowel_mask = vowel_mask.to(device)
        bad_y = bad_y.to(device)

        logit_e, logit_bad, _ = model(hidden_padded, seq_mask, phone_ids, silver)

        # ----- losses -----
        loss_bad_vec = crit_none(logit_bad, bad_y)             # (B,)
        loss_bad = loss_bad_vec.mean()

        # vowel loss only for valid
        valid = (vowel_mask > 0.5)  # bool mask
        if valid.any():
            loss_e_vec = crit_none(logit_e, vowel_y)           # (B,)
            loss_e = (loss_e_vec[valid]).mean()
        else:
            loss_e = 0.0 * loss_bad

        loss_all = loss_bad + lambda_e * loss_e


        loss_all.backward()
        optimizer.step()

        clamp_silver_column(model)

        # ---- validation every 1000 steps ----
        if step % 10 == 0 and step > 0:
            val_stats = evaluate(model, val_loader, crit_none, device)
            log_steps.append(step)

            val_losses_bad.append(val_stats["bad_loss"])
            val_accs_bad.append(val_stats["bad_acc"])
            val_losses_vowel.append(val_stats["vowel_loss"])
            val_accs_vowel.append(val_stats["vowel_acc"])

            model.train()  # switch back

            # ---- plot validation curves ----
            fig, ax1 = plt.subplots(figsize=(6, 4))

            ax1.plot(log_steps, val_losses_bad, label="Val Loss (Bad)")
            ax1.plot(log_steps, val_losses_vowel, label="Val Loss (Vowel)")
            ax1.set_xlabel("Training step")
            ax1.set_ylabel("Loss")

            ax2 = ax1.twinx()
            ax2.plot(log_steps, val_accs_bad, color="tab:orange", label="Val Accuracy (Bad)")
            ax2.plot(log_steps, val_accs_vowel, color="tab:green", label="Val Accuracy (Vowel)")
            ax2.set_ylabel("Accuracy")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)

            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(pretrain_ckpt), "pretrain_val_curve.png"))
            plt.close("all")
            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                best_state = model.state_dict()
                
            if patience_counter >= 20:
                print("Early stopping triggered.")
                model.load_state_dict(best_state)
                break
        if step % 100 == 0 and step > 0:
            torch.save(model.state_dict(), pretrain_ckpt)
            
    print(f"Training completed.")
    print("Checkpoint saved at ", pretrain_ckpt)
    torch.save(model.state_dict(), pretrain_ckpt)

import numpy as np
import torch
import matplotlib.pyplot as plt

import numpy as np
import torch
import matplotlib.pyplot as plt

@torch.no_grad()
def collect_probs_twohead(model, loader, device):
    """
    loader yields:
      hidden_padded, seq_mask, phone_ids, silver, vowel_y, vowel_mask, bad_y

    returns dict of numpy arrays:
      - p_bad, y_bad                 (all samples)
      - p_e,   y_e                   (valid samples only)
    """
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

        # --- bad (all samples) ---
        p_bad = torch.sigmoid(logit_bad)
        p_bad_all.append(p_bad.detach().cpu().numpy())
        y_bad_all.append(bad_y.detach().cpu().numpy())

        # --- vowel (valid only) ---
        valid = (vowel_mask > 0.5)
        if valid.any():
            p_e = torch.sigmoid(logit_e[valid])
            y_e = vowel_y[valid]
            p_e_all.append(p_e.detach().cpu().numpy())
            y_e_all.append(y_e.detach().cpu().numpy())

    out = {
        "p_bad": np.concatenate(p_bad_all) if len(p_bad_all) else np.array([]),
        "y_bad": np.concatenate(y_bad_all) if len(y_bad_all) else np.array([]),
        "p_e":   np.concatenate(p_e_all) if len(p_e_all) else np.array([]),
        "y_e":   np.concatenate(y_e_all) if len(y_e_all) else np.array([]),
    }
    return out


def plot_score_distributions(probs, ys, threshold=0.5, bins=40, title=None, savepath=None,
                             xlabel="Predicted probability P(y=1)"):
    probs = np.asarray(probs)
    ys = np.asarray(ys)

    p_pos = probs[ys == 1]
    p_neg = probs[ys == 0]

    plt.figure(figsize=(6,4))
    plt.hist(p_neg, bins=bins, range=(0,1), alpha=0.5, density=True, label="True negatives (y=0)")
    plt.hist(p_pos, bins=bins, range=(0,1), alpha=0.5, density=True, label="True positives (y=1)")
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold = {threshold:.2f}")

    plt.xlim(0,1)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        plt.close()
    else:
        plt.show()

def train_on_gold_dataset(model_class, model_folder, base_dataset, batch_size=1, device="cpu", pretained_ckpt="model_ckpt/prepretrain.pt", do_eval=True, train_type="pretrain", upsample=True, upsample_ratio=1, save_ckpt="train.pt"):
    os.makedirs(model_folder, exist_ok=True)
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    if pretained_ckpt is not None:
        model.load_state_dict(torch.load(pretained_ckpt))

    train_gold_dataset, train_unique_ids = base_dataset.return_gold_dataset(dataset_type=train_type, upsample=upsample, upsample_ratio=upsample_ratio)
    if len(train_gold_dataset) == 0:
        print("No gold samples found in the training set. Exiting.")
        return
    print(f"Training on {len(train_gold_dataset)} gold samples.")

    if do_eval:
        val_gold_dataset, val_unique_ids = base_dataset.return_gold_dataset(dataset_type="val")
        print(f"Validation on {len(val_gold_dataset)} gold samples.")

    segment_pair_dataset = SegmentPairDataset(train_gold_dataset, dataset_type="gold")

    loader = torch.utils.data.DataLoader(
        segment_pair_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_segment_batch
    )
    if do_eval:
        val_loader = torch.utils.data.DataLoader(
            SegmentPairDataset(val_gold_dataset, dataset_type="gold"),
            batch_size=1,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_segment_batch
        )

    model.train()
    total_loss = 0

    n_skipped = 0
    val_losses = []
    val_accs = []
    val_f1s = []

    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1000):

        for hidden_padded, attn_mask, phone_ids, silver_labels, labels in tqdm.tqdm(loader, total=len(loader)):
            for label in labels:
                assert label != -1
            hidden_padded = hidden_padded.to(device)
            attn_mask = attn_mask.to(device)
            phone_ids = phone_ids.to(device)
            silver_labels = silver_labels.to(device)
            labels = labels.to(device)

            logits, attn = model(hidden_padded, attn_mask, phone_ids, silver_labels)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if do_eval:
            # eval
            val_stats = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_stats["loss"])
            val_accs.append(val_stats["acc"])
            val_f1s.append(val_stats["f1"])

            if val_stats["loss"] < best_val_loss:
                best_val_loss = val_stats["loss"]
                best_state = model.state_dict()
                best_state_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                print("Early stopping triggered. Best validation loss: {:.4f} at epoch {}".format(best_val_loss, best_state_epoch))
                model.load_state_dict(best_state)
                break

    print(f"Trained on {len(train_gold_dataset)} gold samples: loss = {total_loss / len(loader):.4f}")

    # save model checkpoint
    ckpt_path = os.path.join(model_folder, save_ckpt)
    torch.save(model.state_dict(), ckpt_path)
    print("Model checkpoint saved at ", ckpt_path)

    if do_eval:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        # plot validation loss and accuracy curves
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(range(len(val_losses)), val_losses, label="Val Loss")
        ax1.set_xlabel("Evaluation step")
        ax1.set_ylabel("Loss")
        ax2 = ax1.twinx()
        ax2.plot(range(len(val_accs)), val_accs, color="tab:orange", label="Val Accuracy")
        ax2.plot(range(len(val_f1s)), val_f1s, color="tab:green", label="Val F1")
        ax2.set_ylabel("Accuracy / F1")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper left",
            bbox_to_anchor=(1.15, 1),
            frameon=False
        )

        plt.tight_layout()
        plt.savefig(os.path.join(model_folder, "val_curve.png"), dpi=300, bbox_inches="tight")
        plt.close("all")

        # plot ROC curve
        all_logits = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            # for x, phone_ids, silver_label, y in val_loader:
            #     if y==-1:
            #         continue
            #     x = x.to(device)
            #     phone_ids = phone_ids.to(device)
            #     silver_label = silver_label.to(device)
            #     y = y.to(device)

            #     logits = model(x, phone_ids, silver_label)

            for hidden_padded, attn_mask, phone_ids, silver_labels, labels in val_loader:

                hidden_padded = hidden_padded.to(device)
                attn_mask = attn_mask.to(device)
                phone_ids = phone_ids.to(device)
                silver_labels = silver_labels.to(device)
                labels = labels.to(device)

                logits, attn = model(hidden_padded, attn_mask, phone_ids, silver_labels)
                loss = criterion(logits, labels)

                all_logits.append(logits.item())
                all_labels.append(labels.item())

        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(all_labels, all_logits)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(model_folder, "roc_curve.png"), dpi=300, bbox_inches="tight")
        plt.close()

        probs, ys = collect_probs(model, val_loader, device)
        plot_score_distributions(
            probs, ys,
            threshold=0.5,
            savepath=os.path.join(model_folder, "score_distributions.png")
        )

        # plot confusion matrix
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            # for x, phone_ids, silver_label, y in val_loader:
            #     x = x.to(device)
            #     phone_ids = phone_ids.to(device)
            #     silver_label = silver_label.to(device)
            #     y = y.to(device)

            #     logits = model(x, phone_ids, silver_label)


            for hidden_padded, attn_mask, phone_ids, silver_labels, labels in val_loader:
                
                hidden_padded = hidden_padded.to(device)
                attn_mask = attn_mask.to(device)
                phone_ids = phone_ids.to(device)
                silver_labels = silver_labels.to(device)
                labels = labels.to(device)

                logits, attn = model(hidden_padded, attn_mask, phone_ids, silver_labels)
                loss = criterion(logits, labels)


                preds = (logits > 0).long()

                all_preds.append(preds.item())
                all_labels.append(labels.item())
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title("Confusion Matrix on Gold Validation Set")
        plt.savefig(os.path.join(model_folder, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close()




def uncertainty_bias_zero(logit, alpha=1.0):
    p = torch.sigmoid(logit)
    eps = 1e-8
    entropy = -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)
    weight = (p < 0.5) ** alpha
    return entropy * weight, entropy

def uncertainty_bias_one(logits, alpha=1.0):
    p = torch.sigmoid(logits)
    eps = 1e-8
    entropy = -p * torch.log(p + eps) - (1 - p) * torch.log(1 - p + eps)
    weight = (p > 0.5) ** alpha
    return entropy * weight, entropy

def select_uncertain_pos_neg(logits, k, most_certain=False):
    """
    logits: Tensor of shape (N,) or (N, 1)
    k: number of samples per group

    Returns:
        pos_idx: indices of top-k uncertain predicted-positive samples
        neg_idx: indices of top-k uncertain predicted-negative samples
    """
    logits = logits.view(-1)
    p = torch.sigmoid(logits)

    # predicted labels
    preds = (p >= 0.5)

    # uncertainties
    # positives: alpha = 0 (pure entropy)
    # uncert_pos, uncert = uncertainty_bias_zero(logits, alpha=1)

    # negatives: alpha = 1 (biased toward 0)
    # uncert_neg, _ = uncertainty_bias_one(logits, alpha=1)

    _, uncert = uncertainty_bias_one(logits, alpha=0)

    # split indices
    pos_indices = torch.where(preds)[0]
    neg_indices = torch.where(~preds)[0]

    # uncertainties per group
    pos_uncert = uncert[pos_indices]
    neg_uncert = uncert[neg_indices]

    # top-k per group
    k_pos = min(k, len(pos_indices))
    k_neg = min(k, len(neg_indices))

    _, pos_topk_idx = torch.topk(pos_uncert, k_pos, largest=not most_certain)
    _, neg_topk_idx = torch.topk(neg_uncert, k_neg, largest=not most_certain)

    pos_selected = pos_indices[pos_topk_idx]
    neg_selected = neg_indices[neg_topk_idx]

    return pos_selected, neg_selected, uncert

def select_uncertain_samples(
    model,
    target_basedataset,
    random_sample=100,
    k=50,
    device="cpu",
    log_csv_path="uncertainty_log.csv",
    most_certain=False
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

            # x, phone_ids, silver_label, y = target_dataset[idx]


            # logit = model(
            #     x.unsqueeze(0).to(device),
            #     phone_ids.unsqueeze(0).to(device),
            #     silver_label.unsqueeze(0).to(device)
            # ).squeeze(0)

            hidden, phone_ids, silver_label, y = target_dataset[idx]
            hidden_b = hidden.unsqueeze(0).to(device)          # (1, T, 1024)
            mask_b   = torch.ones(1, hidden.size(0), dtype=torch.bool, device=device)  # (1, T)

            phone_b  = phone_ids.unsqueeze(0).to(device)       # (1, 2)
            silver_b = silver_label.view(1).to(device)         # (1,)

            logits, attn = model(hidden_b, mask_b, phone_b, silver_b)  # logits: (1,)
            logit = logits.squeeze(0)

            all_logits.append(logit)
            all_indices.append(idx)

    logits = torch.stack(all_logits)
    probs = torch.sigmoid(logits.detach()).float().cpu().numpy() 

    pos_idx, neg_idx, uncert = select_uncertain_pos_neg(logits, k=k, most_certain=most_certain)

    selected_dataset_indices = (
        [all_indices[i] for i in pos_idx.tolist()] +
        [all_indices[i] for i in neg_idx.tolist()]
    )

    for idx in selected_dataset_indices:
        target_basedataset.add_data_to_gold(idx, label=None)
    
    if os.path.exists(log_csv_path):
        log_df = pd.read_csv(log_csv_path)
        idx = len(log_df)
    else:
        log_df = pd.DataFrame(columns=[
            "idx",
            "mean_uncertainty",
            "selected_mean_uncertainty",
        ])
        idx = 0
    log_df = pd.concat([log_df, pd.DataFrame([{
        "idx": idx,
        "mean_uncertainty": uncert.mean().item(),
        "selected_mean_uncertainty": uncert[pos_idx].mean().item(),
    }])], ignore_index=True)
    log_df.to_csv(log_csv_path, index=False)

    import seaborn as sns
    import matplotlib.pyplot as plt
    import glob

    uncert = uncert.detach().cpu()
    pos_idx = pos_idx.detach().cpu()
    neg_idx = neg_idx.detach().cpu()

    order = torch.argsort(uncert)          # indices that sort uncert ascending
    uncert_sorted = uncert[order]

    # map original indices -> positions in the sorted array
    # inv_order[orig_i] = position of orig_i in sorted order
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(order.numel())

    pos_x = inv_order[pos_idx]
    neg_x = inv_order[neg_idx]

    plt.plot(uncert_sorted.numpy())
    plt.scatter(pos_x.numpy(), uncert_sorted[pos_x].numpy(), label="selected pos")
    plt.scatter(neg_x.numpy(), uncert_sorted[neg_x].numpy(), label="selected neg")
    plt.legend()
    plot_idx = len(glob.glob("log_plots/uncertainty_*.png"))
    plt.savefig(f"log_plots/uncertainty_{plot_idx}.png")
    plt.close()

    hist_idx = len(glob.glob("log_plots/prob_hist_*.png"))

    plt.figure()
    plt.hist(
        probs,
        bins=30,            # adjust as you like
        density=True,       # y-axis = density
        edgecolor="black"
    )
    plt.xlabel("Predicted probability (sigmoid(logit))")
    plt.ylabel("Density")
    plt.title("Predicted probability distribution")
    plt.tight_layout()
    plt.savefig(f"log_plots/prob_hist_{hist_idx}.png", dpi=200)
    plt.close()

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