import torch
import pickle
import os
import torch.nn as nn
from troncamento_datasets import SegmentPairDataset
import random
import tqdm
import pandas as pd

from torch.utils.data import random_split, DataLoader

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
    TP = FP = FN = TN = 0

    for x, phone_ids, silver_label, y in loader:
        x = x.to(device)
        phone_ids = phone_ids.to(device)
        silver_label = silver_label.to(device)
        y = y.to(device)

        logits = model(x, phone_ids, silver_label)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)

        preds = (logits > 0).long()

        TP += ((preds == 1) & (y == 1)).sum().item()
        FP += ((preds == 1) & (y == 0)).sum().item()
        FN += ((preds == 0) & (y == 1)).sum().item()
        TN += ((preds == 0) & (y == 0)).sum().item()

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    acc = (TP + TN) / (TP + TN + FP + FN)

    return {
        "loss": total_loss / (TP + TN + FP + FN),
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
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
    best_val_loss = float("inf")
    patience_counter = 0
    for x, phone_ids, silver_label, y in pbar:
        step += 1
        x = x.to(device)
        phone_ids = phone_ids.to(device)
        silver_label = silver_label.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x, phone_ids, silver_label)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        clamp_silver_column(model)

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

def train_on_gold_dataset(model_class, model_folder, base_dataset, device="cpu", pretained_ckpt="model_ckpt/prepretrain.pt", do_eval=True, train_type="pretrain"):
    os.makedirs(model_folder, exist_ok=True)
    model = model_class().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    if pretained_ckpt is not None:
        model.load_state_dict(torch.load(pretained_ckpt))

    train_gold_dataset, train_unique_ids = base_dataset.return_gold_dataset(dataset_type=train_type)
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
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    if do_eval:
        val_loader = torch.utils.data.DataLoader(
            SegmentPairDataset(val_gold_dataset, dataset_type="gold"),
            batch_size=1,
            shuffle=True,
            num_workers=0
        )

    model.train()
    total_loss = 0

    n_skipped = 0
    val_losses = []
    val_accs = []
    val_f1s = []
    for (x, phone_ids, silver_label, y), unique_id in tqdm.tqdm(zip(loader, train_unique_ids), total=len(train_gold_dataset)):
        if y==-1:
            n_skipped += 1

        else:
            x = x.to(device)
            phone_ids = phone_ids.to(device)
            silver_label = silver_label.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x, phone_ids, silver_label)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if do_eval:
            # eval
            val_stats = evaluate(model, val_loader, criterion, device)
            val_losses.append(val_stats["loss"])
            val_accs.append(val_stats["acc"])
            val_f1s.append(val_stats["f1"])

    print(f"Trained on {len(train_gold_dataset) - n_skipped} gold samples: loss = {total_loss / (len(train_gold_dataset) - n_skipped):.4f}")
    if n_skipped > 0: print(f"Skipped {n_skipped} samples.")

    # save model checkpoint
    ckpt_path = os.path.join(model_folder, f"{train_type}.pt")
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
            for x, phone_ids, silver_label, y in tqdm.tqdm(val_loader):
                if y==-1:
                    continue
                x = x.to(device)
                phone_ids = phone_ids.to(device)
                silver_label = silver_label.to(device)
                y = y.to(device)

                logits = model(x, phone_ids, silver_label)

                all_logits.append(logits.item())
                all_labels.append(y.item())

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

        # plot confusion matrix
        all_preds = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for x, phone_ids, silver_label, y in tqdm.tqdm(val_loader):
                x = x.to(device)
                phone_ids = phone_ids.to(device)
                silver_label = silver_label.to(device)
                y = y.to(device)

                logits = model(x, phone_ids, silver_label)
                preds = (logits > 0).long()

                all_preds.append(preds.item())
                all_labels.append(y.item())
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

            x, phone_ids, silver_label, y = target_dataset[idx]

            logit = model(
                x.unsqueeze(0).to(device),
                phone_ids.unsqueeze(0).to(device),
                silver_label.unsqueeze(0).to(device)
            ).squeeze(0)

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