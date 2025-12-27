import torch
import pickle
import os
import torch.nn as nn
from troncamento_datasets import SegmentPairDataset
import random
import tqdm
from model import uncertainty
import pandas as pd

def pretrain_one_epoch(pretrain_dataset, model, optimizer, criterion, device, pretrain_ckpt):
    loader = torch.utils.data.DataLoader(
        pretrain_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )

    model.train()
    total_loss = 0
    pbar = tqdm.tqdm(loader)

    for x, phone_ids, y in pbar:
        x = x.to(device)
        phone_ids = phone_ids.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x, phone_ids)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        break

        pbar.set_description(f"loss = {total_loss / (pbar.n + 1):.4f}")

    print(f"Training completed. Loss = {total_loss / len(loader):.4f}")
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

def select_uncertain_samples(
    model,
    target_basedataset,
    random_sample=100,
    k=50,
    device="cpu"
):
    model.eval()
    scored = []
    target_dataset = SegmentPairDataset(target_basedataset)

    if random_sample is not None and random_sample < len(target_basedataset):
        pool_indices = random.sample(range(len(target_basedataset)), random_sample)
    else:
        pool_indices = list(range(len(target_basedataset)))

    with torch.no_grad():
        for idx in tqdm.tqdm(pool_indices):
            x, phone_ids, y = target_dataset[idx]
            
            
            logit = model(x.unsqueeze(0).to(device), phone_ids.unsqueeze(0).to(device)).squeeze(0)

            prob = torch.sigmoid(logit).item()
            u = uncertainty(logit).item()

            scored.append({
                "index": idx,
                "logit": logit.item(),
                "prob": prob,
                "uncertainty": u
            })

    # sort by uncertainty (descending)
    scored.sort(key=lambda d: d["uncertainty"], reverse=True)

    # select top-k
    selected = scored[:k]

    for s in selected:
        target_basedataset.add_data_to_gold(s["index"], label=None)

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