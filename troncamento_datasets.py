import torch
import pandas as pd
from IPython.display import Audio
import numpy as np
import librosa
import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import os


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, df, dataset_type, return_player=False):

        self.embed_dim = 1024

        self.phones = sorted(set(list(df["preceding_phone"].unique()) + list(df["target_phone"].unique()) + [phone for phone in df["following_phone"].unique() if pd.notna(phone)]))
        self.phone2id = {p: i for i, p in enumerate(self.phones)}
        self.phone2id["<NULL>"] = len(self.phone2id)
        self.phone2id["<UNKNOWN>"] = len(self.phone2id)

        assert dataset_type in ("pre_train", "target", "all")
        if dataset_type == "pre_train":
            df = df[df["type"] == "non_troncamento"].reset_index(drop=True)
        elif dataset_type == "target":
            df = df[df["type"] == "potential_troncamento"].reset_index(drop=True)
        self.df = df

        self.unique_id2idx = {}
        self.idx2unique_id = {}
        for idx in range(len(self.df)):
            unique_id = self.get_unique_id(idx)
            self.unique_id2idx[unique_id] = idx
            self.idx2unique_id[idx] = unique_id
        
        self._refresh_gold_labels()

        self.return_player = return_player

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        ).to(self.device)

        self.model.eval()


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        start_time = row["preceding_phone_start_time"]
        if pd.isna(row["following_phone_start_time"]) or pd.isna(row["following_phone_end_time"]):
            end_time = row["target_phone_end_time"]
            following_phone = "<NULL>"
        else:
            end_time = row["following_phone_end_time"]
            following_phone = row["following_phone"]

        
        # embed = self._to_embed(row["id"],
        #                        row["mp3_path"],
        #                        start_time,
        #                        end_time)
        hidden = self._to_hidden(row["id"],
                                 row["mp3_path"],
                                 start_time,
                                 end_time)

        data = {
            "word": row["word"],
            "type": row["type"],

            "target_phone": row["target_phone"],
            "target_phone_id": self.phone2id["<UNKNOWN>"],
            # "target_phone_id": self.phone2id[row["target_phone"]],

            "preceding_phone": row["preceding_phone"],
            "preceding_phone_id": self.phone2id[row["preceding_phone"]],

            "following_phone": following_phone,
            "following_phone_id": self.phone2id[following_phone],

            # "embed": embed,
            "hidden": hidden,
        }
        data = self._add_heuristic_label(data)
        data["dataset_index"] = idx
        unique_id = self.get_unique_id(idx)
        if unique_id in self.gold_labels["label"]:
            data["gold_label"] = self.gold_labels["label"][unique_id]
        else:
            data["gold_label"] = None
        return data

    @torch.no_grad()
    def predict_sample_label(
        self,
        row_idx: int,
        model,
        thr_bad: float = 0.5,
        thr_e: float = 0.5,
        device=None,
        refresh_gold: bool = False,
        prefer_gold: bool = False,
    ):
        """
        Predict label in {1, 0, -1} for a given row_idx using MisalignmentDetector.

        Rule:
        if sigmoid(logit_bad) >= thr_bad -> -1
        elif sigmoid(logit_e) >= thr_e   ->  1
        else                             ->  0

        If prefer_gold=True and a gold label exists, return that instead.

        Returns:
        (pred_label:int, info:dict)
        """
        if not (0 <= row_idx < len(self.df)):
            raise IndexError(f"row_idx {row_idx} out of range (0..{len(self.df)-1})")

        if refresh_gold:
            self._refresh_gold_labels()

        # Optional: override with gold label
        if prefer_gold:
            uid = self.get_unique_id(row_idx)
            if uid in self.gold_labels["label"]:
                lab = self.gold_labels["label"][uid]
                if pd.notna(lab):
                    return int(lab), {"source": "gold", "unique_id": uid}

        model.eval()
        if device is None:
            device = next(model.parameters()).device

        sample = self[row_idx]  # uses __getitem__ -> includes hidden + phone ids + silver_label
        hidden_seq = sample["hidden"]  # (T, D)
        T = hidden_seq.shape[0]

        # mask: all frames valid (since your saved_hiddens are variable-length, not padded here)
        mask = torch.ones(T, dtype=torch.bool)

        # phone_ids: your model expects 2 phones (preceding + following) based on phone_emb_dim*2
        phone_ids = torch.tensor(
            [sample["preceding_phone_id"], sample["following_phone_id"]],
            dtype=torch.long
        )

        silver = torch.tensor(sample["silver_label"], dtype=torch.float32)

        # add batch dim
        hidden_seq = hidden_seq.unsqueeze(0).to(device)      # (1, T, D)
        mask = mask.unsqueeze(0).to(device)                  # (1, T)
        phone_ids = phone_ids.unsqueeze(0).to(device)        # (1, 2)
        silver = silver.unsqueeze(0).to(device)              # (1,)

        logit_e, logit_bad, attn = model(hidden_seq, mask, phone_ids, silver)

        p_e = torch.sigmoid(logit_e).item()
        p_bad = torch.sigmoid(logit_bad).item()

        if p_bad >= thr_bad:
            pred = -1
        elif p_e >= thr_e:
            pred = 1
        else:
            pred = 0

        info = {
            "source": "model",
            "p_e": p_e,
            "p_bad": p_bad,
            "thr_e": thr_e,
            "thr_bad": thr_bad,
            "unique_id": self.get_unique_id(row_idx),
        }
        return pred, info



    
    def _refresh_gold_labels(self):
        if not os.path.isfile("gold_labels.csv"):
            gold_labels = {
                "word": [],
                "unique_id": [],
                "label": []
            }
            pd.DataFrame(gold_labels).to_csv("gold_labels.csv", index=False)

        if not os.path.isfile("current_gold_labels.csv"):
            current_gold_labels = {
                "word": [],
                "unique_id": [],
                "label": []
            }
            pd.DataFrame(current_gold_labels).to_csv("current_gold_labels.csv", index=False)
        all_unique_ids = self.unique_id2idx.keys()
        self.gold_labels = pd.read_csv("gold_labels.csv").set_index("unique_id")
        self.gold_labels = self.gold_labels[self.gold_labels.index.isin(all_unique_ids)]
        self.gold_labels["order"] = range(len(self.gold_labels))
        self.gold_labels = self.gold_labels.to_dict()
        self.gold_labels["label_type"] = {}
        for uid in self.gold_labels["label"].keys():
            self.gold_labels["label_type"][uid] = "pretrain"
        
        current_gold_labels = pd.read_csv("current_gold_labels.csv").set_index("unique_id")
        current_gold_labels = current_gold_labels[current_gold_labels.index.isin(all_unique_ids)]
        current_gold_labels["order"] = range(len(current_gold_labels["label"]))
        current_gold_labels = current_gold_labels.to_dict()
        current_gold_labels["label_type"] = {}
        for uid in current_gold_labels["label"].keys():
            current_gold_labels["label_type"][uid] = "current"
        
        self.gold_labels["word"].update(current_gold_labels["word"])
        self.gold_labels["label"].update(current_gold_labels["label"])
        self.gold_labels["label_type"].update(current_gold_labels["label_type"])
        self.gold_labels["order"].update(current_gold_labels["order"])
        
        self.gold_labels["is_val"] = {}
        import random
        random.seed(42)
        # positive_unique_ids = [uid for uid, label in self.gold_labels["label"].items() if label == 1 and self.gold_labels["label_type"][uid]=="pretrain"]
        # negative_unique_ids = [uid for uid, label in self.gold_labels["label"].items() if label == 0 and self.gold_labels["label_type"][uid]=="pretrain"]
        # bad_unique_ids = [uid for uid, label in self.gold_labels["label"].items() if label == -1 and self.gold_labels["label_type"][uid]=="pretrain"]
        # min_n = min(len(positive_unique_ids), len(negative_unique_ids), 40)
        # val_positive_unids = random.sample(positive_unique_ids, k=max(min_n, int(len(negative_unique_ids)*0.6)))
        # val_negative_unids = random.sample(negative_unique_ids, k=max(min_n, int(len(negative_unique_ids)*0.6)))
        # val_bad_unids = random.sample(bad_unique_ids, k=int(len(bad_unique_ids)*0.6))
        # for i, uid in enumerate(self.gold_labels["label"].keys()):
        #     self.gold_labels["is_val"][uid] = (uid in val_positive_unids+val_negative_unids+val_bad_unids)
        val_unique_ids = random.sample(list(self.gold_labels["label"].keys()), k=int(len(self.gold_labels["label"])*0.2))
        for i, uid in enumerate(self.gold_labels["label"].keys()):
            self.gold_labels["is_val"][uid] = (uid in val_unique_ids)

    
    def put_file_to_folder(self, unique_id, folder_path, tgrd_fs_folder):
        import shutil
        import tgt
        _id = unique_id.split("_")[0]
        for _, row in self.df[self.df["id"] == int(_id)].iterrows():
            if unique_id == f"{row.id}_{row.target_phone}_{row.target_phone_start_time}_{row.target_phone_end_time}":
                break
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        f_end = row["mp3_path"].split(".")[-1]
        shutil.copyfile(row["mp3_path"], os.path.join(folder_path, f"{unique_id}.{f_end}"))

        y, sr = librosa.load(row["mp3_path"])
        duration = len(y) / sr
        import tgt
        import re
        import tempfile
        def read_textgrid_safely(path, encoding="utf-8", include_empty_intervals=False):
            try:
                return tgt.read_textgrid(path, encoding=encoding,
                                        include_empty_intervals=include_empty_intervals)
            except IndexError:
                # Load and normalize the specific bad pattern
                with open(path, "r", encoding=encoding, errors="replace") as f:
                    s = f.read()

                # Put 'class = ...' on a new line after 'item [k]:'
                s2 = re.sub(r'(item \[\d+\]:)\s+(class\s+=\s+)', r'\1\n        \2', s)

                # Write to a temp file and retry
                with tempfile.NamedTemporaryFile("w", suffix=".TextGrid", delete=False, encoding=encoding) as tf:
                    tf.write(s2)
                    tmp_path = tf.name

                return tgt.read_textgrid(tmp_path, encoding=encoding,
                                        include_empty_intervals=include_empty_intervals)

        tgrd_f = os.path.join(tgrd_fs_folder, f"common_voice_it_{row['id']}.TextGrid")
        tgrd = read_textgrid_safely(tgrd_f)
        tier = tgt.IntervalTier(name="target", start_time=0.0, end_time=duration)
        tier.add_interval(tgt.Interval(row['target_phone_start_time'], row['target_phone_end_time'], "TARGET"))
        tgrd.add_tier(tier)
        tgt.io.write_to_file(tgrd, os.path.join(folder_path, f"{unique_id}.TextGrid"))
    
    def delete_file_from_folder(self, unique_id, folder_path):
        import os
        
        mp3_path = os.path.join(folder_path, f"{unique_id}.mp3")
        wav_path = os.path.join(folder_path, f"{unique_id}.wav")
        tgrd_path = os.path.join(folder_path, f"{unique_id}.TextGrid")

        if os.path.isfile(mp3_path):
            os.remove(mp3_path)
        if os.path.isfile(wav_path):
            os.remove(wav_path)
        if os.path.isfile(tgrd_path):
            os.remove(tgrd_path)
    
    def return_gold_dataset(self, trained_samples=[], dataset_type="pretrain", upsample=True, upsample_ratio=1):
        self._refresh_gold_labels()
        assert dataset_type in ("pretrain", "train", "val")
        allowed_unique_ids = []
        if dataset_type=="val":
            allowed_unique_ids = [uid for uid, is_val in self.gold_labels["is_val"].items() if is_val]
        elif dataset_type=="train":
            allowed_unique_ids = [uid for uid, is_val in self.gold_labels["is_val"].items() if not is_val and self.gold_labels["label_type"][uid]=="current"]
        elif dataset_type=="pretrain":
            allowed_unique_ids = [uid for uid, is_val in self.gold_labels["is_val"].items() if not is_val and self.gold_labels["label_type"][uid]=="pretrain"]

        
        positive_unique_ids = []
        negative_unique_ids = []
        bad_unique_ids = []
        for unique_id in allowed_unique_ids:
            if pd.notna(self.gold_labels["label"][unique_id]) and unique_id not in trained_samples:
                if self.gold_labels["label"][unique_id] == 1:
                    positive_unique_ids.append(unique_id)
                elif self.gold_labels["label"][unique_id] == 0:
                    negative_unique_ids.append(unique_id)
                elif self.gold_labels["label"][unique_id] == -1:
                    bad_unique_ids.append(unique_id)

        import random

        positive_count = len(positive_unique_ids)
        negative_count = len(negative_unique_ids)
        bad_count = len(bad_unique_ids)
        target_num = max(positive_count, negative_count)
        if dataset_type!="val" and upsample:
            print(f"Positive samples: {positive_count}, Negative samples: {negative_count}, Target samples per class: {target_num}")
            if 0 < positive_count < target_num:
                diff = int((target_num - positive_count) * upsample_ratio)
                sampled_unique_ids = random.choices(positive_unique_ids, k=diff)
                positive_unique_ids.extend(sampled_unique_ids)
                print(f"Upsampled positive samples by {diff}")

            if 0 < negative_count < target_num:
                diff = int((target_num - negative_count) * upsample_ratio)
                sampled_unique_ids = random.choices(negative_unique_ids, k=diff)
                negative_unique_ids.extend(sampled_unique_ids)
                print(f"Upsampled negative samples by {diff}")
            
            if 0 < bad_count < target_num:
                diff = int((target_num - bad_count) * upsample_ratio)
                sampled_unique_ids = random.choices(bad_unique_ids, k=diff)
                bad_unique_ids.extend(sampled_unique_ids)
                print(f"Upsampled bad samples by {diff}")

        else:
            print(f"Positive samples: {positive_count}, Negative samples: {negative_count}, Bad samples: {bad_count}")

        unique_ids = positive_unique_ids + negative_unique_ids + bad_unique_ids

        ordered_unique_ids = sorted(unique_ids, key=lambda x: self.gold_labels["order"][x])
        gold_indices = [self.unique_id2idx[uid] for uid in ordered_unique_ids]

        return torch.utils.data.Subset(self, gold_indices), unique_ids

    def get_unique_id(self, dataset_index):
        if dataset_index in self.idx2unique_id:
            return self.idx2unique_id[dataset_index]
        row = self.df.iloc[dataset_index]
        unique_id = f"{row['id']}_{row['target_phone']}_{row['target_phone_start_time']}_{row['target_phone_end_time']}"
        return unique_id

    def _add_heuristic_label(self, data):
        data["silver_label"] = [0, 1][data["target_phone"] == "e"]
        
        return data

    def add_data_to_gold(self, dataset_index, label=None):
        row = self.df.iloc[dataset_index]
        unique_id = self.get_unique_id(dataset_index)
        
        gold_labels = pd.read_csv("gold_labels.csv")
        if unique_id in gold_labels["unique_id"].values:
            gold_labels.loc[gold_labels["unique_id"] == unique_id, "label"] = label
        else:
            new_entry = {
                "word": row["word"],
                "unique_id": unique_id,
                "label": label
            }
            gold_labels = pd.concat([gold_labels, pd.DataFrame([new_entry])], ignore_index=True)
        
        gold_labels.to_csv("gold_labels.csv", index=False)
        self._refresh_gold_labels()

    def _to_embed(self, id_num, mp3_path, start_time, end_time, min_len=400):
        saved_embed_path = f"saved_embeds/{id_num}_{start_time}_{end_time}.npy"
        if os.path.isfile(saved_embed_path):
            embed = np.load(saved_embed_path)
            return torch.tensor(embed, dtype=torch.float32)

        segment, sr, _ = self._get_audio_segment(
            id_num=id_num,
            mp3_path=mp3_path,
            start_time=start_time,
            end_time=end_time,
            save_temp_files=False,
        )

        if sr != 16000:
            segment = librosa.resample(segment, orig_sr=sr, target_sr=16000)

        segment = np.asarray(segment, dtype=np.float32)

        if len(segment) < min_len:
            pad_width = min_len - len(segment)
            segment = np.pad(segment, (0, pad_width), mode="constant")

        with torch.no_grad():
            inputs = self.feature_extractor(
                segment,
                sampling_rate=16000,
                return_tensors="pt"
            )

            outputs = self.model(inputs.input_values.to(self.device))
            embed = outputs.last_hidden_state.mean(dim=1).squeeze(0)

        np.save(saved_embed_path, embed.cpu().numpy())
        return embed.cpu()

    def _to_hidden(self, id_num, mp3_path, start_time, end_time, min_len=400):
        # cache file: variable length T, fixed dim 1024
        saved_hidden_path = f"saved_hiddens/{id_num}_{start_time}_{end_time}.npy"
        os.makedirs("saved_hiddens", exist_ok=True)

        if os.path.isfile(saved_hidden_path):
            arr = np.load(saved_hidden_path)  # shape (T, 1024)
            return torch.tensor(arr, dtype=torch.float32)

        segment, sr, _ = self._get_audio_segment(
            id_num=id_num,
            mp3_path=mp3_path,
            start_time=start_time,
            end_time=end_time,
            save_temp_files=False,
        )

        if sr != 16000:
            import librosa
            segment = librosa.resample(segment, orig_sr=sr, target_sr=16000)

        segment = np.asarray(segment, dtype=np.float32)

        if len(segment) < min_len:
            pad_width = min_len - len(segment)
            segment = np.pad(segment, (0, pad_width), mode="constant")

        with torch.no_grad():
            inputs = self.feature_extractor(
                segment,
                sampling_rate=16000,
                return_tensors="pt"
            )
            outputs = self.model(inputs.input_values.to(self.device))
            hidden = outputs.last_hidden_state.squeeze(0)  # (T, 1024)

        # Save as float16 to reduce disk (you can switch to float32 if you want)
        np.save(saved_hidden_path, hidden.cpu().numpy().astype(np.float16))
        return hidden.cpu()


    def _get_audio_segment(self, id_num, mp3_path, start_time, end_time, save_temp_files=False):
        if pd.isna(start_time) or pd.isna(end_time):
            return None, None, None

        target_id = f"{id_num}"
        audio, sr = librosa.load(mp3_path, sr=None)
        
        if save_temp_files:
            import shutil
            shutil.copyfile(mp3_path, "temp.mp3")
            shutil.copyfile(f"it_vxc_textgrids17_acoustic17/common_voice_it_{target_id}.TextGrid", "temp.TextGrid")

        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)

        segment = audio[start_idx:end_idx]

        # return segment, sr, Audio(segment, rate=sr)
        return segment, sr, None

class SegmentPairDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset, dataset_type="silver"):
        self.data = raw_dataset
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.data)
    
    def _get_features(self, ex):
        e = ex["embed"]

        features = torch.cat([
            e
        ])

        return features

    def __getitem__(self, idx):
        ex = self.data[idx]

        # features = self._get_features(ex)
        hidden = ex["hidden"]

        phone_ids = torch.tensor([
            ex["preceding_phone_id"],
            # ex["target_phone_id"],
            ex["following_phone_id"],
        ])
        silver_label = torch.tensor(ex["silver_label"], dtype=torch.float32)

        label = ex[f"{self.dataset_type}_label"]
        

        bad_y = 1.0 if label == -1 else 0.0
        vowel_mask = 0.0 if label == -1 else 1.0
        vowel_y = 0.0 if label == -1 else float(label)  # dummy 0.0 when masked

        return (
            hidden,
            phone_ids,
            silver_label,
            torch.tensor(vowel_y, dtype=torch.float32),
            torch.tensor(vowel_mask, dtype=torch.float32),
            torch.tensor(bad_y, dtype=torch.float32),
        )
        # vowel_label = None if label==-1 else label
        # bad_label = 1 if label==-1 else 0
        # # return features, phone_ids, torch.tensor(silver_label, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
        # return hidden, phone_ids, torch.tensor(silver_label, dtype=torch.float32), torch.tensor(vowel_label, dtype=torch.float32), torch.tensor(bad_label, dtype=torch.float32)