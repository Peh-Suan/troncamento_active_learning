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

        self.phones = sorted(df["preceding_phone"].unique())
        self.phone2id = {p: i for i, p in enumerate(self.phones)}
        assert dataset_type in ("pre_train", "target", "all")
        if dataset_type == "pre_train":
            df = df[df["type"] == "non_troncamento"].reset_index(drop=True)
        elif dataset_type == "target":
            df = df[df["type"] == "potential_troncamento"].reset_index(drop=True)
        self.df = df
        
        self._refresh_gold_labels()

        self.return_player = return_player

        self.device = "cpu"

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
        target_segment, sr, target_player = self._get_audio_segment(
            id_num=row["id"],
            mp3_path=row["mp3_path"],
            start_time=row["target_phone_start_time"],
            end_time=row["target_phone_end_time"],
            save_temp_files=False,
        )
        preceding_segment, _, preceding_player = self._get_audio_segment(
            id_num=row["id"],
            mp3_path=row["mp3_path"],
            start_time=row["preceding_phone_start_time"],
            end_time=row["preceding_phone_end_time"],
            save_temp_files=False,
        )

        data = {
            "word": row["word"],
            "type": row["type"],
            "target_phone": row["target_phone"],
            "target_phone_id": self.phone2id[row["target_phone"]],
            "target_audio": target_segment,
            "target_embed": self._to_embed(target_segment, sr),
            "target_player": target_player,
            "preceding_phone": row["preceding_phone"],
            "preceding_phone_id": self.phone2id[row["preceding_phone"]],
            "preceding_audio": preceding_segment,
            "preceding_embed": self._to_embed(preceding_segment, sr),
            "preceding_player": preceding_player,
            "sr": sr
        }
        data = self._add_heuristic_label(data)
        data["dataset_index"] = idx
        unique_id = self.get_unique_id(idx)
        if unique_id in self.gold_labels["label"]:
            data["gold_label"] = self.gold_labels["label"][unique_id]
        else:
            data["gold_label"] = None
        return data
    def _refresh_gold_labels(self):
        if not os.path.isfile("gold_labels.csv"):
            gold_labels = {
                "word": [],
                "unique_id": [],
                "label": []
            }
            pd.DataFrame(gold_labels).to_csv("gold_labels.csv", index=False)
        self.gold_labels = pd.read_csv("gold_labels.csv").set_index("unique_id").to_dict()
    
    def put_file_to_folder(self, unique_id, folder_path, tgrd_fs_folder):
        import shutil
        import tgt
        _id = unique_id.split("_")[0]
        for _, row in self.df[self.df["id"] == int(_id)].iterrows():
            if unique_id == f"{row.id}_{row.target_phone}_{row.target_phone_start_time}_{row.target_phone_end_time}":
                break
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        shutil.copyfile(row["mp3_path"], os.path.join(folder_path, f"{unique_id}.mp3"))

        y, sr = librosa.load(row["mp3_path"])
        duration = len(y) / sr


        import tgt
        tgrd_f = os.path.join(tgrd_fs_folder, f"common_voice_it_{row['id']}.TextGrid")
        tgrd = tgt.read_textgrid(tgrd_f)
        tier = tgt.IntervalTier(name="target", start_time=0.0, end_time=duration)
        tier.add_interval(tgt.Interval(row['target_phone_start_time'], row['target_phone_end_time'], "TARGET"))
        tgrd.add_tier(tier)
        tgt.io.write_to_file(tgrd, os.path.join(folder_path, f"{unique_id}.TextGrid"))
    
    def delete_file_from_folder(self, unique_id, folder_path):
        import os
        
        mp3_path = os.path.join(folder_path, f"{unique_id}.mp3")
        tgrd_path = os.path.join(folder_path, f"{unique_id}.TextGrid")

        if os.path.isfile(mp3_path):
            os.remove(mp3_path)
        if os.path.isfile(tgrd_path):
            os.remove(tgrd_path)
    
    def return_gold_dataset(self, trained_samples=[]):
        self._refresh_gold_labels()
        gold_indices = []
        unique_ids = []
        for idx in range(len(self.df)):
            unique_id = self.get_unique_id(idx)
            if unique_id in self.gold_labels["label"] and pd.notna(self.gold_labels["label"][unique_id]) and unique_id not in trained_samples:
                gold_indices.append(idx)
                unique_ids.append(unique_id)
        return torch.utils.data.Subset(self, gold_indices), unique_ids

    def get_unique_id(self, dataset_index):
        row = self.df.iloc[dataset_index]
        unique_id = f"{row['id']}_{row['target_phone']}_{row['target_phone_start_time']}_{row['target_phone_end_time']}"
        return unique_id

    def _add_heuristic_label(self, data):
        if data["type"] == "non_troncamento":
            data["silver_label"] = [0, 1][data["target_phone"] == "e"]
        else:
            data["silver_label"] = 0
        
        return data

    def add_data_to_gold(self, dataset_index, label=None):
        row = self.df.iloc[dataset_index]
        unique_id = f"{row['id']}_{row['target_phone']}_{row['target_phone_start_time']}_{row['target_phone_end_time']}"
        
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

    def _to_embed(self, segment, sr, min_len=400):
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

        return embed.cpu()

    def _get_audio_segment(self, id_num, mp3_path, start_time, end_time, save_temp_files=False):

        target_id = f"{id_num}"
        audio, sr = librosa.load(mp3_path, sr=None)
        
        if save_temp_files:
            import shutil
            shutil.copyfile(mp3_path, "temp.mp3")
            shutil.copyfile(f"it_vxc_textgrids17_acoustic17/common_voice_it_{target_id}.TextGrid", "temp.TextGrid")

        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr)

        segment = audio[start_idx:end_idx]

        return segment, sr, Audio(segment, rate=sr)

class SegmentPairDataset(torch.utils.data.Dataset):
    def __init__(self, raw_dataset):
        self.data = raw_dataset

    def __len__(self):
        return len(self.data)
    
    def _get_features(self, ex):
        e1 = ex["preceding_embed"]
        e2 = ex["target_embed"]
        features = torch.cat([
            e1,
            e2,
            torch.abs(e1 - e2),
            e1 * e2
        ])
        return features

    def __getitem__(self, idx):
        ex = self.data[idx]

        features = self._get_features(ex)

        phone_ids = torch.tensor([
            ex["preceding_phone_id"],
            ex["target_phone_id"]
        ])

        label = ex["silver_label"]
        return features, phone_ids, torch.tensor(label, dtype=torch.float32)