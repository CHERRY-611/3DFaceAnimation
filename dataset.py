import os
import torch
import numpy as np
from torch.utils.data import Dataset
import librosa

class EmoTalkDataset(Dataset):
    def __init__(self, root_dir, dataset_name="RAVDESS", sr=16000, max_len=None):
        """
        root_dir: 상위 경로, 예: "./3D-ETF"
        dataset_name: "RAVDESS" 또는 "HDTF"
        sr: 오디오 샘플링 비율
        max_len: 길이를 맞추고 싶다면 지정 (ex: 48000)
        """
        self.audio_dir = os.path.join(root_dir, dataset_name, "audio")
        self.bs_dir = os.path.join(root_dir, dataset_name, "blendshape")
        self.sr = sr
        self.max_len = max_len

        self.filenames = sorted([
            f.replace(".wav", "")
            for f in os.listdir(self.audio_dir)
            if f.endswith(".wav") and os.path.exists(os.path.join(self.bs_dir, f.replace(".wav", ".npy")))
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        wav_path = os.path.join(self.audio_dir, name + ".wav")
        bs_path = os.path.join(self.bs_dir, name + ".npy")

        # 오디오 로딩 및 정규화
        waveform, _ = librosa.load(wav_path, sr=self.sr)

        if self.max_len:
            pad_width = max(0, self.max_len - len(waveform))
            waveform = np.pad(waveform, (0, pad_width))[:self.max_len]

        audio = torch.from_numpy(waveform).float()  # [1, T]

        # ▶️ 블렌드셰입 처리 추가
        bs = np.load(bs_path)
        target_len = self.max_len // 16000 * 30  # 예: 64000 samples → 120 frames

        if bs.shape[0] < target_len:
            bs = np.pad(bs, ((0, target_len - bs.shape[0]), (0, 0)))
        else:
            bs = bs[:target_len]

        bs = torch.from_numpy(bs).float()


        # 현재는 고정값 (원하면 외부에서 제공 가능)
        level = torch.tensor(1)   # 감정 강도: 0=low, 1=high
        person = torch.tensor(0)  # 화자: 0~23 중 하나

        return {
            "input12": audio,
            "input21": audio.clone(),     # input21도 동일한 오디오
            "target11": bs,
            "target12": bs.clone(),
            "level": level,
            "person": person
        }