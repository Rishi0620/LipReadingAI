import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class LipReadingDataset(Dataset):
    def __init__(self, root_dir, char2idx, max_frames=75):
        self.root_dir = root_dir
        self.sample_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, d))]
        self.char2idx = char2idx
        self.max_frames = max_frames
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_path = self.sample_dirs[idx]

        # Load all mouth frames
        frames = []
        frame_files = sorted([f for f in os.listdir(sample_path) if f.endswith(".png")])
        for fname in frame_files:
            img_path = os.path.join(sample_path, fname)
            img = Image.open(img_path).convert("L")  # grayscale
            img = self.transform(img)
            frames.append(img)

        if len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        else:
            pad = [torch.zeros_like(frames[0])] * (self.max_frames - len(frames))
            frames.extend(pad)

        video_tensor = torch.stack(frames)  # [T, C, H, W]

        # Load transcript
        transcript_path = os.path.join(sample_path, "transcript.txt")
        with open(transcript_path, "r") as f:
            text = f.read().strip().lower()
        label = [self.char2idx[c] for c in text if c in self.char2idx]

        return video_tensor, torch.tensor(label, dtype=torch.long), len(frames), len(label)


def get_char2idx():
    charset = "abcdefghijklmnopqrstuvwxyz '"
    char2idx = {c: i + 1 for i, c in enumerate(charset)}  # 0 is for CTC blank
    char2idx['-'] = 0
    return char2idx