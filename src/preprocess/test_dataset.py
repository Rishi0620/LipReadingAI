from torch.utils.data import DataLoader
from lipreading_dataset import LipReadingDataset, get_char2idx
import torch

def custom_collate_fn(batch):
    videos, labels, video_lengths, label_lengths = zip(*batch)

    # Stack videos [B, T, C, H, W]
    videos = torch.stack(videos)

    # Concatenate all label tensors into a single 1D tensor (for CTC loss)
    labels_concat = torch.cat(labels)

    # Convert lengths to tensors
    video_lengths = torch.tensor(video_lengths, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)

    return videos, labels_concat, video_lengths, label_lengths

if __name__ == "__main__":
    char2idx = get_char2idx()

    dataset = LipReadingDataset(
        root_dir="/Users/rishabhbhargav/PycharmProjects/LipReadingAI/data/s1/mouth_crops",  # Update path if needed
        char2idx=char2idx
    )

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

    for batch in dataloader:
        videos, labels, video_lengths, label_lengths = batch
        print("Videos:", videos.shape)          # [B, T, C, H, W]
        print("Labels:", labels.shape)          # [Total_chars]
        print("Video Lengths:", video_lengths)  # [B]
        print("Label Lengths:", label_lengths)  # [B]
        break