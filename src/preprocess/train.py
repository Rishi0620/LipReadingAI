import os
import torch
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
from src.models.minimal_lip_reader import MinimalLipReader
from src.preprocess.lipreading_dataset import LipReadingDataset, get_char2idx
from src.preprocess.test_dataset import custom_collate_fn
from src.models.decoder import greedy_decode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset setup
char2idx = get_char2idx()
idx2char = {v: k for k, v in char2idx.items()}
num_classes = len(char2idx)

dataset = LipReadingDataset(
    root_dir="/Users/rishabhbhargav/PycharmProjects/LipReadingAI/data/s1/mouth_crops",
    char2idx=char2idx
)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)

# Model, Loss, Optimizer
model = MinimalLipReader(num_classes).to(device)
criterion = CTCLoss(blank=char2idx['-'])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
for epoch in range(1):
    model.train()
    for batch in loader:
        videos, labels, video_lengths, label_lengths = batch
        videos = videos.to(device)
        labels = labels.to(device)
        video_lengths = video_lengths.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()
        outputs = model(videos)  # [B, T, C]
        outputs = outputs.permute(1, 0, 2)  # [T, B, C] for CTC

        loss = criterion(outputs, labels, video_lengths, label_lengths)
        loss.backward()
        optimizer.step()

        # Decode predictions
        decoded_preds = greedy_decode(outputs, idx2char)

        # Print predictions
        start = 0
        for i in range(len(label_lengths)):
            length = label_lengths[i].item()
            label_seq = labels[start:start + length].cpu().numpy()
            ground_truth = ''.join([idx2char[idx] for idx in label_seq])
            print(f"GT: {ground_truth}")
            print(f"PR: {decoded_preds[i]}")
            print("-" * 30)
            start += length

        print(f"Loss: {loss.item():.4f}")

# Save model
save_path = "/Users/rishabhbhargav/PycharmProjects/LipReadingAI/outputs/model"
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_path, "minimal_lip_reader.pth"))
print(f"Model saved to {os.path.join(save_path, 'minimal_lip_reader.pth')}")