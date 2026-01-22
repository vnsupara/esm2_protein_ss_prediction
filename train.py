import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import SSData
from model import BiLSTMClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--emb", default="embeddings")
    parser.add_argument("--checkpoint", default="checkpoints/model_checkpoint.pt")
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    train_df = pd.read_csv(args.train, sep="\t")
    train_dataset = SSData(train_df, emb_dir=args.emb)
    train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = BiLSTMClassifier().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if os.path.exists(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint))

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total, correct = 0, 0

        for x, y in train_loader:
            mask = (y != -1)
            x, y = x[mask].to(device), y[mask].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            correct += (pred.argmax(1) == y).sum().item()
            total += y.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}/{args.epochs} - Train Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.checkpoint)
            print(f"new best model saved at Epoch {epoch+1} with accuracy: {best_acc:.2f}%")
        else:
            print("no improvement in accuracy, skipping checkpoint save")
