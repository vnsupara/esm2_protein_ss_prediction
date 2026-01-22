import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import SSData
from model import BiLSTMClassifier

# labels mapping (same as in your notebook)
labels = ['H','B','E','G','I','P','T','S','.']
id2label = {i: l for i, l in enumerate(labels)}

device = "cuda" if torch.cuda.is_available() else "cpu"

def predict(test_tsv: str, emb_dir: str, checkpoint: str, out_csv: str, batch_size: int = 256):
    df = pd.read_csv(test_tsv, sep="\t")
    dataset = SSData(df, emb_dir=emb_dir)
    loader = DataLoader(dataset, batch_size=batch_size)

    model = BiLSTMClassifier().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds = []
    for x, _ in loader:
        x = x.to(device)
        with torch.no_grad():
            out = model(x).argmax(1).cpu().numpy()
        preds.extend(out)

    df["prediction"] = [id2label[i] for i in preds]
    df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", required=True, help="Path to test.tsv")
    parser.add_argument("--emb", default="embeddings", help="Directory with .npy embeddings")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint .pt or .pth")
    parser.add_argument("--out", default="prediction.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    predict(args.test, args.emb, args.checkpoint, args.out, args.batch_size)
