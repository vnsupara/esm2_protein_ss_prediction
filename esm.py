import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

def embed_protein(pid, seq, tokenizer, esm_model, save_dir):
    save_path = os.path.join(save_dir, f"{pid}.npy")
    if os.path.exists(save_path):
        return np.load(save_path)

    tokens = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        out = esm_model(**{k:v.to(device) for k,v in tokens.items()})
    emb = out.last_hidden_state.squeeze(0).cpu().numpy()
    np.save(save_path, emb)
    return emb

def load_sequences(fasta_path):
    seqs = {}
    for rec in SeqIO.parse(fasta_path, "fasta"):
        seqs[rec.id] = str(rec.seq)
    return seqs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--out", default="embeddings")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    esm_model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)
    esm_model.eval()

    seqs = load_sequences(args.fasta)
    for pid, seq in tqdm(seqs.items()):
        embed_protein(pid, seq, tokenizer, esm_model, args.out)
