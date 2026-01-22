import os
import random
import string
import argparse
from pathlib import Path
import pandas as pd

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SS_LABELS = ['H','B','E','G','I','P','T','S','.']

def make_sequences(n_seqs=50, min_len=50, max_len=300, seed=42):
    random.seed(seed)
    seqs = {}
    for i in range(1, n_seqs + 1):
        pid = f"P{i:04d}"
        length = random.randint(min_len, max_len)
        seq = "".join(random.choices(AMINO_ACIDS, k=length))
        seqs[pid] = seq
    return seqs

def sample_positions(seqs, n_rows, seed=42, avoid=set()):
    """
    Return list of tuples (pid, pos) sampled across sequences.
    avoid is a set of (pid,pos) to exclude.
    """
    random.seed(seed)
    all_positions = []
    for pid, seq in seqs.items():
        for pos in range(len(seq)):
            if (pid, pos) not in avoid:
                all_positions.append((pid, pos))
    if n_rows > len(all_positions):
        raise ValueError("Requested more rows than available positions.")
    sampled = random.sample(all_positions, n_rows)
    return sampled

def assign_secondary_structure(n, seed=42):
    """
    Return list of secondary structure labels with realistic distribution:
    H 30%, E 20%, T 10%, S 10%, others split remaining 30%.
    """
    random.seed(seed)
    probs = {
        'H': 0.30,
        'E': 0.20,
        'T': 0.10,
        'S': 0.10
    }
    remaining = 1.0 - sum(probs.values())
    others = [l for l in SS_LABELS if l not in probs]
    per_other = remaining / len(others)
    for o in others:
        probs[o] = per_other
    labels = list(probs.keys())
    weights = [probs[l] for l in labels]
    return random.choices(labels, weights=weights, k=n)

def write_fasta(seqs, out_path):
    with open(out_path, "w") as f:
        for pid, seq in seqs.items():
            f.write(f">{pid}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

def build_rows(seqs, sampled_positions, include_ss=True, seed=42):
    rows = []
    if include_ss:
        ss_labels = assign_secondary_structure(len(sampled_positions), seed=seed)
    else:
        ss_labels = [None] * len(sampled_positions)
    for (pid, pos), ss in zip(sampled_positions, ss_labels):
        aa = seqs[pid][pos]
        rid = f"{pid}_{aa}_{pos}"
        seq_full = seqs[pid]
        if include_ss:
            rows.append({"id": rid, "sequence": seq_full, "secondary_structure": ss})
        else:
            rows.append({"id": rid, "sequence": seq_full})
    return rows

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic protein dataset.")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory. Defaults to the user's Downloads folder.")
    parser.add_argument("--n-seqs", type=int, default=50)
    parser.add_argument("--train-rows", type=int, default=2000)
    parser.add_argument("--test-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir).expanduser().resolve()
    else:
        home = Path.home()
        downloads = home / "Downloads"
        out_dir = downloads if downloads.exists() else home

    data_dir = out_dir / "synthetic_protein_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    seqs = make_sequences(n_seqs=args.n_seqs, seed=args.seed)

    fasta_path = data_dir / "sequences.fasta"
    write_fasta(seqs, fasta_path)

    train_sampled = sample_positions(seqs, args.train_rows, seed=args.seed)
    train_rows = build_rows(seqs, train_sampled, include_ss=True, seed=args.seed + 1)
    train_df = pd.DataFrame(train_rows)
    train_path = data_dir / "train.tsv"
    train_df.to_csv(train_path, sep="\t", index=False)

    avoid = set(train_sampled)
    test_sampled = sample_positions(seqs, args.test_rows, seed=args.seed + 2, avoid=avoid)
    test_rows = build_rows(seqs, test_sampled, include_ss=False)
    test_df = pd.DataFrame(test_rows)
    test_path = data_dir / "test.tsv"
    test_df.to_csv(test_path, sep="\t", index=False)

    total_residues = sum(len(s) for s in seqs.values())
    print("Synthetic protein data generated.")
    print(f"Output directory: {data_dir}")
    print(f"FASTA: {fasta_path}")
    print(f"Train TSV: {train_path}")
    print(f"Test TSV: {test_path}")
    print(f"Number of sequences: {len(seqs)}")
    print(f"Total residues: {total_residues}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print("\nExample rows (train):")
    print(train_df.head(3).to_string(index=False))
    print("\nExample rows (test):")
    print(test_df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
