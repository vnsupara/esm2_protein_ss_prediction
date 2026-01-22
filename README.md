# ESM2 Protein Secondary Structure Prediction

This project predicts protein secondary structure using embeddings from **ESM2 (facebook/esm2_t12_35M_UR50D)** and a **BiLSTM classifier**.  
The workflow includes:
- Extracting per-residue embeddings from ESM2
- Training a BiLSTM classifier on labeled residues
- Generating predictions for test sequences

---

## Features
- ESM2 embedding extraction with caching
- Custom PyTorch dataset for residue-level labels
- BiLSTM classifier for 9-class secondary structure prediction
- Training loop with checkpointing
- Prediction script for test data

---

## Data Availability
The dataset used in this project was provided as part of a university course and is not publicly available.  
To run the code, place the following files in the `data/` directory:

- `train.tsv`
- `test.tsv`
- `sequences.fasta`

These files are not included in the repository.
