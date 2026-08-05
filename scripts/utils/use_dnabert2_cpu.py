"""
Load DNABERT-2-117M on CPU without GPU/Triton

This script demonstrates how to use the full DNABERT-2 model on CPU-only systems.
The model has been configured to use standard PyTorch attention instead of Triton flash attention.

Performance:
- Single sequence: ~100ms
- Batch of 5: ~60ms total (~12ms per sequence)
- Model size: 447 MB in memory
- Parameters: 117 million
"""

import io
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer


def load_dnabert2_cpu(model_path="./models/dnabert2_cpu"):
    """
    Load DNABERT-2 model for CPU inference.

    Args:
        model_path: Path to CPU-compatible model directory

    Returns:
        tuple: (model, tokenizer)
    """
    print("Loading DNABERT-2 (CPU-compatible version)...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()  # Set to evaluation mode

    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    return model, tokenizer


def generate_embeddings(sequences, model, tokenizer, batch_size=16):
    """
    Generate DNABERT-2 embeddings for DNA sequences.

    Args:
        sequences: List of DNA sequences
        model: DNABERT-2 model
        tokenizer: DNABERT-2 tokenizer
        batch_size: Batch size for inference

    Returns:
        numpy array of shape (n_sequences, 768)
    """
    import numpy as np

    embeddings = []

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=512
        )

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

            # Get [CLS] token embedding (first token)
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs.last_hidden_state

            # Use [CLS] token (position 0) as sequence embedding
            batch_embeddings = hidden_states[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)

    return np.vstack(embeddings)


def main():
    """Demonstration of DNABERT-2 on CPU."""

    # Load model
    model, tokenizer = load_dnabert2_cpu()

    # Example eDNA sequences
    sequences = [
        "ATCGATCGATCGATCG",
        "GCTAGCTAGCTAGCTA",
        "TTAATTAATTAATTAA",
        "CCGGCCGGCCGGCCGG",
        "AATTCCGGAATTCCGG",
    ]

    print(f"\nGenerating embeddings for {len(sequences)} sequences...")
    start = time.time()

    embeddings = generate_embeddings(sequences, model, tokenizer)

    elapsed = time.time() - start

    print(f"✓ Generated embeddings: {embeddings.shape}")
    print(
        f"  Time: {elapsed*1000:.0f} ms ({elapsed*1000/len(sequences):.0f} ms per sequence)"
    )
    print(f"  First embedding (5 dims): {embeddings[0, :5]}")

    # Show similarity between sequences
    print(f"\nSequence similarity (cosine):")
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(embeddings)

    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            sim = similarities[i, j]
            print(f"  Seq {i} vs Seq {j}: {sim:.3f}")

    print("\n✅ DNABERT-2 is working on CPU!")
    print("   You can now use this for:")
    print("   - eDNA sequence classification")
    print("   - Organism identification")
    print("   - Sequence clustering")
    print("   - Transfer learning tasks")


if __name__ == "__main__":
    main()
