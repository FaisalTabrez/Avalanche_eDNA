# Troubleshooting Guide

## Common Issues

### 1. Memory Errors
**Error:** Out of memory during embedding generation
**Solution:** Reduce batch size or use smaller embedding dimension

Edit `config/config.yaml`:
```yaml
embedding:
  training:
    batch_size: 16  # Reduce from default 32
```

### 2. BLAST Not Found
**Error:** BLAST tools not found
**Solution:** Install BLAST and add to PATH

See [Installation Guide](installation.md) for BLAST installation instructions.

### 3. GPU Not Detected
**Error:** GPU not available for acceleration
**Solution:** Check GPU availability and installation

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

Ensure PyTorch GPU version is installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Large Dataset Processing
For datasets with >10,000 sequences:
- Use chunked processing
- Enable GPU acceleration
- Consider cloud deployment

## Performance Optimization

### GPU Acceleration
```yaml
performance:
  use_gpu: true
  gpu_memory_fraction: 0.8
```

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move model to GPU
model = model.to(device)
```

### Parallel Processing
```yaml
performance:
  n_jobs: -1  # Use all CPU cores
  chunk_size: 1000
```

```python
from multiprocessing import Pool
from functools import partial

def parallel_analysis(sequences, n_processes=4):
    with Pool(n_processes) as pool:
        process_func = partial(analyze_sequence, param1=value1)
        results = pool.map(process_func, sequences)
    return results
```

### Memory Management
- Process data in batches
- Use data streaming for large files
- Enable garbage collection

```python
# Process large datasets in chunks
def process_large_dataset(sequences, chunk_size=1000):
    for i in range(0, len(sequences), chunk_size):
        chunk = sequences[i:i+chunk_size]
        # Process chunk
        yield process_chunk(chunk)

# Use generators for memory efficiency
def sequence_generator(file_path):
    from Bio import SeqIO
    for record in SeqIO.parse(file_path, "fasta"):
        yield str(record.seq)
```

## Error Handling

### Common Exceptions

```python
from models.tokenizer import DNATokenizer

try:
    tokenizer = DNATokenizer.load("nonexistent.pkl")
except FileNotFoundError:
    print("Tokenizer file not found")

try:
    encoded = tokenizer.encode_sequence("INVALID_SEQUENCE")
except ValueError as e:
    print(f"Invalid sequence: {e}")
```

### Logging

```python
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("Starting analysis...")
```

## Getting Help

- Check the logs in the `logs/` directory
- Run test suite: `python tests/test_system.py`
- Review example notebooks in `notebooks/`
- Check the [User Guide](user_guide.md) for detailed usage instructions