# Installation Guide

## Prerequisites
- Python 3.9 or higher
- Conda (recommended) or pip
- Git
- 8GB+ RAM recommended
- GPU support optional but recommended for large datasets

## Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Avalanche
```

2. **Create conda environment:**
```bash
conda create -n edna-biodiversity python=3.9
conda activate edna-biodiversity
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install BLAST tools (optional for taxonomy assignment):**
```bash
# On Ubuntu/Debian
sudo apt-get install ncbi-blast+

# On macOS with Homebrew
brew install blast

# On Windows
# Download from NCBI and add to PATH
```

5. **Download reference databases (optional):**
```bash
python scripts/download_data.py
```

## Alternative Virtual Environment Setup
If you prefer venv over conda:
```bash
# It is recommended to use a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install all required packages
pip install -r requirements.txt
```

*Note: The system may also require external tools like NCBI BLAST to be installed and configured in `config/config.yaml`.*