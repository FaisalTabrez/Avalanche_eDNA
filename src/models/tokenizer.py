"""
DNA Sequence Tokenization and Encoding utilities
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import itertools
from collections import Counter
import pickle
from pathlib import Path

class DNATokenizer:
    """Tokenizer for DNA sequences supporting k-mer and character-level encoding"""
    
    def __init__(self, 
                 encoding_type: str = "kmer",
                 kmer_size: int = 4,  # OPTIMIZED: Best performance in eDNA tuning (was 6)
                 stride: int = 1,
                 add_special_tokens: bool = True):
        """
        Initialize DNA tokenizer
        
        Args:
            encoding_type: Type of encoding ('kmer', 'char', 'both')
            kmer_size: Size of k-mers for k-mer encoding
            stride: Stride for k-mer generation
            add_special_tokens: Whether to add special tokens (PAD, UNK, CLS, SEP)
        """
        self.encoding_type = encoding_type
        self.kmer_size = kmer_size
        self.stride = stride
        self.add_special_tokens = add_special_tokens
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # Create token-to-id and id-to-token mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Special token IDs
        if add_special_tokens:
            self.pad_token_id = self.token_to_id['[PAD]']
            self.unk_token_id = self.token_to_id['[UNK]']
            self.cls_token_id = self.token_to_id['[CLS]']
            self.sep_token_id = self.token_to_id['[SEP]']
    
    def _build_vocabulary(self) -> List[str]:
        """Build vocabulary based on encoding type"""
        vocab = []
        
        # Add special tokens
        if self.add_special_tokens:
            vocab.extend(['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
        
        if self.encoding_type in ['char', 'both']:
            # Character-level vocabulary
            nucleotides = ['A', 'T', 'G', 'C', 'N']
            vocab.extend(nucleotides)
        
        if self.encoding_type in ['kmer', 'both']:
            # K-mer vocabulary
            nucleotides = ['A', 'T', 'G', 'C']
            kmers = [''.join(kmer) for kmer in itertools.product(nucleotides, repeat=self.kmer_size)]
            vocab.extend(kmers)
        
        return vocab
    
    def sequence_to_kmers(self, sequence: str) -> List[str]:
        """
        Convert DNA sequence to k-mers
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            List of k-mers
        """
        sequence = sequence.upper()
        kmers = []
        
        for i in range(0, len(sequence) - self.kmer_size + 1, self.stride):
            kmer = sequence[i:i + self.kmer_size]
            
            # Only include k-mers without N bases for k-mer encoding
            if 'N' not in kmer:
                kmers.append(kmer)
            elif self.add_special_tokens:
                kmers.append('[UNK]')
        
        return kmers
    
    def sequence_to_chars(self, sequence: str) -> List[str]:
        """
        Convert DNA sequence to character list
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            List of characters
        """
        return list(sequence.upper())
    
    def encode_sequence(self, sequence: str, max_length: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Encode DNA sequence to token IDs
        
        Args:
            sequence: DNA sequence string
            max_length: Maximum sequence length (for padding/truncation)
            
        Returns:
            Dictionary containing token IDs and attention mask
        """
        if self.encoding_type == 'kmer':
            tokens = self.sequence_to_kmers(sequence)
        elif self.encoding_type == 'char':
            tokens = self.sequence_to_chars(sequence)
        elif self.encoding_type == 'both':
            # Combine character and k-mer tokens
            char_tokens = self.sequence_to_chars(sequence)
            kmer_tokens = self.sequence_to_kmers(sequence)
            tokens = char_tokens + ['[SEP]'] + kmer_tokens
        else:
            raise ValueError(f"Invalid encoding_type: {self.encoding_type}")
        
        # Add special tokens
        if self.add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # Convert to IDs
        token_ids = [self.token_to_id.get(token, self.unk_token_id) for token in tokens]
        
        # Handle max_length
        if max_length is not None:
            if len(token_ids) > max_length:
                # Truncate
                token_ids = token_ids[:max_length]
                # Ensure we end with [SEP] if using special tokens
                if self.add_special_tokens:
                    token_ids[-1] = self.sep_token_id
            else:
                # Pad
                padding_length = max_length - len(token_ids)
                token_ids.extend([self.pad_token_id] * padding_length)
        
        # Create attention mask
        attention_mask = [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]
        
        return {
            'input_ids': np.array(token_ids),
            'attention_mask': np.array(attention_mask),
            'tokens': tokens
        }
    
    def encode_sequences(self, sequences: List[str], max_length: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Encode multiple DNA sequences
        
        Args:
            sequences: List of DNA sequence strings
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing batch of token IDs and attention masks
        """
        encoded_sequences = [self.encode_sequence(seq, max_length) for seq in sequences]
        
        return {
            'input_ids': np.array([enc['input_ids'] for enc in encoded_sequences]),
            'attention_mask': np.array([enc['attention_mask'] for enc in encoded_sequences])
        }
    
    def decode_sequence(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to sequence
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded sequence string
        """
        tokens = [self.id_to_token.get(token_id, '[UNK]') for token_id in token_ids]
        
        # Remove special tokens
        if self.add_special_tokens:
            tokens = [token for token in tokens if token not in ['[PAD]', '[CLS]', '[SEP]']]
        
        # For k-mer encoding, we need to reconstruct the sequence
        if self.encoding_type == 'kmer':
            if not tokens:
                return ""
            
            # Reconstruct by overlapping k-mers
            sequence = tokens[0]
            for token in tokens[1:]:
                if token != '[UNK]':
                    # Add the last character of the k-mer
                    sequence += token[-1]
            
            return sequence
        
        elif self.encoding_type == 'char':
            return ''.join(tokens)
        
        else:  # both
            # Find the separator
            if '[SEP]' in tokens:
                sep_idx = tokens.index('[SEP]')
                char_tokens = tokens[:sep_idx]
                return ''.join(char_tokens)
            else:
                return ''.join(tokens)
    
    def save(self, save_path: Path) -> None:
        """Save tokenizer to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        tokenizer_data = {
            'encoding_type': self.encoding_type,
            'kmer_size': self.kmer_size,
            'stride': self.stride,
            'add_special_tokens': self.add_special_tokens,
            'vocab': self.vocab,
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
    
    @classmethod
    def load(cls, load_path: Path) -> 'DNATokenizer':
        """Load tokenizer from file"""
        with open(load_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        tokenizer = cls(
            encoding_type=tokenizer_data['encoding_type'],
            kmer_size=tokenizer_data['kmer_size'],
            stride=tokenizer_data['stride'],
            add_special_tokens=tokenizer_data['add_special_tokens']
        )
        
        # Override with saved data
        tokenizer.vocab = tokenizer_data['vocab']
        tokenizer.token_to_id = tokenizer_data['token_to_id']
        tokenizer.id_to_token = tokenizer_data['id_to_token']
        tokenizer.vocab_size = len(tokenizer.vocab)
        
        return tokenizer

class SequenceDataset:
    """Dataset class for DNA sequences"""
    
    def __init__(self, 
                 sequences: List[str],
                 labels: Optional[List[str]] = None,
                 tokenizer: Optional[DNATokenizer] = None,
                 max_length: int = 512):
        """
        Initialize sequence dataset
        
        Args:
            sequences: List of DNA sequences
            labels: Optional list of labels
            tokenizer: DNA tokenizer instance
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
        # Initialize tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = DNATokenizer()
        else:
            self.tokenizer = tokenizer
        
        # Pre-encode sequences for efficiency
        self.encoded_sequences = self._encode_all_sequences()
    
    def _encode_all_sequences(self) -> Dict[str, np.ndarray]:
        """Pre-encode all sequences"""
        return self.tokenizer.encode_sequences(self.sequences, self.max_length)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        """Get item by index"""
        item = {
            'input_ids': self.encoded_sequences['input_ids'][idx],
            'attention_mask': self.encoded_sequences['attention_mask'][idx],
            'sequence': self.sequences[idx]
        }
        
        if self.labels is not None:
            item['label'] = self.labels[idx]
        
        return item
    
    def get_batch(self, indices: List[int]) -> Dict[str, np.ndarray]:
        """Get batch of sequences by indices"""
        batch = {
            'input_ids': self.encoded_sequences['input_ids'][indices],
            'attention_mask': self.encoded_sequences['attention_mask'][indices],
            'sequences': [self.sequences[i] for i in indices]
        }
        
        if self.labels is not None:
            batch['labels'] = [self.labels[i] for i in indices]
        
        return batch