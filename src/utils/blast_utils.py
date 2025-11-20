"""
BLAST utilities for Windows environment
Handles BLAST operations with proper Windows path management
"""

import os
import sys
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from Bio import SeqIO
from Bio.Blast import NCBIXML

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

logger = logging.getLogger(__name__)

class WindowsBLASTRunner:
    """BLAST runner optimized for Windows environment"""
    
    def __init__(self, config_section: Optional[Dict[str, Any]] = None):
        """
        Initialize BLAST runner
        
        Args:
            config_section: BLAST configuration section, defaults to config['taxonomy']['blast']
        """
        self.config = config_section or config.get('taxonomy', {}).get('blast', {})
        
        # Get BLAST executable paths
        self.blastn_path = self.config.get('blastn_path', 'blastn')
        self.makeblastdb_path = self.config.get('makeblastdb_path', 'makeblastdb')
        
        # BLAST parameters
        self.evalue = self.config.get('evalue', 1e-5)
        self.max_targets = self.config.get('max_targets', 10)
        self.identity_threshold = self.config.get('identity_threshold', 97.0)
        self.output_format = self.config.get('output_format', 5)  # XML format
        self.num_threads = self.config.get('num_threads', 4)
        
        # Verify BLAST installation
        self._verify_blast_installation()
        
        logger.info(f"BLAST runner initialized with executable: {self.blastn_path}")
    
    def _verify_blast_installation(self) -> bool:
        """Verify that BLAST tools are accessible"""
        try:
            # Test blastn
            result = subprocess.run(
                [self.blastn_path, '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logger.info(f"BLAST verification successful: {version_info}")
                return True
            else:
                raise RuntimeError(f"BLAST version check failed: {result.stderr}")
                
        except FileNotFoundError:
            raise RuntimeError(f"BLAST executable not found: {self.blastn_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("BLAST version check timed out")
        except Exception as e:
            raise RuntimeError(f"Error verifying BLAST installation: {e}")
    
    def create_blast_database(self, 
                            fasta_file: str, 
                            database_name: str,
                            database_type: str = 'nucl') -> bool:
        """
        Create a BLAST database from a FASTA file
        
        Args:
            fasta_file: Path to input FASTA file
            database_name: Name for the output database
            database_type: Database type ('nucl' or 'prot')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                self.makeblastdb_path,
                '-in', fasta_file,
                '-dbtype', database_type,
                '-out', database_name,
                '-parse_seqids',
                '-title', f'Database_{database_type}'
            ]
            
            logger.info(f"Creating BLAST database: {database_name}")
            logger.debug(f"Running command: {' '.join(cmd)}")
            logger.debug(f"Working directory: {os.getcwd()}")
            logger.debug(f"Input file exists: {os.path.exists(fasta_file)}")
            logger.debug(f"Output directory exists: {Path(database_name).parent.exists()}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Check if database files were created (regardless of return code)
            database_path = Path(database_name)
            parent_dir = database_path.parent if database_path.parent.exists() else Path('.')
            db_name = database_path.name
            
            # Look for database files
            db_files = list(parent_dir.glob(f"{db_name}.*"))
            
            if db_files:
                # Database files were created - success!
                logger.info(f"BLAST database created successfully: {database_name}")
                logger.info(f"Created {len(db_files)} database files")
                return True
            elif result.returncode == 0:
                logger.info(f"BLAST database created successfully: {database_name}")
                return True
            else:
                logger.error(f"Failed to create BLAST database: {result.stderr}")
                logger.error(f"Command: {' '.join(cmd)}")
                logger.error(f"Return code: {result.returncode}")
                if result.stdout:
                    logger.error(f"STDOUT: {result.stdout}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating BLAST database: {e}")
            return False
    
    def run_blastn_search(self, 
                         query_sequences: List[str],
                         database_path: str,
                         sequence_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Run BLASTN search against a database
        
        Args:
            query_sequences: List of DNA sequences to search
            database_path: Path to BLAST database
            sequence_ids: Optional list of sequence IDs
            
        Returns:
            List of BLAST results
        """
        if sequence_ids is None:
            sequence_ids = [f"seq_{i}" for i in range(len(query_sequences))]
        
        # Create temporary query file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as temp_query:
            for seq_id, sequence in zip(sequence_ids, query_sequences):
                temp_query.write(f">{seq_id}\n{sequence}\n")
            temp_query_path = temp_query.name
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as temp_output:
            temp_output_path = temp_output.name
        
        try:
            # Run BLAST
            cmd = [
                self.blastn_path,
                '-query', temp_query_path,
                '-db', database_path,
                '-out', temp_output_path,
                '-outfmt', str(self.output_format),
                '-evalue', str(self.evalue),
                '-max_target_seqs', str(self.max_targets),
                '-num_threads', str(self.num_threads)
            ]
            
            logger.info(f"Running BLAST search with {len(query_sequences)} sequences")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                # Parse XML results
                return self._parse_blast_xml(temp_output_path, sequence_ids)
            else:
                logger.error(f"BLAST search failed: {result.stderr}")
                return []
                
        except Exception as e:
            logger.error(f"Error running BLAST search: {e}")
            return []
        finally:
            # Clean up temporary files
            try:
                os.unlink(temp_query_path)
                os.unlink(temp_output_path)
            except:
                pass
    
    def _parse_blast_xml(self, xml_file: str, sequence_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Parse BLAST XML output file
        
        Args:
            xml_file: Path to BLAST XML output
            sequence_ids: List of query sequence IDs
            
        Returns:
            List of parsed BLAST results
        """
        results = []
        
        try:
            with open(xml_file, 'r') as handle:
                blast_records = NCBIXML.parse(handle)
                
                for i, blast_record in enumerate(blast_records):
                    seq_id = sequence_ids[i] if i < len(sequence_ids) else f"seq_{i}"
                    
                    if blast_record.alignments:
                        # Get best hit
                        best_alignment = blast_record.alignments[0]
                        best_hsp = best_alignment.hsps[0]
                        
                        # Calculate identity percentage
                        identity_pct = (best_hsp.identities / best_hsp.align_length) * 100
                        
                        result = {
                            'query_id': seq_id,
                            'subject_id': best_alignment.title,
                            'identity_pct': identity_pct,
                            'alignment_length': best_hsp.align_length,
                            'evalue': best_hsp.expect,
                            'bit_score': best_hsp.bits,
                            'query_start': best_hsp.query_start,
                            'query_end': best_hsp.query_end,
                            'subject_start': best_hsp.sbjct_start,
                            'subject_end': best_hsp.sbjct_end,
                            'has_hit': True,
                            'above_threshold': identity_pct >= self.identity_threshold
                        }
                    else:
                        # No hits found
                        result = {
                            'query_id': seq_id,
                            'subject_id': None,
                            'identity_pct': 0.0,
                            'alignment_length': 0,
                            'evalue': float('inf'),
                            'bit_score': 0.0,
                            'query_start': 0,
                            'query_end': 0,
                            'subject_start': 0,
                            'subject_end': 0,
                            'has_hit': False,
                            'above_threshold': False
                        }
                    
                    results.append(result)
            
            logger.info(f"Parsed {len(results)} BLAST results")
            return results
            
        except Exception as e:
            logger.error(f"Error parsing BLAST XML: {e}")
            return []
    
    def extract_taxonomy_from_hit(self, hit_title: str) -> Dict[str, Optional[str]]:
        """
        Extract taxonomic information from BLAST hit title
        
        Args:
            hit_title: BLAST hit title/description
            
        Returns:
            Dictionary with taxonomic levels
        """
        # Initialize taxonomy dict
        taxonomy = {
            'kingdom': None,
            'phylum': None,
            'class': None,
            'order': None,
            'family': None,
            'genus': None,
            'species': None
        }
        
        # Simple parsing - can be enhanced based on database format
        try:
            # Look for species name pattern (Genus species)
            import re
            species_pattern = r'([A-Z][a-z]+\s+[a-z]+)'
            species_match = re.search(species_pattern, hit_title)
            
            if species_match:
                species_name = species_match.group(1)
                parts = species_name.split()
                if len(parts) >= 2:
                    taxonomy['genus'] = parts[0]
                    taxonomy['species'] = species_name
            
        except Exception as e:
            logger.warning(f"Error extracting taxonomy from hit title: {e}")
        
        return taxonomy

# Convenience functions
def get_blast_runner(config_section: Optional[Dict[str, Any]] = None) -> WindowsBLASTRunner:
    """Get a BLAST runner instance"""
    return WindowsBLASTRunner(config_section)

def create_database(fasta_file: str, database_name: str, database_type: str = 'nucl') -> bool:
    """Create a BLAST database"""
    runner = get_blast_runner()
    return runner.create_blast_database(fasta_file, database_name, database_type)

def search_sequences(sequences: List[str], 
                    database_path: str,
                    sequence_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Search sequences against a BLAST database"""
    runner = get_blast_runner()
    return runner.run_blastn_search(sequences, database_path, sequence_ids)