"""
Configuration management for eDNA Biodiversity Assessment System
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional

class Config:
    """Configuration manager for the eDNA analysis system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._resolve_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing configuration file: {e}")

    def _resolve_env_overrides(self) -> None:
        """
        Override YAML config values with environment variables at runtime.

        This implements the 12-factor app config principle: any value that
        differs between environments (paths, credentials, endpoints) must
        come from the environment, not from a committed config file.

        Precedence: environment variable > config.yaml value.
        """
        env_map = {
            # SRA Toolkit tool paths
            "databases.sra.sra_tools.prefetch_path":     "PREFETCH_PATH",
            "databases.sra.sra_tools.fastq_dump_path":   "FASTQ_DUMP_PATH",
            "databases.sra.sra_tools.fasterq_dump_path": "FASTERQ_DUMP_PATH",
            "databases.sra.sra_tools.bin_dir":           "SRA_BIN_DIR",
            # BLAST+ executable paths
            "taxonomy.blast.blast_bin_dir":              "BLAST_BIN_DIR",
            "taxonomy.blast.blastn_path":                "BLASTN_PATH",
            "taxonomy.blast.makeblastdb_path":           "MAKEBLASTDB_PATH",
            "taxonomy.blast_fallback.blastn_path":       "BLASTN_PATH",
            # NCBI taxdump directories
            "taxonomy.taxdump_dir":                      "TAXDUMP_DIR",
            "taxonomy.taxdump_backup_dir":               "TAXDUMP_BACKUP_DIR",
            # Data I/O directories
            "data.raw_dir":                              "DATA_RAW_DIR",
            "data.processed_dir":                        "DATA_PROCESSED_DIR",
            "data.reference_dir":                        "DATA_REFERENCE_DIR",
            "data.output_dir":                           "DATA_OUTPUT_DIR",
            # Storage
            "storage.datasets_dir":                      "ANALYSIS_DATASETS_DIR",
            "storage.runs_dir":                          "ANALYSIS_RUNS_DIR",
            # Logging
            "logging.file":                              "LOG_FILE",
            "logging.level":                             "LOG_LEVEL",
        }
        for config_key, env_var in env_map.items():
            val = os.getenv(env_var)
            if val:
                self.set(config_key, val)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'data.raw_dir')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'data.raw_dir')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """
        Save configuration to file
        
        Args:
            path: Path to save configuration. If None, saves to original path
        """
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access"""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style assignment"""
        self.set(key, value)

# Global configuration instance
config = Config()