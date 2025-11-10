"""
Enhanced Taxonomy Integration Module

This module provides seamless integration between the original taxonomy system 
and the new enhanced lineage system, ensuring backward compatibility while 
enabling advanced features.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import config

# Import original taxonomy components
from .taxonomy import TaxdumpResolver, HybridTaxonomyAssigner

# Import enhanced lineage components
from .enhanced_lineage import (
    EnhancedTaxdumpResolver, EnhancedLineage, TaxonomicRank, 
    EvidenceType, LineageEvidence, create_enhanced_resolver
)
from .multi_source_taxonomy import (
    MultiSourceTaxonomyResolver, create_standard_resolver
)
from .taxdump_updater import TaxdumpUpdater, create_updater_from_config
from .lineage_visualization import (
    TaxonomicTreeVisualizer, LineageExporter, LineageReportGenerator,
    create_visualizer_from_config
)

# Setup logging
logger = logging.getLogger(__name__)

class EnhancedTaxonomySystem:
    """Unified interface for enhanced taxonomic lineage system"""
    
    def __init__(self, 
                 config_dict: Optional[Dict[str, Any]] = None,
                 enable_enhanced_features: bool = True):
        
        self.config_dict = config_dict or config.get('taxonomy', {})
        self.enable_enhanced = enable_enhanced_features
        
        # Initialize components
        self.enhanced_resolver: Optional[EnhancedTaxdumpResolver] = None
        self.multi_source_resolver: Optional[MultiSourceTaxonomyResolver] = None
        self.updater: Optional[TaxdumpUpdater] = None
        self.visualizer: Optional[TaxonomicTreeVisualizer] = None
        self.exporter: Optional[LineageExporter] = None
        self.report_generator: Optional[LineageReportGenerator] = None
        
        # Backward compatibility - original resolver
        self.legacy_resolver: Optional[TaxdumpResolver] = None
        
        # Initialize based on configuration
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all taxonomy system components"""
        try:
            # Initialize enhanced resolver if enabled
            if self.enable_enhanced:
                self._initialize_enhanced_resolver()
                self._initialize_multi_source_resolver()
                self._initialize_updater()
                self._initialize_visualization()
            
            # Always initialize legacy resolver for backward compatibility
            self._initialize_legacy_resolver()
            
            logger.info(f"Enhanced taxonomy system initialized (enhanced: {self.enable_enhanced})")
            
        except Exception as e:
            logger.error(f"Error initializing taxonomy system: {e}")
            # Fall back to legacy resolver only
            self.enable_enhanced = False
            self._initialize_legacy_resolver()
    
    def _initialize_enhanced_resolver(self) -> None:
        """Initialize enhanced taxdump resolver"""
        try:
            taxdump_dir = self.config_dict.get('taxdump_dir')
            enhanced_config = self.config_dict.get('enhanced_lineage', {})
            
            if taxdump_dir:
                self.enhanced_resolver = EnhancedTaxdumpResolver(
                    taxdump_dir=taxdump_dir,
                    enable_caching=enhanced_config.get('enable_caching', True)
                )
                logger.info("Enhanced taxdump resolver initialized")
            else:
                logger.warning("No taxdump_dir configured - enhanced resolver disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize enhanced resolver: {e}")
    
    def _initialize_multi_source_resolver(self) -> None:
        """Initialize multi-source taxonomy resolver"""
        try:
            multi_source_config = self.config_dict.get('multi_source', {})
            
            if multi_source_config.get('enabled', False):
                self.multi_source_resolver = create_standard_resolver(multi_source_config)
                logger.info("Multi-source resolver initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-source resolver: {e}")
    
    def _initialize_updater(self) -> None:
        """Initialize taxdump updater"""
        try:
            auto_update_config = self.config_dict.get('auto_update', {})
            
            if auto_update_config.get('enabled', False):
                self.updater = create_updater_from_config()
                logger.info("Taxdump updater initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize updater: {e}")
    
    def _initialize_visualization(self) -> None:
        """Initialize visualization components"""
        try:
            viz_config = config.get('visualization', {}).get('lineage_viz', {})
            
            if viz_config:
                self.visualizer = create_visualizer_from_config()
                self.exporter = LineageExporter()
                self.report_generator = LineageReportGenerator()
                logger.info("Visualization components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize visualization: {e}")
    
    def _initialize_legacy_resolver(self) -> None:
        """Initialize legacy taxdump resolver"""
        try:
            taxdump_dir = self.config_dict.get('taxdump_dir')
            
            if taxdump_dir:
                self.legacy_resolver = TaxdumpResolver(taxdump_dir)
                logger.info("Legacy taxdump resolver initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize legacy resolver: {e}")
    
    def get_lineage(self, 
                   identifier: str,
                   enhanced: Optional[bool] = None,
                   prefer_source: Optional[str] = None) -> Union[EnhancedLineage, Dict[str, Optional[str]]]:
        """
        Get taxonomic lineage for an identifier
        
        Args:
            identifier: Scientific name or taxonomic ID
            enhanced: Return enhanced lineage if True, basic if False, auto-detect if None
            prefer_source: Preferred taxonomy source for multi-source resolution
            
        Returns:
            EnhancedLineage object or basic lineage dictionary
        """
        use_enhanced = enhanced if enhanced is not None else self.enable_enhanced
        
        if use_enhanced and self.enhanced_resolver:
            try:
                # Try enhanced resolver first
                if identifier.isdigit():
                    # Numeric ID - treat as taxid
                    lineage = self.enhanced_resolver.enhanced_lineage_by_taxid(int(identifier))
                else:
                    # Text - treat as scientific name
                    lineage = self.enhanced_resolver.enhanced_lineage_by_name(identifier)
                
                # Try multi-source resolution if primary failed
                if (not lineage or not lineage.scientific_name) and self.multi_source_resolver:
                    ms_lineage = self.multi_source_resolver.resolve_lineage(
                        identifier, prefer_source=prefer_source
                    )
                    if ms_lineage:
                        lineage = ms_lineage
                
                if lineage:
                    return lineage
                    
            except Exception as e:
                logger.debug(f"Enhanced lineage lookup failed: {e}")
        
        # Fall back to legacy resolver
        if self.legacy_resolver:
            try:
                if identifier.isdigit():
                    return self.legacy_resolver.lineage_by_taxid(int(identifier))
                else:
                    return self.legacy_resolver.lineage_by_name(identifier)
            except Exception as e:
                logger.debug(f"Legacy lineage lookup failed: {e}")
        
        # Return empty result
        if use_enhanced:
            return EnhancedLineage()
        else:
            return {r: None for r in ['kingdom','phylum','class','order','family','genus','species']}
    
    def search_by_common_name(self, common_name: str) -> List[EnhancedLineage]:
        """Search for taxa by common name"""
        results = []
        
        if self.enhanced_resolver:
            try:
                results.extend(self.enhanced_resolver.search_by_common_name(common_name))
            except Exception as e:
                logger.debug(f"Enhanced common name search failed: {e}")
        
        if self.multi_source_resolver and not results:
            try:
                ms_results = self.multi_source_resolver.search_all_sources(common_name)
                results.extend([lineage for _, lineage in ms_results])
            except Exception as e:
                logger.debug(f"Multi-source search failed: {e}")
        
        return results
    
    def batch_lineage_lookup(self, 
                           identifiers: List[str],
                           enhanced: bool = True) -> List[Union[EnhancedLineage, Dict[str, Optional[str]]]]:
        """Perform batch lineage lookup for multiple identifiers"""
        results = []
        
        for identifier in identifiers:
            lineage = self.get_lineage(identifier, enhanced=enhanced)
            results.append(lineage)
        
        return results
    
    def check_for_updates(self) -> Tuple[bool, str]:
        """Check if taxdump updates are available"""
        if not self.updater:
            return False, "Automatic updates not configured"
        
        try:
            return self.updater.check_for_updates()
        except Exception as e:
            return False, f"Update check failed: {e}"
    
    def update_taxdump(self, force: bool = False) -> Tuple[bool, str]:
        """Update taxdump files"""
        if not self.updater:
            return False, "Automatic updates not configured"
        
        try:
            success, message = self.updater.download_and_update(force=force)
            
            # Reinitialize resolvers after successful update
            if success:
                self._reinitialize_after_update()
            
            return success, message
        except Exception as e:
            return False, f"Update failed: {e}"
    
    def _reinitialize_after_update(self) -> None:
        """Reinitialize resolvers after taxdump update"""
        try:
            if self.enhanced_resolver:
                # Clear cache and reinitialize
                self.enhanced_resolver._name_to_taxid = None
                self.enhanced_resolver._taxid_to_name = None
                self.enhanced_resolver._nodes = None
                logger.info("Enhanced resolver reinitialized after update")
            
            if self.legacy_resolver:
                # Clear cache
                self.legacy_resolver._name_to_taxid = None
                self.legacy_resolver._taxid_to_name = None
                self.legacy_resolver._nodes = None
                logger.info("Legacy resolver reinitialized after update")
                
        except Exception as e:
            logger.error(f"Error reinitializing after update: {e}")
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            'enhanced_features_enabled': self.enable_enhanced,
            'components_initialized': {
                'enhanced_resolver': self.enhanced_resolver is not None,
                'multi_source_resolver': self.multi_source_resolver is not None,
                'updater': self.updater is not None,
                'visualizer': self.visualizer is not None,
                'legacy_resolver': self.legacy_resolver is not None
            },
            'last_updated': datetime.now().isoformat()
        }
        
        # Enhanced resolver stats
        if self.enhanced_resolver:
            try:
                enhanced_stats = self.enhanced_resolver.get_lineage_statistics()
                stats['enhanced_resolver'] = enhanced_stats
            except Exception as e:
                logger.debug(f"Error getting enhanced resolver stats: {e}")
        
        # Multi-source resolver stats
        if self.multi_source_resolver:
            try:
                ms_stats = self.multi_source_resolver.get_source_statistics()
                stats['multi_source'] = ms_stats
            except Exception as e:
                logger.debug(f"Error getting multi-source stats: {e}")
        
        # Updater stats
        if self.updater:
            try:
                update_stats = self.updater.get_status()
                stats['updater'] = update_stats
            except Exception as e:
                logger.debug(f"Error getting updater stats: {e}")
        
        return stats
    
    def create_visualization_report(self, 
                                  lineages: List[EnhancedLineage],
                                  output_dir: str,
                                  report_name: str = "taxonomy_report") -> Optional[str]:
        """Create comprehensive visualization report"""
        if not self.report_generator:
            logger.error("Report generator not initialized")
            return None
        
        try:
            report_path = self.report_generator.generate_comprehensive_report(
                lineages, output_dir, report_name
            )
            return report_path
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return None
    
    def export_lineages(self, 
                       lineages: List[EnhancedLineage],
                       output_path: str,
                       format_type: str = 'json') -> bool:
        """Export lineages in specified format"""
        if not self.exporter:
            logger.error("Exporter not initialized")
            return False
        
        try:
            return self.exporter.export_lineages(lineages, output_path, format_type)
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False
    
    def add_custom_taxonomy_override(self, 
                                   scientific_name: str,
                                   custom_lineage: Dict[str, str]) -> bool:
        """Add custom taxonomy override"""
        if not self.multi_source_resolver:
            logger.error("Multi-source resolver required for custom overrides")
            return False
        
        try:
            # Find custom source
            custom_source = None
            for source in self.multi_source_resolver.sources:
                if hasattr(source, 'add_override'):
                    custom_source = source
                    break
            
            if not custom_source:
                logger.error("No custom taxonomy source available")
                return False
            
            # Create enhanced lineage from custom data
            enhanced_lineage = EnhancedLineage(
                scientific_name=scientific_name,
                last_updated=datetime.now()
            )
            
            # Map basic ranks
            rank_mapping = {
                'kingdom': TaxonomicRank.KINGDOM,
                'phylum': TaxonomicRank.PHYLUM,
                'class': TaxonomicRank.CLASS,
                'order': TaxonomicRank.ORDER,
                'family': TaxonomicRank.FAMILY,
                'genus': TaxonomicRank.GENUS,
                'species': TaxonomicRank.SPECIES
            }
            
            for rank_name, rank_enum in rank_mapping.items():
                if rank_name in custom_lineage and custom_lineage[rank_name]:
                    enhanced_lineage.lineage[rank_enum] = custom_lineage[rank_name]
            
            # Add as override
            custom_source.add_override(scientific_name, enhanced_lineage)
            
            logger.info(f"Added custom taxonomy override for {scientific_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add custom override: {e}")
            return False

# Factory function for easy initialization
def create_enhanced_taxonomy_system(config_dict: Optional[Dict[str, Any]] = None) -> EnhancedTaxonomySystem:
    """Create enhanced taxonomy system with automatic configuration"""
    return EnhancedTaxonomySystem(config_dict)

# Backward compatibility wrapper
class BackwardCompatibilityWrapper:
    """Wrapper to maintain backward compatibility with existing code"""
    
    def __init__(self, enhanced_system: EnhancedTaxonomySystem):
        self.system = enhanced_system
    
    def lineage_by_name(self, scientific_name: str) -> Dict[str, Optional[str]]:
        """Legacy interface for lineage by name"""
        result = self.system.get_lineage(scientific_name, enhanced=False)
        if isinstance(result, dict):
            return result
        else:
            return result.get_basic_lineage() if hasattr(result, 'get_basic_lineage') else {}
    
    def lineage_by_taxid(self, taxid: int) -> Dict[str, Optional[str]]:
        """Legacy interface for lineage by taxid"""
        result = self.system.get_lineage(str(taxid), enhanced=False)
        if isinstance(result, dict):
            return result
        else:
            return result.get_basic_lineage() if hasattr(result, 'get_basic_lineage') else {}
    
    def available(self) -> bool:
        """Check if taxonomy system is available"""
        return (self.system.enhanced_resolver is not None or 
                self.system.legacy_resolver is not None)

# Global instance for backward compatibility
_global_enhanced_system: Optional[EnhancedTaxonomySystem] = None
_global_compatibility_wrapper: Optional[BackwardCompatibilityWrapper] = None

def get_enhanced_taxonomy_system() -> EnhancedTaxonomySystem:
    """Get global enhanced taxonomy system instance"""
    global _global_enhanced_system
    
    if _global_enhanced_system is None:
        _global_enhanced_system = create_enhanced_taxonomy_system()
    
    return _global_enhanced_system

def get_compatibility_wrapper() -> BackwardCompatibilityWrapper:
    """Get backward compatibility wrapper"""
    global _global_compatibility_wrapper
    
    if _global_compatibility_wrapper is None:
        enhanced_system = get_enhanced_taxonomy_system()
        _global_compatibility_wrapper = BackwardCompatibilityWrapper(enhanced_system)
    
    return _global_compatibility_wrapper

# For easy import and usage
def initialize_enhanced_taxonomy() -> Tuple[EnhancedTaxonomySystem, BackwardCompatibilityWrapper]:
    """Initialize both enhanced system and compatibility wrapper"""
    enhanced_system = get_enhanced_taxonomy_system()
    compatibility_wrapper = get_compatibility_wrapper()
    
    return enhanced_system, compatibility_wrapper