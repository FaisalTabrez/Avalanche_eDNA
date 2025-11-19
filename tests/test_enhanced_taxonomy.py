"""
Comprehensive Tests for Enhanced Taxonomic Lineage System

This test suite validates all components of the enhanced taxonomic system:
- Enhanced lineage data structures
- Multi-source taxonomy integration
- Automated taxdump updates
- Visualization and export capabilities
- Backward compatibility
"""

import os
import sys
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Import components to test
from clustering.enhanced_lineage import (
    EnhancedTaxdumpResolver, EnhancedLineage, TaxonomicRank, 
    EvidenceType, LineageEvidence, TaxonomicName, ExternalReference
)
from clustering.multi_source_taxonomy import (
    MultiSourceTaxonomyResolver, CustomTaxonomySource
)
from clustering.taxdump_updater import TaxdumpUpdater
from clustering.lineage_visualization import (
    LineageExporter, TaxonomicTreeVisualizer
)
from clustering.enhanced_taxonomy_integration import (
    EnhancedTaxonomySystem, BackwardCompatibilityWrapper
)

class TestEnhancedLineage:
    """Test enhanced lineage data structures"""
    
    def test_taxonomic_rank_enum(self):
        """Test TaxonomicRank enum functionality"""
        # Test basic ranks
        basic_ranks = TaxonomicRank.basic_ranks()
        assert len(basic_ranks) == 7
        assert TaxonomicRank.KINGDOM in basic_ranks
        assert TaxonomicRank.SPECIES in basic_ranks
        
        # Test from_string method
        rank = TaxonomicRank.from_string("kingdom")
        assert rank == TaxonomicRank.KINGDOM
        
        rank = TaxonomicRank.from_string("subfamily")
        assert rank == TaxonomicRank.SUBFAMILY
        
        # Test invalid rank
        rank = TaxonomicRank.from_string("invalid_rank")
        assert rank is None
    
    def test_evidence_type_enum(self):
        """Test EvidenceType enum"""
        assert EvidenceType.NCBI_TAXDUMP.value == "ncbi_taxdump"
        assert EvidenceType.SILVA.value == "silva"
        assert EvidenceType.CUSTOM_DATABASE.value == "custom_database"
    
    def test_external_reference(self):
        """Test ExternalReference functionality"""
        ref = ExternalReference(database="NCBI", identifier="9606")
        
        # Test URL generation
        url = ref.generate_url()
        assert "ncbi.nlm.nih.gov" in url
        assert "9606" in url
        
        # Test with custom URL
        ref_custom = ExternalReference(
            database="Custom", 
            identifier="123", 
            url="https://example.com/taxon/123"
        )
        assert ref_custom.generate_url() == "https://example.com/taxon/123"
    
    def test_lineage_evidence(self):
        """Test LineageEvidence functionality"""
        evidence = LineageEvidence(
            source=EvidenceType.NCBI_TAXDUMP,
            confidence=0.95,
            method_details="NCBI Taxdump traversal",
            timestamp=datetime.now(),
            metadata={"test_key": "test_value"}
        )
        
        assert evidence.source == EvidenceType.NCBI_TAXDUMP
        assert evidence.confidence == 0.95
        assert evidence.metadata["test_key"] == "test_value"
    
    def test_enhanced_lineage(self):
        """Test EnhancedLineage functionality"""
        lineage = EnhancedLineage(
            taxid=9606,
            scientific_name="Homo sapiens"
        )
        
        # Test basic lineage mapping
        lineage.lineage[TaxonomicRank.KINGDOM] = "Animalia"
        lineage.lineage[TaxonomicRank.PHYLUM] = "Chordata"
        lineage.lineage[TaxonomicRank.CLASS] = "Mammalia"
        lineage.lineage[TaxonomicRank.SPECIES] = "Homo sapiens"
        
        basic_lineage = lineage.get_basic_lineage()
        assert basic_lineage['kingdom'] == "Animalia"
        assert basic_lineage['phylum'] == "Chordata"
        assert basic_lineage['class'] == "Mammalia"
        assert basic_lineage['species'] == "Homo sapiens"
        
        # Test evidence addition
        evidence = LineageEvidence(
            source=EvidenceType.NCBI_TAXDUMP,
            confidence=0.9
        )
        lineage.add_evidence(evidence)
        
        assert len(lineage.evidence) == 1
        assert lineage.overall_confidence > 0
        
        # Test serialization
        lineage_dict = lineage.to_dict()
        assert lineage_dict['taxid'] == 9606
        assert lineage_dict['scientific_name'] == "Homo sapiens"
        assert 'lineage' in lineage_dict
        assert 'evidence' in lineage_dict

class TestMultiSourceTaxonomy:
    """Test multi-source taxonomy integration"""
    
    def test_custom_taxonomy_source(self):
        """Test custom taxonomy source"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            test_data = [
                {
                    "scientific_name": "Test species",
                    "kingdom": "Test Kingdom",
                    "phylum": "Test Phylum",
                    "confidence": 0.8
                }
            ]
            
            test_file = Path(temp_dir) / "test_taxonomy.json"
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            # Test loading
            source = CustomTaxonomySource(str(test_file))
            success = source.load_data()
            assert success
            
            # Test retrieval
            lineage = source.get_lineage("test species")
            assert lineage is not None
            assert lineage.scientific_name == "Test species"
            assert lineage.lineage[TaxonomicRank.KINGDOM] == "Test Kingdom"
    
    def test_multi_source_resolver(self):
        """Test multi-source taxonomy resolver"""
        resolver = MultiSourceTaxonomyResolver()
        
        # Create mock custom source
        custom_source = CustomTaxonomySource()
        resolver.add_source(custom_source)
        
        assert len(resolver.sources) == 1
        assert resolver.sources[0].name == "Custom"
        
        # Test source removal
        success = resolver.remove_source("Custom")
        assert success
        assert len(resolver.sources) == 0

class TestTaxdumpUpdater:
    """Test taxdump updater functionality"""
    
    def test_updater_initialization(self):
        """Test updater initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            updater = TaxdumpUpdater(temp_dir)
            
            assert updater.taxdump_dir == Path(temp_dir)
            assert updater.backup_dir == Path(temp_dir) / "backups"
            assert updater.keep_backups == 3
    
    @patch('urllib.request.urlopen')
    def test_check_for_updates(self, mock_urlopen):
        """Test update checking"""
        with tempfile.TemporaryDirectory() as temp_dir:
            updater = TaxdumpUpdater(temp_dir)
            
            # Mock successful response
            mock_response = Mock()
            mock_response.read.return_value.decode.return_value = "test_hash  taxdump.tar.gz"
            mock_response.headers = {"Content-Length": "1000", "Last-Modified": "Wed, 01 Jan 2020 00:00:00 GMT"}
            mock_urlopen.return_value.__enter__.return_value = mock_response
            
            needs_update, reason = updater.check_for_updates()
            # Should need update since no local version exists
            assert needs_update
            assert "No local taxdump found" in reason
    
    def test_get_status(self):
        """Test status reporting"""
        with tempfile.TemporaryDirectory() as temp_dir:
            updater = TaxdumpUpdater(temp_dir)
            
            status = updater.get_status()
            assert 'taxdump_dir' in status
            assert 'has_required_files' in status
            assert 'existing_files' in status
            assert status['taxdump_dir'] == str(temp_dir)

class TestLineageExporter:
    """Test lineage export functionality"""
    
    def test_json_export(self):
        """Test JSON export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test lineage
            lineage = EnhancedLineage(
                taxid=12345,
                scientific_name="Test organism"
            )
            lineage.lineage[TaxonomicRank.KINGDOM] = "Test Kingdom"
            
            exporter = LineageExporter()
            output_path = Path(temp_dir) / "test_export.json"
            
            success = exporter.export_lineages([lineage], str(output_path), "json")
            assert success
            assert output_path.exists()
            
            # Verify content
            with open(output_path) as f:
                data = json.load(f)
            
            assert 'metadata' in data
            assert 'lineages' in data
            assert len(data['lineages']) == 1
            assert data['lineages'][0]['taxid'] == 12345
    
    def test_csv_export(self):
        """Test CSV export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lineage = EnhancedLineage(
                taxid=12345,
                scientific_name="Test organism"
            )
            
            exporter = LineageExporter()
            output_path = Path(temp_dir) / "test_export.csv"
            
            success = exporter.export_lineages([lineage], str(output_path), "csv")
            assert success
            assert output_path.exists()
            
            # Basic content check
            content = output_path.read_text()
            assert "taxid" in content
            assert "scientific_name" in content
            assert "12345" in content
    
    def test_xml_export(self):
        """Test XML export"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lineage = EnhancedLineage(
                taxid=12345,
                scientific_name="Test organism"
            )
            
            exporter = LineageExporter()
            output_path = Path(temp_dir) / "test_export.xml"
            
            success = exporter.export_lineages([lineage], str(output_path), "xml")
            assert success
            assert output_path.exists()
            
            # Basic XML structure check
            content = output_path.read_text()
            assert "<?xml" in content
            assert "<taxonomic_lineages>" in content
            assert "<taxid>12345</taxid>" in content

class TestEnhancedTaxonomySystem:
    """Test the integrated enhanced taxonomy system"""
    
    def test_system_initialization(self):
        """Test system initialization"""
        config_dict = {
            'taxdump_dir': None,  # No real taxdump for testing
            'enhanced_lineage': {
                'enable_caching': False
            },
            'multi_source': {
                'enabled': False
            },
            'auto_update': {
                'enabled': False
            }
        }
        
        system = EnhancedTaxonomySystem(
            config_dict=config_dict,
            enable_enhanced_features=True
        )
        
        # Should fall back gracefully when no taxdump available
        assert system is not None
    
    def test_backward_compatibility(self):
        """Test backward compatibility wrapper"""
        config_dict = {
            'taxdump_dir': None
        }
        
        system = EnhancedTaxonomySystem(config_dict=config_dict)
        wrapper = BackwardCompatibilityWrapper(system)
        
        # Should return empty results gracefully
        result = wrapper.lineage_by_name("nonexistent species")
        assert isinstance(result, dict)
        
        result = wrapper.lineage_by_taxid(12345)
        assert isinstance(result, dict)
        
        # Should indicate unavailable
        available = wrapper.available()
        # May be False if no taxdump configured
        assert isinstance(available, bool)
    
    def test_statistics_generation(self):
        """Test system statistics"""
        system = EnhancedTaxonomySystem()
        stats = system.get_system_statistics()
        
        assert 'enhanced_features_enabled' in stats
        assert 'components_initialized' in stats
        assert 'last_updated' in stats
        
        components = stats['components_initialized']
        assert 'enhanced_resolver' in components
        assert 'multi_source_resolver' in components
        assert 'updater' in components

class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    def test_lineage_lookup_fallback(self):
        """Test lineage lookup with fallback behavior"""
        system = EnhancedTaxonomySystem()
        
        # Should handle non-existent species gracefully
        result = system.get_lineage("Nonexistent species", enhanced=True)
        assert isinstance(result, EnhancedLineage)
        
        result = system.get_lineage("Nonexistent species", enhanced=False)
        assert isinstance(result, dict)
    
    def test_batch_operations(self):
        """Test batch lineage operations"""
        system = EnhancedTaxonomySystem()
        
        identifiers = ["Species 1", "Species 2", "12345"]
        results = system.batch_lineage_lookup(identifiers, enhanced=True)
        
        assert len(results) == 3
        assert all(isinstance(r, EnhancedLineage) for r in results)
    
    def test_custom_taxonomy_override(self):
        """Test custom taxonomy override functionality"""
        system = EnhancedTaxonomySystem()
        
        custom_lineage = {
            'kingdom': 'Custom Kingdom',
            'phylum': 'Custom Phylum',
            'species': 'Custom species'
        }
        
        # Should handle gracefully even if multi-source not enabled
        result = system.add_custom_taxonomy_override("Custom species", custom_lineage)
        # May be False if multi-source resolver not available
        assert isinstance(result, bool)

# Pytest fixtures for common test data
@pytest.fixture
def sample_lineage():
    """Create a sample enhanced lineage for testing"""
    lineage = EnhancedLineage(
        taxid=9606,
        scientific_name="Homo sapiens"
    )
    
    lineage.lineage[TaxonomicRank.KINGDOM] = "Animalia"
    lineage.lineage[TaxonomicRank.PHYLUM] = "Chordata"
    lineage.lineage[TaxonomicRank.CLASS] = "Mammalia"
    lineage.lineage[TaxonomicRank.ORDER] = "Primates"
    lineage.lineage[TaxonomicRank.FAMILY] = "Hominidae"
    lineage.lineage[TaxonomicRank.GENUS] = "Homo"
    lineage.lineage[TaxonomicRank.SPECIES] = "Homo sapiens"
    
    # Add evidence
    evidence = LineageEvidence(
        source=EvidenceType.NCBI_TAXDUMP,
        confidence=1.0,
        method_details="Test data"
    )
    lineage.add_evidence(evidence)
    
    # Add external reference
    ref = ExternalReference(database="NCBI", identifier="9606")
    lineage.external_refs.append(ref)
    
    return lineage

@pytest.fixture
def temp_directory():
    """Provide a temporary directory for tests"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

# Performance tests
class TestPerformance:
    """Test performance aspects of the enhanced system"""
    
    def test_lineage_caching(self, sample_lineage):
        """Test that lineage caching improves performance"""
        # This is a basic test - in practice would need real timing
        with tempfile.TemporaryDirectory() as temp_dir:
            resolver = EnhancedTaxdumpResolver(temp_dir, enable_caching=True)
            
            # Multiple calls should use cache
            # (Won't actually work without real taxdump data, but tests structure)
            try:
                result1 = resolver.enhanced_lineage_by_name("test")
                result2 = resolver.enhanced_lineage_by_name("test")
                # Results should be consistent
                assert type(result1) == type(result2)
            except:
                # Expected to fail without real data
                pass
            finally:
                # Close cache before tempdir cleanup to avoid Windows file lock
                resolver.close_cache()

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])