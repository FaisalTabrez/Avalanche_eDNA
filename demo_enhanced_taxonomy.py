#!/usr/bin/env python3
"""
Enhanced Taxonomy System Demonstration

This script demonstrates the new enhanced taxonomic lineage features:
- Extended taxonomic ranks (28 vs basic 7)
- Common names and synonyms
- Confidence scoring and evidence tracking
- Multi-source taxonomy integration
- Enhanced visualization capabilities
- Export in multiple formats
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    # Import enhanced taxonomy components
    from clustering.enhanced_lineage import (
        EnhancedTaxdumpResolver, EnhancedLineage, TaxonomicRank, 
        EvidenceType, LineageEvidence, TaxonomicName, ExternalReference
    )
    from clustering.enhanced_taxonomy_integration import (
        EnhancedTaxonomySystem, create_enhanced_taxonomy_system
    )
    from clustering.lineage_visualization import LineageExporter
    print("✅ Enhanced taxonomy modules imported successfully!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("The enhanced taxonomy system modules are not available.")
    sys.exit(1)

def demonstrate_enhanced_features():
    """Demonstrate the enhanced taxonomy features"""
    
    print("\n" + "="*60)
    print("🧬 Enhanced Taxonomic Lineage System Demonstration")
    print("="*60)
    
    # 1. Demonstrate extended taxonomic ranks
    print("\n1. 📊 Extended Taxonomic Ranks")
    print("-" * 30)
    
    basic_ranks = TaxonomicRank.basic_ranks()
    print(f"Basic ranks (old system): {len(basic_ranks)}")
    for rank in basic_ranks:
        print(f"   • {rank.rank_name}")
    
    all_ranks = list(TaxonomicRank)
    print(f"\nExtended ranks (new system): {len(all_ranks)}")
    print("Additional intermediate ranks:")
    extended_only = [r for r in all_ranks if r not in basic_ranks]
    for rank in extended_only[:10]:  # Show first 10
        print(f"   • {rank.rank_name}")
    print(f"   ... and {len(extended_only)-10} more intermediate ranks")
    
    # 2. Create sample enhanced lineage
    print("\n2. 🔬 Enhanced Lineage Example")
    print("-" * 30)
    
    # Create a detailed lineage for Homo sapiens
    human_lineage = EnhancedLineage(
        taxid=9606,
        scientific_name="Homo sapiens",
        last_updated=datetime.now()
    )
    
    # Add comprehensive taxonomic hierarchy
    human_lineage.lineage[TaxonomicRank.SUPERKINGDOM] = "Eukaryota"
    human_lineage.lineage[TaxonomicRank.KINGDOM] = "Animalia"
    human_lineage.lineage[TaxonomicRank.SUBKINGDOM] = "Eumetazoa"
    human_lineage.lineage[TaxonomicRank.PHYLUM] = "Chordata"
    human_lineage.lineage[TaxonomicRank.SUBPHYLUM] = "Vertebrata"
    human_lineage.lineage[TaxonomicRank.SUPERCLASS] = "Tetrapoda"
    human_lineage.lineage[TaxonomicRank.CLASS] = "Mammalia"
    human_lineage.lineage[TaxonomicRank.SUBCLASS] = "Theria"
    human_lineage.lineage[TaxonomicRank.INFRACLASS] = "Eutheria"
    human_lineage.lineage[TaxonomicRank.SUPERORDER] = "Euarchontoglires"
    human_lineage.lineage[TaxonomicRank.ORDER] = "Primates"
    human_lineage.lineage[TaxonomicRank.SUBORDER] = "Simiiformes"
    human_lineage.lineage[TaxonomicRank.INFRAORDER] = "Catarrhini"
    human_lineage.lineage[TaxonomicRank.SUPERFAMILY] = "Hominoidea"
    human_lineage.lineage[TaxonomicRank.FAMILY] = "Hominidae"
    human_lineage.lineage[TaxonomicRank.SUBFAMILY] = "Homininae"
    human_lineage.lineage[TaxonomicRank.TRIBE] = "Hominini"
    human_lineage.lineage[TaxonomicRank.GENUS] = "Homo"
    human_lineage.lineage[TaxonomicRank.SPECIES] = "Homo sapiens"
    
    # Add names and synonyms
    human_lineage.names = TaxonomicName(
        scientific_name="Homo sapiens",
        common_names=["Human", "Modern human", "Anatomically modern human"],
        synonyms=["Homo sapiens sapiens"],
        authority="Linnaeus, 1758"
    )
    
    # Add evidence from multiple sources
    evidence1 = LineageEvidence(
        source=EvidenceType.NCBI_TAXDUMP,
        confidence=1.0,
        method_details="NCBI Taxonomy database",
        timestamp=datetime.now(),
        metadata={"database_version": "2024.01"}
    )
    human_lineage.add_evidence(evidence1)
    
    evidence2 = LineageEvidence(
        source=EvidenceType.MANUAL_CURATION,
        confidence=0.95,
        method_details="Expert taxonomic review",
        timestamp=datetime.now(),
        metadata={"reviewer": "Taxonomic expert", "review_date": "2024-01-15"}
    )
    human_lineage.add_evidence(evidence2)
    
    # Add external references
    refs = [
        ExternalReference(database="NCBI", identifier="9606"),
        ExternalReference(database="ITIS", identifier="180092"),
        ExternalReference(database="GBIF", identifier="2436436"),
        ExternalReference(database="EOL", identifier="327955")
    ]
    human_lineage.external_refs.extend(refs)
    
    print(f"Scientific name: {human_lineage.scientific_name}")
    print(f"NCBI TaxID: {human_lineage.taxid}")
    print(f"Overall confidence: {human_lineage.overall_confidence:.3f}")
    print(f"Common names: {', '.join(human_lineage.names.common_names)}")
    print(f"Authority: {human_lineage.names.authority}")
    print(f"Evidence sources: {len(human_lineage.evidence)}")
    print(f"External references: {len(human_lineage.external_refs)}")
    
    print("\nComplete taxonomic hierarchy:")
    for rank, name in human_lineage.lineage.items():
        if name:
            print(f"   {rank.rank_name:15} : {name}")
    
    # 3. Demonstrate backward compatibility
    print("\n3. 🔄 Backward Compatibility")
    print("-" * 30)
    
    basic_lineage = human_lineage.get_basic_lineage()
    print("Basic 7-rank lineage (compatible with old system):")
    for rank, name in basic_lineage.items():
        if name:
            print(f"   {rank:10} : {name}")
    
    # 4. Demonstrate external references
    print("\n4. 🔗 External Database Links")
    print("-" * 30)
    
    print("Clickable links to external databases:")
    for ref in human_lineage.external_refs[:3]:  # Show first 3
        url = ref.generate_url()
        print(f"   {ref.database:8} : {url}")
    
    # 5. Export demonstration
    print("\n5. 📤 Export Capabilities")
    print("-" * 30)
    
    exporter = LineageExporter()
    supported_formats = exporter.supported_formats
    print(f"Supported export formats ({len(supported_formats)}):")
    for fmt in supported_formats:
        print(f"   • {fmt.upper()}")
    
    # Export sample data
    output_dir = Path("demo_exports")
    output_dir.mkdir(exist_ok=True)
    
    sample_lineages = [human_lineage]
    
    # Export to JSON
    json_file = output_dir / "sample_lineage.json"
    success = exporter.export_lineages(sample_lineages, str(json_file), "json")
    if success:
        print(f"\n✅ Exported to JSON: {json_file}")
    
    # Export to XML
    xml_file = output_dir / "sample_lineage.xml"
    success = exporter.export_lineages(sample_lineages, str(xml_file), "xml")
    if success:
        print(f"✅ Exported to XML: {xml_file}")
    
    # Export to JSON-LD (Linked Data)
    jsonld_file = output_dir / "sample_lineage.jsonld"
    success = exporter.export_lineages(sample_lineages, str(jsonld_file), "json-ld")
    if success:
        print(f"✅ Exported to JSON-LD: {jsonld_file}")
    
    # 6. Integration system demonstration
    print("\n6. 🔧 Integrated Taxonomy System")
    print("-" * 30)
    
    try:
        # Create integrated system (will work even without taxdump files)
        taxonomy_system = create_enhanced_taxonomy_system()
        stats = taxonomy_system.get_system_statistics()
        
        print("System components status:")
        for component, status in stats['components_initialized'].items():
            emoji = "✅" if status else "❌"
            print(f"   {emoji} {component.replace('_', ' ').title()}")
        
        print(f"\nEnhanced features enabled: {'✅ Yes' if stats['enhanced_features_enabled'] else '❌ No'}")
        
    except Exception as e:
        print(f"Integration system demo failed: {e}")
    
    print("\n" + "="*60)
    print("✅ Enhanced Taxonomy System demonstration complete!")
    print("="*60)
    
    print(f"""
🚀 Next Steps:
   1. Visit the web UI at: http://localhost:8503
   2. Upload a FASTA/FASTQ file for analysis
   3. Check the enhanced taxonomy features in action
   4. View exported files in: {output_dir.absolute()}
   5. Explore the advanced visualization options
    """)

if __name__ == "__main__":
    demonstrate_enhanced_features()