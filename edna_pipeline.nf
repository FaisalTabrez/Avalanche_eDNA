#!/usr/bin/env nextflow
// =============================================================================
// Avalanche eDNA - Nextflow Pipeline
// Orchestrates the full eDNA biodiversity assessment at scale
// Compatible with: GKE, GCP Batch, Google Life Sciences, and local execution
// =============================================================================
// Run locally:
//   nextflow run edna_pipeline.nf -profile local
// Run on GCP Batch:
//   nextflow run edna_pipeline.nf -profile gcp_batch -bucket-dir gs://YOUR_BUCKET/work
// Run on GKE:
//   nextflow run edna_pipeline.nf -profile gke
// =============================================================================

nextflow.enable.dsl = 2

// ---------- Pipeline parameters (overridable via --param or nextflow.config) --
params {
    // I/O
    input_dir        = "${baseDir}/data/raw"
    output_dir       = "${baseDir}/data/output"
    reference_dir    = "${baseDir}/reference"

    // eDNA-specific
    min_read_length  = 50
    max_read_length  = 500
    quality_threshold= 20
    kmer_size        = 4

    // Model
    model_id         = "zhihan1996/DNABERT-2-117M"
    batch_size       = 256
    embedding_dim    = 256

    // Taxonomy
    blast_evalue     = 1e-5
    blast_identity   = 97.0
    knn_k            = 50
    knn_min_similarity = 0.65

    // Clustering
    cluster_method   = "hdbscan"
    min_cluster_size = 10

    // GCP
    gcp_project      = System.getenv('GCP_PROJECT_ID') ?: 'your-gcp-project'
    gcp_region       = System.getenv('GCP_REGION') ?: 'us-central1'
    gcs_bucket       = System.getenv('GCS_BUCKET_NAME') ?: 'your-edna-data-bucket'

    // Container image
    container        = "${params.gcp_region}-docker.pkg.dev/${params.gcp_project}/edna-pipeline/avalanche-edna:latest"

    // Resource defaults
    max_cpus         = 32
    max_memory       = '120.GB'
    max_time         = '24.h'

    // Skip flags
    skip_preprocessing = false
    skip_embedding     = false
    skip_clustering    = false
    skip_taxonomy      = false
    skip_novelty       = false
}

// ---------- Process: Preprocessing (Quality Filtering + Adapter Trimming) ----
process PREPROCESS_READS {
    tag "${sample_id}"
    label 'process_medium'

    container params.container
    publishDir "${params.output_dir}/preprocessed", mode: 'copy', pattern: '*.fastq.gz'

    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("${sample_id}_trimmed.fastq.gz"), emit: trimmed_reads
    path "${sample_id}_qc_report.json",                          emit: qc_reports

    script:
    """
    python3 ${baseDir}/scripts/run_pipeline.py \\
        preprocess \\
        --input ${reads} \\
        --output ${sample_id}_trimmed.fastq.gz \\
        --min-length ${params.min_read_length} \\
        --max-length ${params.max_read_length} \\
        --quality-threshold ${params.quality_threshold} \\
        --report ${sample_id}_qc_report.json
    """
}

// ---------- Process: Generate DNABERT-2 Embeddings ---------------------------
process GENERATE_EMBEDDINGS {
    tag "${sample_id}"
    label 'process_gpu'

    container params.container
    publishDir "${params.output_dir}/embeddings", mode: 'copy', pattern: '*.parquet'
    accelerator 1, type: 'nvidia-tesla-t4'

    input:
    tuple val(sample_id), path(reads)

    output:
    tuple val(sample_id), path("${sample_id}_embeddings.parquet"), emit: embeddings

    script:
    """
    python3 -c "
import sys
sys.path.insert(0, '${baseDir}')
from src.models.dnabert import DNABERTEmbedder
import pandas as pd, numpy as np

embedder = DNABERTEmbedder(model_size='dnabert2', device='cuda')
# Read FASTA/FASTQ and generate embeddings
emb = embedder.embed_file('${reads}', batch_size=${params.batch_size})
df = pd.DataFrame(emb)
df.to_parquet('${sample_id}_embeddings.parquet', index=False)
print(f'Generated {len(df)} embeddings for ${sample_id}')
"
    """
}

// ---------- Process: Cluster Embeddings (UMAP + HDBSCAN) ---------------------
process CLUSTER_SEQUENCES {
    tag "${sample_id}"
    label 'process_high_memory'

    container params.container
    publishDir "${params.output_dir}/clusters", mode: 'copy'

    input:
    tuple val(sample_id), path(embeddings)

    output:
    tuple val(sample_id), path("${sample_id}_clusters.parquet"), emit: clusters

    script:
    """
    python3 -c "
import sys
sys.path.insert(0, '${baseDir}')
from src.clustering.algorithms import EmbeddingClusterer
import pandas as pd

df = pd.read_parquet('${embeddings}')
embeddings_arr = df.values

clusterer = EmbeddingClusterer(
    method='${params.cluster_method}',
    min_cluster_size=${params.min_cluster_size}
)
labels = clusterer.fit_predict(embeddings_arr)
df['cluster_label'] = labels
df.to_parquet('${sample_id}_clusters.parquet', index=False)
print(f'Found {len(set(labels)) - 1} clusters in ${sample_id}')
"
    """
}

// ---------- Process: BLAST Taxonomy Assignment -------------------------------
process ASSIGN_TAXONOMY_BLAST {
    tag "${sample_id}"
    label 'process_blast'

    container params.container
    publishDir "${params.output_dir}/taxonomy", mode: 'copy'

    input:
    tuple val(sample_id), path(clusters), path(reads)

    output:
    tuple val(sample_id), path("${sample_id}_taxonomy.parquet"), emit: taxonomy

    script:
    """
    python3 -c "
import sys
sys.path.insert(0, '${baseDir}')
from src.clustering.taxonomy import HybridTaxonomyAssigner
import pandas as pd

df = pd.read_parquet('${clusters}')
assigner = HybridTaxonomyAssigner(
    blast_db='${params.reference_dir}/indices/18S/combined_18S',
    evalue=${params.blast_evalue},
    identity_threshold=${params.blast_identity},
    knn_k=${params.knn_k},
    knn_min_similarity=${params.knn_min_similarity}
)
taxonomy = assigner.assign(df, reads_path='${reads}')
taxonomy.to_parquet('${sample_id}_taxonomy.parquet', index=False)
print(f'Taxonomy assigned for ${sample_id}')
"
    """
}

// ---------- Process: Novelty Detection ---------------------------------------
process DETECT_NOVELTY {
    tag "${sample_id}"
    label 'process_medium'

    container params.container
    publishDir "${params.output_dir}/novelty", mode: 'copy'

    input:
    tuple val(sample_id), path(taxonomy)

    output:
    tuple val(sample_id), path("${sample_id}_novelty.parquet"), emit: novelty
    path "${sample_id}_novelty_report.json",                    emit: reports

    script:
    """
    python3 -c "
import sys
sys.path.insert(0, '${baseDir}')
from src.novelty.detection import NoveltyAnalyzer
import pandas as pd, json

df = pd.read_parquet('${taxonomy}')
analyzer = NoveltyAnalyzer()
result = analyzer.detect(df)
result.to_parquet('${sample_id}_novelty.parquet', index=False)

report = {
    'sample_id': '${sample_id}',
    'novel_sequences': int((result['is_novel'] == True).sum()),
    'total_sequences': len(result)
}
with open('${sample_id}_novelty_report.json', 'w') as f:
    json.dump(report, f, indent=2)
"
    """
}

// ---------- Process: Aggregate & Generate Final Report -----------------------
process GENERATE_REPORT {
    label 'process_low'

    container params.container
    publishDir "${params.output_dir}/reports", mode: 'copy'

    input:
    path taxonomy_files
    path novelty_reports

    output:
    path "biodiversity_report.html", emit: html_report
    path "biodiversity_summary.json", emit: summary

    script:
    """
    python3 -c "
import sys, glob, json
sys.path.insert(0, '${baseDir}')
import pandas as pd

# Aggregate all taxonomy results
dfs = [pd.read_parquet(f) for f in glob.glob('*_taxonomy.parquet')]
combined = pd.concat(dfs, ignore_index=True)

# Aggregate novelty reports
reports = [json.load(open(f)) for f in glob.glob('*_novelty_report.json')]
summary = {
    'total_samples': len(reports),
    'total_sequences': sum(r['total_sequences'] for r in reports),
    'total_novel': sum(r['novel_sequences'] for r in reports),
    'unique_taxa': combined['species'].nunique() if 'species' in combined.columns else 0
}

with open('biodiversity_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Simple HTML report
html = f'<html><body><h1>Avalanche eDNA Report</h1><pre>{json.dumps(summary, indent=2)}</pre></body></html>'
with open('biodiversity_report.html', 'w') as f:
    f.write(html)

print('Report generated:', summary)
"
    """
}

// ---------- Workflow: Main ---------------------------------------------------
workflow {

    // Discover input FASTQ files
    reads_ch = Channel
        .fromFilePairs("${params.input_dir}/*_{1,2}.fastq.gz", flat: true)
        .ifEmpty { error "No FASTQ pairs found in ${params.input_dir}" }

    // Stage 1: Preprocessing
    if (!params.skip_preprocessing) {
        preprocessed = PREPROCESS_READS(reads_ch)
        reads_for_embedding = preprocessed.trimmed_reads
    } else {
        reads_for_embedding = reads_ch
    }

    // Stage 2: Embeddings
    if (!params.skip_embedding) {
        embedded = GENERATE_EMBEDDINGS(reads_for_embedding)
    }

    // Stage 3: Clustering
    if (!params.skip_clustering) {
        clustered = CLUSTER_SEQUENCES(embedded.embeddings)
    }

    // Stage 4: Taxonomy (joins clusters with reads for BLAST)
    if (!params.skip_taxonomy) {
        joined = clustered.clusters.join(reads_for_embedding)
        taxonomy = ASSIGN_TAXONOMY_BLAST(joined)
    }

    // Stage 5: Novelty Detection
    if (!params.skip_novelty) {
        novelty = DETECT_NOVELTY(taxonomy.taxonomy)
    }

    // Stage 6: Report Generation
    GENERATE_REPORT(
        taxonomy.taxonomy.map { sid, path -> path }.collect(),
        novelty.reports.collect()
    )
}

// ---------- Workflow: Training Only ------------------------------------------
workflow TRAIN {
    GENERATE_EMBEDDINGS(
        Channel.fromFilePairs("${params.input_dir}/*_{1,2}.fastq.gz", flat: true)
    )
}
