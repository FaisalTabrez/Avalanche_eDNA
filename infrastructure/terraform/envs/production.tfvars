# =============================================================================
# Production environment variables for Terraform
# =============================================================================
project_id           = "your-gcp-project-id"
region               = "us-central1"
zone                 = "us-central1-a"
environment          = "production"
gke_node_count       = 3
gke_machine_type     = "n2-standard-4"
db_tier              = "db-perf-optimized-N-2"
db_password          = "REPLACE_WITH_SECRET_MANAGER_REF"

