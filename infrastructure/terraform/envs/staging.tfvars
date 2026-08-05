# =============================================================================
# Staging environment variables for Terraform
# =============================================================================
project_id           = "your-gcp-project-id"
region               = "us-central1"
zone                 = "us-central1-a"
environment          = "staging"
gke_node_count       = 2
gke_machine_type     = "n2-standard-4"
db_tier              = "db-g1-small"
db_password          = "REPLACE_IN_CI_OR_VAULT"

