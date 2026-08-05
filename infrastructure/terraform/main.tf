// =============================================================================
// Avalanche eDNA - Terraform Infrastructure
// Provisions: GKE cluster, GCS bucket, Artifact Registry, Cloud SQL (PostgreSQL)
// =============================================================================
// Usage:
//   terraform init
//   terraform workspace new staging   # or production
//   terraform plan -var-file=envs/staging.tfvars
//   terraform apply -var-file=envs/staging.tfvars
// =============================================================================

terraform {
  required_version = ">= 1.7.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.20"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.20"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.27"
    }
  }

  # Remote state in GCS — replace with your bucket
  backend "gcs" {
    bucket = "YOUR_TERRAFORM_STATE_BUCKET"
    prefix = "avalanche-edna/terraform"
  }
}

# =============================================================================
# Variables
# =============================================================================
variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Deployment environment: staging | production"
  type        = string
  default     = "staging"
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "environment must be 'staging' or 'production'."
  }
}

variable "gke_node_count" {
  description = "Number of GKE worker nodes per zone (CPU pool)"
  type        = number
  default     = 3
}

variable "gke_machine_type" {
  description = "Machine type for GKE CPU node pool"
  type        = string
  default     = "n2-standard-4"
}


variable "db_tier" {
  description = "Cloud SQL instance tier"
  type        = string
  default     = "db-g1-small"
}

variable "db_password" {
  description = "PostgreSQL password for edna_user"
  type        = string
  sensitive   = true
}

# =============================================================================
# Providers
# =============================================================================
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# =============================================================================
# Locals (naming convention)
# =============================================================================
locals {
  name_prefix = "edna-${var.environment}"
  labels = {
    project     = "avalanche-edna"
    environment = var.environment
    managed_by  = "terraform"
  }
}

# =============================================================================
# Enable Required APIs
# =============================================================================
resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",         # GKE
    "artifactregistry.googleapis.com",  # Container Registry
    "sqladmin.googleapis.com",          # Cloud SQL
    "storage.googleapis.com",           # GCS
    "batch.googleapis.com",             # Cloud Batch (Nextflow)
    "compute.googleapis.com",           # Compute Engine
    "iam.googleapis.com",               # IAM
    "secretmanager.googleapis.com",     # Secret Manager
    "logging.googleapis.com",           # Cloud Logging
    "monitoring.googleapis.com",        # Cloud Monitoring
  ])

  service            = each.key
  disable_on_destroy = false
}

# =============================================================================
# VPC Network
# =============================================================================
resource "google_compute_network" "edna_vpc" {
  name                    = "${local.name_prefix}-vpc"
  auto_create_subnetworks = false
  depends_on              = [google_project_service.apis]
}

resource "google_compute_subnetwork" "edna_subnet" {
  name          = "${local.name_prefix}-subnet"
  ip_cidr_range = "10.10.0.0/20"
  region        = var.region
  network       = google_compute_network.edna_vpc.id

  # Enable Private Google Access for GKE nodes
  private_ip_google_access = true

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.20.0.0/16"
  }
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.30.0.0/20"
  }
}

# =============================================================================
# GCS Bucket (eDNA datasets + model artifacts)
# =============================================================================
resource "google_storage_bucket" "edna_data" {
  name          = "${local.name_prefix}-data-${var.project_id}"
  location      = var.region
  force_destroy = var.environment == "staging"

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    action { type = "SetStorageClass"; storage_class = "NEARLINE" }
    condition { age = 30; matches_prefix = ["datasets/raw/"] }
  }

  lifecycle_rule {
    action { type = "SetStorageClass"; storage_class = "COLDLINE" }
    condition { age = 90; matches_prefix = ["datasets/archived/"] }
  }

  labels = local.labels
}

resource "google_storage_bucket_iam_member" "edna_pipeline_sa_storage" {
  bucket = google_storage_bucket.edna_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.edna_pipeline.email}"
}

# =============================================================================
# Google Artifact Registry (Docker images)
# =============================================================================
resource "google_artifact_registry_repository" "edna_registry" {
  provider      = google-beta
  location      = var.region
  repository_id = "edna-pipeline"
  description   = "Avalanche eDNA container images"
  format        = "DOCKER"

  labels = local.labels
}

# =============================================================================
# Service Account for the pipeline
# =============================================================================
resource "google_service_account" "edna_pipeline" {
  account_id   = "${local.name_prefix}-sa"
  display_name = "Avalanche eDNA Pipeline Service Account"
}

resource "google_project_iam_member" "edna_sa_roles" {
  for_each = toset([
    "roles/artifactregistry.reader",
    "roles/storage.objectAdmin",
    "roles/batch.jobsEditor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/secretmanager.secretAccessor",
    "roles/cloudsql.client",
  ])

  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.edna_pipeline.email}"
}

# =============================================================================
# GKE Cluster (Autopilot for simplicity, or Standard for GPU control)
# =============================================================================
resource "google_container_cluster" "edna_gke" {
  provider = google-beta
  name     = "${local.name_prefix}-gke"
  location = var.region   # Regional cluster for HA

  # Use Autopilot mode for cost-efficient scaling
  enable_autopilot = var.environment == "staging"

  # Standard mode config (used in production for GPU pools)
  dynamic "node_config" {
    for_each = var.environment == "production" ? [1] : []
    content {
      machine_type    = var.gke_machine_type
      service_account = google_service_account.edna_pipeline.email
      oauth_scopes    = ["https://www.googleapis.com/auth/cloud-platform"]
      labels          = local.labels
      tags            = ["edna-node"]
    }
  }

  network    = google_compute_network.edna_vpc.id
  subnetwork = google_compute_subnetwork.edna_subnet.id

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Enable GKE Dataplane V2 for network observability
  datapath_provider = "ADVANCED_DATAPATH"

  release_channel {
    channel = "REGULAR"
  }

  depends_on = [google_project_service.apis]
}


# =============================================================================
# Cloud SQL (PostgreSQL 15)
# =============================================================================
resource "google_sql_database_instance" "edna_postgres" {
  name             = "${local.name_prefix}-postgres"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = var.db_tier
    availability_type = var.environment == "production" ? "REGIONAL" : "ZONAL"

    disk_type       = "PD_SSD"
    disk_autoresize = true
    disk_size       = 50

    backup_configuration {
      enabled                        = true
      start_time                     = "02:00"
      point_in_time_recovery_enabled = var.environment == "production"
      backup_retention_settings {
        retained_backups = 14
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.edna_vpc.id
    }

    database_flags {
      name  = "max_connections"
      value = "200"
    }
  }

  deletion_protection = var.environment == "production"
  depends_on          = [google_project_service.apis]
}

resource "google_sql_database" "edna_db" {
  name     = "edna_reports"
  instance = google_sql_database_instance.edna_postgres.name
}

resource "google_sql_user" "edna_user" {
  name     = "edna_user"
  instance = google_sql_database_instance.edna_postgres.name
  password = var.db_password
}

# =============================================================================
# Secret Manager (store DB password and HF token securely)
# =============================================================================
resource "google_secret_manager_secret" "db_password" {
  secret_id = "${local.name_prefix}-db-password"
  replication { auto {} }
}

resource "google_secret_manager_secret_version" "db_password_v1" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

# =============================================================================
# Outputs
# =============================================================================
output "gke_cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.edna_gke.name
}

output "gcs_bucket_name" {
  description = "GCS data bucket name"
  value       = google_storage_bucket.edna_data.name
}

output "artifact_registry_url" {
  description = "Artifact Registry URL for Docker images"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/edna-pipeline"
}

output "db_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.edna_postgres.connection_name
  sensitive   = true
}

output "pipeline_sa_email" {
  description = "Service account email for the pipeline"
  value       = google_service_account.edna_pipeline.email
}
