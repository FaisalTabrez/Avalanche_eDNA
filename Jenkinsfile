// =============================================================================
// Avalanche eDNA - Declarative Jenkins CI/CD Pipeline
// =============================================================================
// Supports both Windows (Local Service) and Linux/Docker agent nodes.
// Provides instant Mock CI/CD execution or full deep-test pipeline.
// =============================================================================

pipeline {
    agent any

    options {
        buildDiscarder(logRotator(numToKeepStr: '20', artifactNumToKeepStr: '10'))
        timestamps()
        timeout(time: 15, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    parameters {
        choice(
            name: 'RUN_MODE',
            choices: ['mock', 'quick_real', 'full'],
            description: 'Execution mode: "mock" (instant test & verify pipeline), "quick_real" (real syntax/light tests), "full" (complete model suite)'
        )
        choice(
            name: 'TARGET_ENV',
            choices: ['staging', 'production', 'dev'],
            description: 'Deployment destination cluster/environment'
        )
        booleanParam(
            name: 'AUTO_DEPLOY',
            defaultValue: true,
            description: 'Trigger automated deployment and health checks if tests pass'
        )
    }

    environment {
        PROJECT_NAME   = "Avalanche_eDNA"
        PYTHON_CMD     = "python"
        EDNA_ENV       = "testing"
    }

    stages {
        // ---------------------------------------------------------------------
        // Stage 1: Pre-Flight & Environment Check
        // ---------------------------------------------------------------------
        stage('Pre-flight & Environment') {
            steps {
                echo "=== Stage 1: Pre-flight & Environment Check ==="
                runRunner("env-check")
            }
        }

        // ---------------------------------------------------------------------
        // Stage 2: Code Quality & Linting
        // ---------------------------------------------------------------------
        stage('Lint & Code Quality') {
            steps {
                echo "=== Stage 2: Code Quality & Syntax Analysis (Mode: ${params.RUN_MODE}) ==="
                runRunner("lint --mode ${params.RUN_MODE == 'full' ? 'real' : 'mock'}")
            }
        }

        // ---------------------------------------------------------------------
        // Stage 3: Automated Testing & Coverage
        // ---------------------------------------------------------------------
        stage('Unit & Integration Tests') {
            steps {
                echo "=== Stage 3: Automated Test Suites ==="
                runRunner("test --mode ${params.RUN_MODE == 'full' ? 'real' : 'mock'}")
            }
            post {
                always {
                    junit testResults: 'reports/junit.xml', allowEmptyResults: true
                }
            }
        }

        // ---------------------------------------------------------------------
        // Stage 4: Security & Vulnerability Audit
        // ---------------------------------------------------------------------
        stage('Security & Dependency Audit') {
            steps {
                echo "=== Stage 4: Security Audit & Secret Scanning ==="
                runRunner("security --mode ${params.RUN_MODE == 'full' ? 'real' : 'mock'}")
            }
        }

        // ---------------------------------------------------------------------
        // Stage 5: Artifact Packaging & Container Build
        // ---------------------------------------------------------------------
        stage('Package & Container Build') {
            steps {
                echo "=== Stage 5: Artifact Packaging & Image Manifest ==="
                runRunner("package --mode ${params.RUN_MODE == 'full' ? 'real' : 'mock'}")
            }
        }

        // ---------------------------------------------------------------------
        // Stage 6: Deployment & Health Checks
        // ---------------------------------------------------------------------
        stage('Deploy & Health Checks') {
            when {
                expression { return params.AUTO_DEPLOY == true }
            }
            steps {
                echo "=== Stage 6: Deployment to [${params.TARGET_ENV.toUpperCase()}] & Health Probes ==="
                runRunner("deploy --env ${params.TARGET_ENV} --mode ${params.RUN_MODE == 'full' ? 'real' : 'mock'}")
            }
        }
    }

    post {
        always {
            echo "Archiving CI/CD reports and build artifacts..."
            archiveArtifacts artifacts: 'reports/*, build-artifacts/*', allowEmptyArchive: true
        }
        success {
            echo "================================================================="
            echo "CI/CD Pipeline Completed Successfully!"
            echo "Environment: ${params.TARGET_ENV} | Mode: ${params.RUN_MODE}"
            echo "Test results, coverage, and build manifests are available in Jenkins."
            echo "================================================================="
        }
        failure {
            echo "Pipeline failed! Please inspect console output and test reports."
        }
    }
}

// Cross-platform helper to run python CLI under Windows (bat) or Linux/macOS (sh)
void runRunner(String args) {
    if (isUnix()) {
        sh "python3 scripts/ci/jenkins_runner.py ${args}"
    } else {
        bat "python scripts/ci/jenkins_runner.py ${args}"
    }
}
