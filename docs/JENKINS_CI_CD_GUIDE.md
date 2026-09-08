# Avalanche eDNA - Jenkins CI/CD Implementation Guide

This guide documents the Mock CI/CD pipeline implementation for the Avalanche eDNA project using your local Jenkins instance running at **`http://localhost:8081`**.

---

## 1. Overview of the Pipeline

The pipeline is implemented as a **Declarative Jenkins Pipeline** ([Jenkinsfile](file:///c:/Volume%20D/Avalanche_eDNA/Jenkinsfile)) orchestrated by an OS-agnostic execution engine ([scripts/ci/jenkins_runner.py](file:///c:/Volume%20D/Avalanche_eDNA/scripts/ci/jenkins_runner.py)).

### Key Advantages:
- **Instant Execution (Mock Mode)**: Runs the entire 6-stage lifecycle in ~10 seconds, perfect for validating CI/CD hooks, Jenkins plugins, and visual reporting without needing heavy 30-minute GPU dependencies.
- **Rich Jenkins Integration**: Generates standard **JUnit XML** test results and **Cobertura Coverage XML**, automatically populating Jenkins' **Test Result Trend** graph and stage duration timelines.
- **Multi-Environment Ready**: Supports automated zero-downtime deployment simulation and health probes to `staging`, `production`, and `dev` environments.
- **Cross-Platform**: Seamlessly runs on Windows Jenkins agents (`bat`) as well as Linux/Docker agent nodes (`sh`).

---

## 2. Pipeline Stages

| Stage | Name | Action & Output |
|---|---|---|
| **Stage 1** | **Pre-flight & Environment** | Checks Python version, Git commit/branch, and Docker daemon. Outputs `reports/env-check.json`. |
| **Stage 2** | **Lint & Code Quality** | Validates syntax of all Python source modules (with flake8 fallback). Outputs `reports/lint-results.txt`. |
| **Stage 3** | **Unit & Integration Tests** | Executes 10 test suites across taxonomy, continual learning, model memory, and system endpoints. Outputs standard JUnit XML (`reports/junit.xml`) and Cobertura XML (`reports/coverage.xml`). |
| **Stage 4** | **Security & Dependency Audit** | Performs automated vulnerability and secret scanning. Outputs `reports/security-audit.json`. |
| **Stage 5** | **Package & Container Build** | Creates deployment bundle and Docker container build manifest. Outputs to `build-artifacts/`. |
| **Stage 6** | **Deploy & Health Checks** | Simulates rolling zero-downtime deployment to the target environment with automated HTTP/API health probes. Outputs `reports/deployment-manifest.json`. |

---

## 3. How to Set Up the Job in Jenkins (`http://localhost:8081`)

We have already placed the job configuration directly into your Jenkins jobs directory (`C:\ProgramData\Jenkins\.jenkins\jobs\Avalanche-eDNA-Pipeline\config.xml`).

### Option A: Reload from Disk (Fastest)
1. Open your browser and navigate to: **`http://localhost:8081`**
2. In the left navigation menu, click **Manage Jenkins**.
3. Under **System Administration**, click **Reload Configuration from Disk**.
4. Return to the Jenkins Dashboard. You will see **`Avalanche-eDNA-Pipeline`** ready to run!

---

### Option B: Create Job via Jenkins UI
If you prefer creating the job via the UI:
1. Go to **Dashboard** -> **New Item**.
2. Enter the Item Name: **`Avalanche-eDNA-Pipeline`**.
3. Select **Pipeline** and click **OK**.
4. Scroll down to the **Pipeline** section:
   - **Definition**: Select **Pipeline script from SCM**.
   - **SCM**: Select **Git**.
   - **Repository URL**: 
     - Local repository path: `C:\Volume D\Avalanche_eDNA` *(recommended for instant local builds)*
     - Or GitHub URL: `https://github.com/FaisalTabrez/Avalanche_eDNA.git`
   - **Branch Specifier**: `*/main`
   - **Script Path**: `Jenkinsfile`
5. Click **Save**.

---

## 4. Running the Pipeline

1. In the job page, click **Build with Parameters** (or **Build Now** on first run to register the parameters).
2. Configure parameters:
   - **`RUN_MODE`**:
     - `mock`: Runs fast simulation in ~10 seconds with full JUnit/coverage reports.
     - `quick_real`: Checks real syntax on all local Python source files.
     - `full`: Complete suite.
   - **`TARGET_ENV`**: Select `staging`, `production`, or `dev`.
   - **`AUTO_DEPLOY`**: Check/uncheck to enable/disable the deployment stage.
3. Click **Build**.

---

## 5. Reviewing Results in Jenkins

Once the build completes:
- **Stage View**: See each of the 6 stages turn green with individual execution times.
- **Test Result Trend**: Click **Test Result** in the build page to view the 10 passing tests broken down by package (`tests.test_enhanced_taxonomy`, `tests.test_continual_learning`, etc.).
- **Build Artifacts**: Download generated test reports and packages:
  - `reports/junit.xml`
  - `reports/coverage.xml`
  - `reports/security-audit.json`
  - `reports/deployment-manifest.json`
  - `build-artifacts/avalanche-edna-build_*.zip`
  - `build-artifacts/avalanche-edna-build_*-manifest.json`

---

## 6. Command-Line Verification (Without Jenkins UI)

You can also run any pipeline stage directly from your terminal using the Python runner:

```powershell
# Environment pre-flight check
python scripts/ci/jenkins_runner.py env-check

# Lint & code quality
python scripts/ci/jenkins_runner.py lint --mode mock

# Unit & integration tests (generates reports/junit.xml)
python scripts/ci/jenkins_runner.py test --mode mock

# Security vulnerability audit
python scripts/ci/jenkins_runner.py security --mode mock

# Artifact packaging
python scripts/ci/jenkins_runner.py package --mode mock

# Simulated deployment to staging
python scripts/ci/jenkins_runner.py deploy --env staging --mode mock
```
