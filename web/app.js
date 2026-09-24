/**
 * Avalanche eDNA — Jenkins CI/CD Mission Control & GitHub Pages Script
 * Interactive real-time pipeline simulator, stage controller, and telemetry.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const triggerBtn = document.getElementById('triggerBuildBtn');
  const deployGhPagesBtn = document.getElementById('deployGhPagesBtn');
  const clearConsoleBtn = document.getElementById('clearConsoleBtn');
  const copyConsoleBtn = document.getElementById('copyConsoleBtn');
  const autoscrollCheck = document.getElementById('autoscrollCheck');
  const terminalOutput = document.getElementById('terminalOutput');
  const failToggle = document.getElementById('failToggle');
  const modeSelect = document.getElementById('modeSelect');
  const envSelect = document.getElementById('envSelect');
  const pipelineStatusBadge = document.getElementById('pipelineStatusBadge');
  const lastRunTime = document.getElementById('lastRunTime');

  let currentBuildNumber = 42;
  let isRunning = false;

  // Tab switching
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(`pane-${btn.dataset.tab}`);
      if (targetPane) {
        targetPane.classList.add('active');
      }
    });
  });

  // Copy Logs to Clipboard
  copyConsoleBtn.addEventListener('click', () => {
    const text = terminalOutput.innerText;
    navigator.clipboard.writeText(text).then(() => {
      const originalText = copyConsoleBtn.textContent;
      copyConsoleBtn.textContent = '✓ Copied!';
      copyConsoleBtn.style.color = '#10b981';
      setTimeout(() => {
        copyConsoleBtn.textContent = originalText;
        copyConsoleBtn.style.color = '';
      }, 2000);
    }).catch(err => {
      console.error('Failed to copy logs', err);
    });
  });

  // Clear Console
  clearConsoleBtn.addEventListener('click', () => {
    terminalOutput.innerHTML = '<code>[Pipeline] Console cleared by operator.</code>';
  });

  function appendLog(line) {
    const code = terminalOutput.querySelector('code');
    if (code) {
      code.textContent += `\n${line}`;
      if (autoscrollCheck.checked) {
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
      }
    }
  }

  function getTimestamp() {
    const now = new Date();
    return now.toISOString().replace('T', ' ').substring(0, 19);
  }

  // Stages configuration
  const stages = [
    {
      id: 1,
      name: 'Pre-flight & Env',
      baseTime: 1.2,
      runMsg: 'Checking Python 3.13, Docker 29.6, and Git rev 7cab19f...',
      successMsg: 'Pre-flight checks passed successfully. [reports/env-check.json]'
    },
    {
      id: 2,
      name: 'Lint & Quality',
      baseTime: 1.8,
      runMsg: 'Verifying syntax across 72 Python modules in src/ and scripts/...',
      successMsg: 'Linting passed: 0 syntax errors, PEP8 conformant.'
    },
    {
      id: 3,
      name: 'Unit & Tests',
      baseTime: 2.8,
      runMsg: 'Running 10 test suites (Taxonomy, Continual Learning, BLAST, API)...',
      successMsg: 'All 10 tests passed! Test results archived to reports/junit.xml (87.04% coverage).'
    },
    {
      id: 4,
      name: 'Security Audit',
      baseTime: 1.5,
      runMsg: 'Scanning packages and git tree for CVEs and hardcoded secrets...',
      successMsg: 'Security audit passed: 0 Critical, 0 High vulnerabilities detected.'
    },
    {
      id: 5,
      name: 'Package & Build',
      baseTime: 2.2,
      runMsg: 'Building wheel distribution and container buildx manifest...',
      successMsg: 'Artifacts generated: build-artifacts/avalanche-edna-build.zip (2.7 KB)'
    },
    {
      id: 6,
      name: 'Deploy & Health',
      baseTime: 2.5,
      runMsg: 'Initiating zero-downtime rolling deployment and endpoint probes...',
      successMsg: 'Deployment healthy: /api/v1/health (14ms latency) & /api/v1/taxonomy/predict ONLINE.'
    },
    {
      id: 7,
      name: 'GitHub Pages',
      baseTime: 1.4,
      runMsg: 'Syncing UI Mission Control build to branch "gh-pages"...',
      successMsg: 'GitHub Pages live at: https://faisaltabrez.github.io/Avalanche_eDNA/'
    }
  ];

  function resetStages() {
    stages.forEach(s => {
      const card = document.getElementById(`stageCard${s.id}`);
      const badge = document.getElementById(`stageBadge${s.id}`);
      const timeEl = document.getElementById(`stageTime${s.id}`);
      if (card) {
        card.className = 'stage-card is-idle';
      }
      if (badge) {
        badge.textContent = 'QUEUED';
      }
      if (timeEl) {
        timeEl.textContent = '0.0s';
      }
    });
  }

  function runPipeline() {
    if (isRunning) return;
    isRunning = true;
    currentBuildNumber++;

    const mode = modeSelect.value;
    const targetEnv = envSelect.value;
    const simulateFailure = failToggle.checked;

    // Update UI elements
    document.querySelector('.build-tag').textContent = `BUILD #${currentBuildNumber}`;
    pipelineStatusBadge.textContent = 'BUILDING';
    pipelineStatusBadge.className = 'badge badge-primary';
    triggerBtn.disabled = true;
    triggerBtn.innerHTML = '<span class="btn-icon">⏳</span><span>Pipeline Executing...</span>';

    // Switch to console tab
    document.querySelector('[data-tab="terminal"]').click();

    terminalOutput.innerHTML = `<code>[Pipeline] Start of Pipeline (Build #${currentBuildNumber})
[Pipeline] node
Running on Jenkins Local Windows Node (Jenkins Home: C:\\ProgramData\\Jenkins\\.jenkins)
[Pipeline] {
2026-09-24 [INFO] Mode: [${mode.toUpperCase()}], Target Environment: [${targetEnv.toUpperCase()}]</code>`;

    resetStages();

    let stageIndex = 0;

    function executeNextStage() {
      if (stageIndex >= stages.length) {
        // Complete Pipeline
        isRunning = false;
        triggerBtn.disabled = false;
        triggerBtn.innerHTML = '<span class="btn-icon">▶</span><span>Trigger Jenkins Pipeline</span>';
        pipelineStatusBadge.textContent = 'SUCCESS';
        pipelineStatusBadge.className = 'badge badge-success';
        lastRunTime.textContent = 'Just now';

        appendLog(`[Pipeline] archiveArtifacts
Archiving reports/* and build-artifacts/*
[Pipeline] End of Pipeline
Finished: SUCCESS (All 7 stages passed in ${(stages.reduce((a, b) => a + b.baseTime, 0)).toFixed(1)}s)`);
        return;
      }

      const s = stages[stageIndex];
      const card = document.getElementById(`stageCard${s.id}`);
      const badge = document.getElementById(`stageBadge${s.id}`);
      const timeEl = document.getElementById(`stageTime${s.id}`);

      // Start stage
      card.className = 'stage-card is-running';
      badge.textContent = 'RUNNING';
      appendLog(`[Pipeline] stage: ${s.name}\n${getTimestamp()} [INFO] [${s.name.toUpperCase()}] ${s.runMsg}`);

      // Simulate failure at Stage 6 if toggle is active
      const willFail = simulateFailure && s.id === 6;

      const durationMs = s.baseTime * 1000;
      const startTime = performance.now();

      const timerInterval = setInterval(() => {
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        if (timeEl) timeEl.textContent = `${elapsed}s`;
      }, 100);

      setTimeout(() => {
        clearInterval(timerInterval);
        const finalTime = ((performance.now() - startTime) / 1000).toFixed(1);
        if (timeEl) timeEl.textContent = `${finalTime}s`;

        if (willFail) {
          card.className = 'stage-card is-failed';
          badge.textContent = 'FAILED';
          pipelineStatusBadge.textContent = 'FAILED';
          pipelineStatusBadge.className = 'badge badge-primary';
          pipelineStatusBadge.style.background = 'rgba(244, 63, 94, 0.2)';
          pipelineStatusBadge.style.color = '#f43f5e';

          appendLog(`${getTimestamp()} [ERROR] [${s.name.toUpperCase()}] Health probe failed on mock cluster endpoint!`);
          appendLog(`${getTimestamp()} [WARN]  [DEPLOY] Initiating automated rollback to previous healthy revision...`);
          appendLog(`${getTimestamp()} [SUCCESS] [DEPLOY] Automated rollback complete. Previous revision active.`);
          appendLog(`[Pipeline] End of Pipeline\nFinished: FAILURE (Pipeline failed at Stage 6)`);

          isRunning = false;
          triggerBtn.disabled = false;
          triggerBtn.innerHTML = '<span class="btn-icon">▶</span><span>Trigger Jenkins Pipeline</span>';
          return;
        }

        // Stage success
        card.className = 'stage-card';
        badge.textContent = 'SUCCESS';
        appendLog(`${getTimestamp()} [SUCCESS] [${s.name.toUpperCase()}] ${s.successMsg}`);

        stageIndex++;
        executeNextStage();
      }, durationMs);
    }

    executeNextStage();
  }

  // Trigger build event
  triggerBtn.addEventListener('click', runPipeline);

  // Deploy to GitHub Pages button
  deployGhPagesBtn.addEventListener('click', () => {
    deployGhPagesBtn.disabled = true;
    deployGhPagesBtn.innerHTML = '<span class="btn-icon">⏳</span><span>Syncing gh-pages...</span>';

    setTimeout(() => {
      deployGhPagesBtn.disabled = false;
      deployGhPagesBtn.innerHTML = '<span class="btn-icon">✓</span><span>Synced to GitHub Pages</span>';
      appendLog(`\n[GitHub Pages] Direct sync requested by operator.`);
      appendLog(`[GitHub Pages] Remote branch origin/gh-pages updated successfully.`);
      appendLog(`[GitHub Pages] Live Site: https://faisaltabrez.github.io/Avalanche_eDNA/`);

      setTimeout(() => {
        deployGhPagesBtn.innerHTML = '<span class="btn-icon">🚀</span><span>Sync to GitHub Pages</span>';
      }, 3000);
    }, 1500);
  });

  // Stage card click inspection
  document.querySelectorAll('.stage-card').forEach(card => {
    card.addEventListener('click', () => {
      const stageNum = card.dataset.stage;
      const stage = stages.find(s => s.id == stageNum);
      if (!stage) return;

      if (stage.id === 3) {
        document.querySelector('[data-tab="tests"]').click();
      } else if (stage.id === 5) {
        document.querySelector('[data-tab="artifacts"]').click();
      } else if (stage.id === 7) {
        document.querySelector('[data-tab="jenkins-arch"]').click();
      } else {
        document.querySelector('[data-tab="terminal"]').click();
      }
    });
  });
});
