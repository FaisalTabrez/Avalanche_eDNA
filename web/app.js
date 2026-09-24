/**
 * Avalanche Mission Control - Streamlit UI Client Logic
 * Interactive state management, page router, analysis wizard, charts & data tables.
 */

// Sample real sequence predictions from Alpine eDNA dataset
const sampleSequences = [
  { id: "seq_0142", rank: "species", label: "Salmo trutta", conf: 0.985, blast: "99.2%", phylum: "Chordata", species: "Salmo trutta (Brown Trout)" },
  { id: "seq_0143", rank: "species", label: "Oncorhynchus mykiss", conf: 0.974, blast: "98.8%", phylum: "Chordata", species: "Oncorhynchus mykiss (Rainbow Trout)" },
  { id: "seq_0144", rank: "species", label: "Gammarus pulex", conf: 0.962, blast: "98.1%", phylum: "Arthropoda", species: "Gammarus pulex (Freshwater Shrimp)" },
  { id: "seq_0145", rank: "species", label: "Baetis rhodani", conf: 0.958, blast: "97.6%", phylum: "Arthropoda", species: "Baetis rhodani (Mayfly)" },
  { id: "seq_0146", rank: "species", label: "Cottus gobio", conf: 0.971, blast: "99.0%", phylum: "Chordata", species: "Cottus gobio (European Bullhead)" },
  { id: "seq_0147", rank: "species", label: "Pseudomonas fluorescens", conf: 0.942, blast: "96.4%", phylum: "Proteobacteria", species: "Pseudomonas fluorescens" },
  { id: "seq_0148", rank: "species", label: "Chironomus riparius", conf: 0.935, blast: "97.2%", phylum: "Arthropoda", species: "Chironomus riparius (Midge)" },
  { id: "seq_0149", rank: "species", label: "Anabaena flos-aquae", conf: 0.920, blast: "95.8%", phylum: "Cyanobacteria", species: "Anabaena flos-aquae" },
  { id: "seq_0150", rank: "species", label: "Chlamydomonas nivalis", conf: 0.966, blast: "98.5%", phylum: "Chlorophyta", species: "Chlamydomonas nivalis (Snow Algae)" },
  { id: "seq_0151", rank: "species", label: "Tubifex tubifex", conf: 0.890, blast: "94.2%", phylum: "Annelida", species: "Tubifex tubifex (Sludge Worm)" },
  { id: "seq_0152", rank: "species", label: "Salvelinus fontinalis", conf: 0.978, blast: "99.4%", phylum: "Chordata", species: "Salvelinus fontinalis (Brook Trout)" },
  { id: "seq_0153", rank: "species", label: "Daphnia pulex", conf: 0.912, blast: "95.0%", phylum: "Arthropoda", species: "Daphnia pulex (Water Flea)" },
  { id: "seq_0154", rank: "species", label: "Rana temporaria", conf: 0.950, blast: "98.2%", phylum: "Chordata", species: "Rana temporaria (Common Frog)" },
  { id: "seq_0155", rank: "species", label: "Rhithrogena semicolorata", conf: 0.945, blast: "97.4%", phylum: "Arthropoda", species: "Rhithrogena semicolorata (Heptageniid)" },
  { id: "seq_0156", rank: "species", label: "Diatoma vulgaris", conf: 0.880, blast: "93.8%", phylum: "Bacillariophyta", species: "Diatoma vulgaris (Diatom)" }
];

const topSpeciesCounts = [
  { name: "Salmo trutta (Brown Trout)", count: 482, pct: 100 },
  { name: "Gammarus pulex (Shrimp)", count: 395, pct: 82 },
  { name: "Baetis rhodani (Mayfly)", count: 310, pct: 64 },
  { name: "Oncorhynchus mykiss", count: 240, pct: 50 },
  { name: "Chironomus riparius", count: 185, pct: 38 },
  { name: "Cottus gobio (Bullhead)", count: 142, pct: 29 },
  { name: "Pseudomonas fluorescens", count: 98, pct: 20 },
  { name: "Chlamydomonas nivalis", count: 76, pct: 16 },
  { name: "Anabaena flos-aquae", count: 54, pct: 11 },
  { name: "Rana temporaria (Frog)", count: 32, pct: 7 }
];

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initDataExplorer();
  initSystemMonitorInterval();
});

// Navigation Handling
function initNavigation() {
  const navButtons = document.querySelectorAll('.nav-item');
  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const pageKey = btn.dataset.page;
      navigateToPage(pageKey);
    });
  });

  const btnRefresh = document.getElementById('btnRefreshDashboard');
  if (btnRefresh) {
    btnRefresh.addEventListener('click', () => {
      btnRefresh.innerHTML = '<span>⏳ Refreshing...</span>';
      setTimeout(() => {
        btnRefresh.innerHTML = '<span>🔄 Refresh Status</span>';
        randomizeMetrics();
      }, 500);
    });
  }
}

function navigateToPage(pageKey) {
  // Update sidebar active button
  document.querySelectorAll('.nav-item').forEach(b => {
    b.classList.toggle('active', b.dataset.page === pageKey);
  });

  // Switch view
  document.querySelectorAll('.page-view').forEach(view => {
    view.classList.toggle('active', view.id === `page-${pageKey}`);
  });

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Wizard Step Navigation
function goToWizardStep(stepNum) {
  // Update wizard progress indicator
  for (let i = 1; i <= 4; i++) {
    const ind = document.getElementById(`wizardStepInd${i}`);
    const pane = document.getElementById(`wizardPane${i}`);
    if (ind) {
      ind.classList.remove('active', 'passed');
      if (i === stepNum) ind.classList.add('active');
      else if (i < stepNum) ind.classList.add('passed');
    }
    if (pane) {
      pane.classList.toggle('active', i === stepNum);
    }
  }

  // Update Review Summary when entering step 3
  if (stepNum === 3) {
    const db = document.getElementById('dbSelect').options[document.getElementById('dbSelect').selectedIndex].text;
    const ident = document.getElementById('identityVal').textContent;
    const evalue = document.getElementById('evalueSelect').value;
    const threads = document.getElementById('threadsVal').textContent + " Workers";
    const sra = document.getElementById('sraAccessionInput').value || "Uploaded Files";

    document.getElementById('revSource').textContent = `SRA: ${sra}`;
    document.getElementById('revDb').textContent = db;
    document.getElementById('revIdentity').textContent = ident;
    document.getElementById('revEvalue').textContent = evalue;
    document.getElementById('revThreads').textContent = threads;
  }
}

function toggleInputMethod(val) {
  document.getElementById('sraInputSection').style.display = val === 'sra' ? 'block' : 'none';
  document.getElementById('fileUploadSection').style.display = val === 'upload' ? 'block' : 'none';
}

function prefillDemoSRA() {
  document.getElementById('sraAccessionInput').value = 'SRR35551197';
}

// Analysis Execution Simulator
function startAnalysisExecution() {
  goToWizardStep(4);

  const progressBar = document.getElementById('execProgress');
  const stepLabel = document.getElementById('execStepLabel');
  const percentage = document.getElementById('execPercentage');
  const logBox = document.getElementById('execLogBox');
  const resultsPreview = document.getElementById('execResultsPreview');

  progressBar.style.width = '0%';
  percentage.textContent = '0%';
  resultsPreview.style.display = 'none';

  logBox.innerHTML = `<code>[0.0s] Initializing Avalanche Taxonomic Engine v2...
[0.5s] Allocated 4 CPU worker threads. Connecting to Silva 138 reference database.</code>`;

  const steps = [
    { pct: 15, label: "Ingesting fastq reads from accession SRR35551197...", log: "[1.2s] Ingested 2,000 paired-end reads. Quality Q30 score: 94.2%." },
    { pct: 35, label: "Running k-mer tokenization & Snappy vector compression...", log: "[2.8s] Tokenized 2,000 reads into 6-mers. Generated FAISS HNSW queries." },
    { pct: 60, label: "Executing NCBI BLAST+ & KNN taxonomic search...", log: "[5.4s] BLAST alignments matched. KNN top-5 neighbor voting active." },
    { pct: 85, label: "Applying Bayesian confidence calibration & EWC refinement...", log: "[8.1s] Calibrated confidence scores. Resolved 14 conflict cases." },
    { pct: 100, label: "Taxonomic assignments completed!", log: "[10.5s] Pipeline finished: 2,000 reads assigned to 148 species. Results indexed." }
  ];

  let current = 0;

  function runNext() {
    if (current >= steps.length) {
      resultsPreview.style.display = 'block';
      return;
    }

    const s = steps[current];
    progressBar.style.width = `${s.pct}%`;
    percentage.textContent = `${s.pct}%`;
    stepLabel.textContent = s.label;

    const code = logBox.querySelector('code');
    if (code) {
      code.textContent += `\n${s.log}`;
      logBox.scrollTop = logBox.scrollHeight;
    }

    current++;
    setTimeout(runNext, 1800);
  }

  setTimeout(runNext, 500);
}

function downloadMockResults() {
  const csvContent = "data:text/csv;charset=utf-8," + 
    "sequence_id,assigned_rank,assigned_label,confidence,blast_identity,phylum,species\n" +
    sampleSequences.map(e => `${e.id},${e.rank},${e.label},${e.conf},${e.blast},${e.phylum},"${e.species}"`).join("\n");
  
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "avalanche_edna_taxonomy_results.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// Data Explorer Logic
function initDataExplorer() {
  renderSpeciesBarChart();
  renderSequenceTable(sampleSequences);
}

function renderSpeciesBarChart() {
  const container = document.getElementById('speciesBarChart');
  if (!container) return;

  container.innerHTML = topSpeciesCounts.map(item => `
    <div class="species-bar-row">
      <div class="species-name">${item.name}</div>
      <div class="species-bar-track">
        <div class="species-bar-fill" style="width: ${item.pct}%;"></div>
      </div>
      <div class="species-count">${item.count}</div>
    </div>
  `).join('');
}

function renderSequenceTable(items) {
  const tbody = document.getElementById('seqTableBody');
  if (!tbody) return;

  tbody.innerHTML = items.map(seq => `
    <tr>
      <td><code>${seq.id}</code></td>
      <td><span class="badge badge-cyan">${seq.rank}</span></td>
      <td><strong>${seq.label}</strong></td>
      <td><span class="text-green">${(seq.conf * 100).toFixed(1)}%</span></td>
      <td>${seq.blast}</td>
      <td>${seq.phylum}</td>
      <td><em>${seq.species}</em></td>
    </tr>
  `).join('');
}

function filterSequenceTable(query) {
  const q = query.toLowerCase();
  const filtered = sampleSequences.filter(s => 
    s.id.toLowerCase().includes(q) ||
    s.label.toLowerCase().includes(q) ||
    s.phylum.toLowerCase().includes(q) ||
    s.species.toLowerCase().includes(q)
  );
  renderSequenceTable(filtered);
}

function updateExplorerFilters(minConf) {
  document.getElementById('minConfLabel').textContent = minConf;
  const filtered = sampleSequences.filter(s => s.conf >= parseFloat(minConf));
  document.getElementById('expTotalSeq').textContent = filtered.length * 133;
  document.getElementById('expUniqueSpecies').textContent = filtered.length;
  renderSequenceTable(filtered);
}

function switchDataSubtab(tabName) {
  document.querySelectorAll('[data-subtab]').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.subtab === tabName);
  });
  document.getElementById('subtab-abundance').classList.toggle('active', tabName === 'abundance');
  document.getElementById('subtab-sunburst').classList.toggle('active', tabName === 'sunburst');
  document.getElementById('subtab-rawdata').classList.toggle('active', tabName === 'rawdata');
}

// Model Forge Logic
function switchForgeSubtab(tabName) {
  document.querySelectorAll('#page-model-forge .stTabBtn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.subtab === tabName);
  });
  document.getElementById('subtab-training-arena').classList.toggle('active', tabName === 'training-arena');
  document.getElementById('subtab-dynamic-scaling').classList.toggle('active', tabName === 'dynamic-scaling');
}

function simulateTraining() {
  const btn = document.getElementById('btnStartTraining');
  const log = document.getElementById('trainingLog');
  const lossVal = document.getElementById('trainLossVal');
  const accVal = document.getElementById('trainAccVal');

  btn.disabled = true;
  btn.innerHTML = '<span>⏳ Training in Progress...</span>';

  let epoch = 1;
  const maxEpochs = parseInt(document.getElementById('epochSlider').value) || 5;

  log.innerHTML = `<code>[Initialization] Loading DNABERT-2 CPU weights...
[Optimizer] AdamW (lr=2e-5, weight_decay=0.01) initialized.</code>`;

  const interval = setInterval(() => {
    if (epoch > maxEpochs) {
      clearInterval(interval);
      btn.disabled = false;
      btn.innerHTML = '<span>🏋️ Start Fine-Tuning Run</span>';
      const code = log.querySelector('code');
      if (code) {
        code.textContent += `\n[Complete] Checkpoint saved: models/dnabert2_cpu/epoch_${maxEpochs}.pt`;
        log.scrollTop = log.scrollHeight;
      }
      return;
    }

    const loss = (2.1 / epoch + Math.random() * 0.05).toFixed(3);
    const acc = (82 + epoch * 2.5 + Math.random() * 0.5).toFixed(1);

    lossVal.textContent = loss;
    accVal.textContent = `${acc}%`;

    const code = log.querySelector('code');
    if (code) {
      code.textContent += `\n[Epoch ${epoch}/${maxEpochs}] Loss: ${loss} - Accuracy: ${acc}% (elapsed: 1.4s)`;
      log.scrollTop = log.scrollHeight;
    }

    epoch++;
  }, 1200);
}

// System Monitor Refresh & Animation
function refreshSystemMonitor() {
  randomizeMetrics();
}

function randomizeMetrics() {
  const cpu = Math.floor(18 + Math.random() * 15);
  const mem = Math.floor(38 + Math.random() * 8);

  const gaugeCpu = document.getElementById('gaugeCpu');
  const gaugeCpuBar = document.getElementById('gaugeCpuBar');
  const gaugeMem = document.getElementById('gaugeMem');
  const gaugeMemBar = document.getElementById('gaugeMemBar');
  const sidebarCpu = document.getElementById('sidebarCpu');
  const sidebarMem = document.getElementById('sidebarMem');

  if (gaugeCpu) gaugeCpu.textContent = `${cpu}%`;
  if (gaugeCpuBar) gaugeCpuBar.style.width = `${cpu}%`;
  if (gaugeMem) gaugeMem.textContent = `${mem}%`;
  if (gaugeMemBar) gaugeMemBar.style.width = `${mem}%`;
  if (sidebarCpu) sidebarCpu.textContent = `${cpu}%`;
  if (sidebarMem) sidebarMem.textContent = `${mem}%`;
}

function initSystemMonitorInterval() {
  setInterval(randomizeMetrics, 6000);
}
