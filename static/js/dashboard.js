/* ── RAG Evaluation Framework — Dashboard JS ── */

// ── Tab Navigation ──────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
        if (btn.dataset.tab === 'experiments') refreshExperiments();
        if (btn.dataset.tab === 'tradeoffs') loadTradeoffs();
    });
});

// ── Score Helpers ────────────────────────────────
function scoreBadge(val) {
    const v = parseFloat(val) || 0;
    const cls = v >= 0.75 ? 'high' : v >= 0.5 ? 'mid' : 'low';
    return `<span class="score-badge ${cls}">${v.toFixed(2)}</span>`;
}

function scoreColor(val) {
    const v = parseFloat(val) || 0;
    return v >= 0.75 ? '#34d399' : v >= 0.5 ? '#fbbf24' : '#fb7185';
}

// ── Load Demo Data ──────────────────────────────
async function loadDemoResults() {
    try {
        const res = await fetch('/api/demo-results', { method: 'POST' });
        const data = await res.json();
        if (data.message) {
            refreshExperiments();
            loadOverview();
            loadTradeoffs();
        }
    } catch (e) { console.error('Demo load error:', e); }
}

// ── Load Overview ───────────────────────────────
async function loadOverview() {
    try {
        const res = await fetch('/api/experiments');
        const data = await res.json();
        const exps = data.experiments || [];
        if (!exps.length) return;

        const latest = exps[0];
        const detailRes = await fetch(`/api/experiment/${latest.filename}`);
        const detail = await detailRes.json();
        const results = detail.results || {};
        const agg = results.aggregate || {};
        const perQ = results.per_question || [];

        let html = `
        <div class="metric-grid">
            <div class="metric-card faith">
                <div class="metric-label">Faithfulness</div>
                <div class="metric-value">${(agg.faithfulness || 0).toFixed(2)}</div>
                <div class="metric-desc">Is the answer grounded in context?</div>
                <div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:${(agg.faithfulness || 0) * 100}%"></div></div></div>
            </div>
            <div class="metric-card relevancy">
                <div class="metric-label">Answer Relevancy</div>
                <div class="metric-value">${(agg.answer_relevancy || 0).toFixed(2)}</div>
                <div class="metric-desc">Does the answer address the question?</div>
                <div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:${(agg.answer_relevancy || 0) * 100}%"></div></div></div>
            </div>
            <div class="metric-card recall">
                <div class="metric-label">Context Recall</div>
                <div class="metric-value">${(agg.context_recall || 0).toFixed(2)}</div>
                <div class="metric-desc">Did retrieval find the right docs?</div>
                <div class="score-bar-wrap"><div class="score-bar-bg"><div class="score-bar-fill" style="width:${(agg.context_recall || 0) * 100}%"></div></div></div>
            </div>
        </div>

        <div class="chart-section">
            <div class="glass-card chart-card">
                <h3>Metric Distribution — ${latest.experiment_name}</h3>
                <div class="radar-container">
                    ${renderRadar(agg)}
                </div>
            </div>
            <div class="glass-card chart-card">
                <h3>Score Distribution</h3>
                <div style="padding:20px 0">
                    ${renderDistribution(perQ)}
                </div>
            </div>
        </div>

        <div class="glass-card table-card">
            <h3>Per-Question Breakdown (${perQ.length} samples)</h3>
            <div style="overflow-x:auto">
            <table class="eval-table">
                <thead><tr>
                    <th>#</th><th>Question</th><th>Faithfulness</th><th>Relevancy</th><th>Recall</th>
                </tr></thead>
                <tbody>
                    ${perQ.map((q, i) => `<tr>
                        <td>${i + 1}</td>
                        <td class="q-cell" title="${escHtml(q.question)}">${escHtml(q.question)}</td>
                        <td>${scoreBadge(q.faithfulness)}</td>
                        <td>${scoreBadge(q.answer_relevancy)}</td>
                        <td>${scoreBadge(q.context_recall)}</td>
                    </tr>`).join('')}
                </tbody>
            </table>
            </div>
        </div>`;

        document.getElementById('overviewContent').innerHTML = html;
    } catch (e) { console.error('Overview error:', e); }
}

// ── SVG Radar ───────────────────────────────────
function renderRadar(agg) {
    const cx = 140, cy = 130, r = 100;
    const metrics = [
        { key: 'faithfulness', label: 'Faithfulness', angle: -90, color: '#8b5cf6' },
        { key: 'answer_relevancy', label: 'Relevancy', angle: 30, color: '#06b6d4' },
        { key: 'context_recall', label: 'Recall', angle: 150, color: '#f59e0b' }
    ];

    const gridLines = [0.25, 0.5, 0.75, 1.0].map(scale => {
        const pts = metrics.map(m => {
            const rad = (m.angle * Math.PI) / 180;
            return `${cx + r * scale * Math.cos(rad)},${cy + r * scale * Math.sin(rad)}`;
        }).join(' ');
        return `<polygon points="${pts}" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="1"/>`;
    }).join('');

    const dataPoints = metrics.map(m => {
        const val = agg[m.key] || 0;
        const rad = (m.angle * Math.PI) / 180;
        return { x: cx + r * val * Math.cos(rad), y: cy + r * val * Math.sin(rad), ...m, val };
    });

    const polygon = dataPoints.map(p => `${p.x},${p.y}`).join(' ');

    const labels = dataPoints.map(p => {
        const rad = (p.angle * Math.PI) / 180;
        const lx = cx + (r + 30) * Math.cos(rad);
        const ly = cy + (r + 30) * Math.sin(rad);
        return `<text x="${lx}" y="${ly}" text-anchor="middle" fill="${p.color}" font-size="11" font-weight="600">${p.label}</text>
                <text x="${lx}" y="${ly + 14}" text-anchor="middle" fill="#94a3b8" font-size="10">${p.val.toFixed(2)}</text>`;
    }).join('');

    const dots = dataPoints.map(p =>
        `<circle cx="${p.x}" cy="${p.y}" r="5" fill="${p.color}" stroke="rgba(0,0,0,0.3)" stroke-width="2"/>`
    ).join('');

    return `<svg viewBox="0 0 280 260" style="width:100%;max-width:280px">
        ${gridLines}
        <polygon points="${polygon}" fill="rgba(99,102,241,0.12)" stroke="rgba(99,102,241,0.5)" stroke-width="2"/>
        ${dots}${labels}
    </svg>`;
}

// ── Score Distribution Bars ─────────────────────
function renderDistribution(perQ) {
    if (!perQ.length) return '<p style="color:var(--text-muted)">No data</p>';
    const buckets = { faith: Array(10).fill(0), rel: Array(10).fill(0), rec: Array(10).fill(0) };
    perQ.forEach(q => {
        const fi = Math.min(9, Math.floor((q.faithfulness || 0) * 10));
        const ri = Math.min(9, Math.floor((q.answer_relevancy || 0) * 10));
        const ci = Math.min(9, Math.floor((q.context_recall || 0) * 10));
        buckets.faith[fi]++; buckets.rel[ri]++; buckets.rec[ci]++;
    });
    const max = Math.max(...buckets.faith, ...buckets.rel, ...buckets.rec, 1);
    const bw = 24, gap = 4, gw = bw * 3 + gap * 2;

    let bars = '';
    for (let i = 0; i < 10; i++) {
        const x = i * (gw + 8) + 30;
        const hf = (buckets.faith[i] / max) * 180;
        const hr = (buckets.rel[i] / max) * 180;
        const hc = (buckets.rec[i] / max) * 180;
        bars += `<rect x="${x}" y="${200 - hf}" width="${bw}" height="${hf}" rx="3" fill="#6366f1" opacity="0.7"/>`;
        bars += `<rect x="${x + bw + gap}" y="${200 - hr}" width="${bw}" height="${hr}" rx="3" fill="#06b6d4" opacity="0.7"/>`;
        bars += `<rect x="${x + (bw + gap) * 2}" y="${200 - hc}" width="${bw}" height="${hc}" rx="3" fill="#f59e0b" opacity="0.7"/>`;
        bars += `<text x="${x + gw / 2}" y="218" text-anchor="middle" fill="#64748b" font-size="9">${(i / 10).toFixed(1)}</text>`;
    }

    return `<svg viewBox="0 0 820 240" style="width:100%">
        <line x1="30" y1="200" x2="810" y2="200" stroke="rgba(255,255,255,0.08)" stroke-width="1"/>
        ${bars}
        <text x="10" y="204" fill="#64748b" font-size="9">0</text>
    </svg>
    <div class="bar-legend">
        <span class="legend-item"><span class="legend-dot faith"></span>Faithfulness</span>
        <span class="legend-item"><span class="legend-dot relevancy"></span>Relevancy</span>
        <span class="legend-item"><span class="legend-dot recall"></span>Recall</span>
    </div>`;
}

// ── Experiments List ────────────────────────────
async function refreshExperiments() {
    try {
        const res = await fetch('/api/experiments');
        const data = await res.json();
        const exps = data.experiments || [];
        const el = document.getElementById('experimentsTable');
        if (!exps.length) {
            el.innerHTML = '<div class="empty-state"><div class="empty-icon">🧪</div><h3>No experiments found</h3><p>Run an evaluation to see results here.</p></div>';
            return;
        }
        el.innerHTML = exps.map(e => {
            const a = e.aggregate || {};
            const cfg = e.pipeline_config || {};
            return `<div class="exp-card" onclick="showExperimentDetail('${e.filename}')">
                <div class="exp-info">
                    <h4>${escHtml(e.experiment_name)}</h4>
                    <div class="exp-meta">
                        <span>📅 ${new Date(e.timestamp).toLocaleString()}</span>
                        <span>📝 ${e.sample_count} samples</span>
                        ${cfg.chunk_size ? `<span>📏 chunk=${cfg.chunk_size}</span>` : ''}
                        ${cfg.retrieval_k ? `<span>🔍 k=${cfg.retrieval_k}</span>` : ''}
                    </div>
                </div>
                <div class="exp-scores">
                    <div class="exp-score faith"><div class="label">Faith</div><div class="val">${(a.faithfulness || 0).toFixed(2)}</div></div>
                    <div class="exp-score relevancy"><div class="label">Relev</div><div class="val">${(a.answer_relevancy || 0).toFixed(2)}</div></div>
                    <div class="exp-score recall"><div class="label">Recall</div><div class="val">${(a.context_recall || 0).toFixed(2)}</div></div>
                </div>
                <div class="exp-actions">
                    <button class="btn btn-sm btn-danger" onclick="event.stopPropagation(); deleteExperiment('${e.filename}')">✕</button>
                </div>
            </div>`;
        }).join('');
    } catch (e) { console.error('Experiments error:', e); }
}

// ── Experiment Detail Modal ─────────────────────
async function showExperimentDetail(filename) {
    try {
        const res = await fetch(`/api/experiment/${filename}`);
        const data = await res.json();
        const r = data.results || {};
        const agg = r.aggregate || {};
        const perQ = r.per_question || [];
        const cfg = data.pipeline_config || {};

        let html = `<h2 style="margin-bottom:8px">${escHtml(data.experiment_name)}</h2>
        <p style="color:var(--text-muted);font-size:0.82rem;margin-bottom:20px">${new Date(data.timestamp).toLocaleString()}</p>
        <div class="metric-grid" style="margin-bottom:20px">
            <div class="metric-card faith"><div class="metric-label">Faithfulness</div><div class="metric-value">${(agg.faithfulness||0).toFixed(2)}</div></div>
            <div class="metric-card relevancy"><div class="metric-label">Relevancy</div><div class="metric-value">${(agg.answer_relevancy||0).toFixed(2)}</div></div>
            <div class="metric-card recall"><div class="metric-label">Recall</div><div class="metric-value">${(agg.context_recall||0).toFixed(2)}</div></div>
        </div>
        <h3 style="margin-bottom:6px">Configuration</h3>
        <div style="background:rgba(15,23,42,0.6);padding:12px;border-radius:8px;font-family:var(--mono);font-size:0.8rem;color:var(--text-secondary);margin-bottom:20px">
            ${Object.entries(cfg).map(([k,v]) => `<div><span style="color:var(--accent-cyan)">${k}</span>: ${Array.isArray(v) ? v.join(', ') : v}</div>`).join('')}
        </div>
        <h3 style="margin-bottom:12px">Per-Question Results</h3>
        <div style="overflow-x:auto"><table class="eval-table"><thead><tr><th>#</th><th>Question</th><th>Faith</th><th>Relev</th><th>Recall</th></tr></thead><tbody>
        ${perQ.map((q, i) => `<tr><td>${i+1}</td><td class="q-cell" title="${escHtml(q.question)}">${escHtml(q.question)}</td><td>${scoreBadge(q.faithfulness)}</td><td>${scoreBadge(q.answer_relevancy)}</td><td>${scoreBadge(q.context_recall)}</td></tr>`).join('')}
        </tbody></table></div>`;

        document.getElementById('modalBody').innerHTML = html;
        document.getElementById('modalOverlay').classList.add('open');
    } catch (e) { console.error('Detail error:', e); }
}

function closeModal() { document.getElementById('modalOverlay').classList.remove('open'); }

// ── Delete Experiment ───────────────────────────
async function deleteExperiment(filename) {
    if (!confirm('Delete this experiment?')) return;
    await fetch(`/api/experiment/${filename}`, { method: 'DELETE' });
    refreshExperiments(); loadOverview(); loadTradeoffs();
}

// ── Trade-offs ──────────────────────────────────
async function loadTradeoffs() {
    try {
        const res = await fetch('/api/tradeoff');
        const data = await res.json();
        const exps = data.experiments || [];
        const best = data.best_per_metric || {};
        const el = document.getElementById('tradeoffContent');

        if (exps.length < 2) {
            el.innerHTML = '<div class="empty-state"><div class="empty-icon">⚖️</div><h3>Not Enough Data</h3><p>Run at least 2 experiments to see trade-off analysis.</p></div>';
            return;
        }

        const metrics = ['faithfulness', 'answer_relevancy', 'context_recall'];
        const labels = ['Faithfulness', 'Answer Relevancy', 'Context Recall'];
        const colors = ['#8b5cf6', '#06b6d4', '#f59e0b'];

        let cards = metrics.map((m, i) => {
            const b = best[m] || {};
            return `<div class="tradeoff-card">
                <h4>${labels[i]}</h4>
                <div class="tradeoff-best">🏆 Best: ${escHtml(b.best_experiment || '—')} (${(b.best_score || 0).toFixed(2)})</div>
                <div class="tradeoff-worst">📉 Worst: ${escHtml(b.worst_experiment || '—')} (${(b.worst_score || 0).toFixed(2)})</div>
                <div class="tradeoff-range" style="color:${colors[i]}">${((b.range || 0) * 100).toFixed(0)}%</div>
                <div class="tradeoff-range-label">Score Range</div>
            </div>`;
        }).join('');

        // Bar chart
        let barGroups = exps.map(e => {
            const a = e.aggregate || {};
            return `<div class="bar-group">
                <div class="bar-group-bars">
                    <div class="bar-item faith" style="height:${(a.faithfulness || 0) * 220}px" title="F: ${(a.faithfulness||0).toFixed(2)}"></div>
                    <div class="bar-item relevancy" style="height:${(a.answer_relevancy || 0) * 220}px" title="AR: ${(a.answer_relevancy||0).toFixed(2)}"></div>
                    <div class="bar-item recall" style="height:${(a.context_recall || 0) * 220}px" title="CR: ${(a.context_recall||0).toFixed(2)}"></div>
                </div>
                <div class="bar-label" title="${e.experiment_name}">${e.experiment_name}</div>
            </div>`;
        }).join('');

        el.innerHTML = `
            <div class="tradeoff-grid">${cards}</div>
            <div class="glass-card">
                <h3 style="margin-bottom:4px">Experiment Comparison</h3>
                <p style="color:var(--text-muted);font-size:0.8rem;margin-bottom:12px">${exps.length} experiments compared</p>
                <div class="comparison-chart">${barGroups}</div>
                <div class="bar-legend">
                    <span class="legend-item"><span class="legend-dot faith"></span>Faithfulness</span>
                    <span class="legend-item"><span class="legend-dot relevancy"></span>Relevancy</span>
                    <span class="legend-item"><span class="legend-dot recall"></span>Recall</span>
                </div>
            </div>`;
    } catch (e) { console.error('Tradeoff error:', e); }
}

// ── Run Pipeline ────────────────────────────────
let pollInterval = null;

async function runPipeline(event) {
    event.preventDefault();
    const body = {
        experiment_name: document.getElementById('experimentName').value,
        chunk_size: document.getElementById('chunkSize').value,
        chunk_overlap: document.getElementById('chunkOverlap').value,
        retrieval_k: document.getElementById('retrievalK').value,
        testset_size: document.getElementById('testsetSize').value,
        topics: document.getElementById('topics').value.split(',').map(t => t.trim()).filter(Boolean),
    };

    try {
        const res = await fetch('/api/run', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const data = await res.json();
        if (data.error) { alert(data.error); return; }

        document.getElementById('progressCard').style.display = 'block';
        document.getElementById('runBtn').disabled = true;
        updateIndicator('running', 'Running...');
        pollInterval = setInterval(pollStatus, 2000);
    } catch (e) { console.error('Run error:', e); alert('Failed to start pipeline'); }
}

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const s = await res.json();

        document.getElementById('progressBar').style.width = s.progress + '%';
        document.getElementById('progressPct').textContent = s.progress + '%';
        document.getElementById('progressStage').textContent = s.stage;
        document.getElementById('progressMessage').textContent = s.message;

        // Update stage indicators
        const stages = ['loading', 'chunking', 'generating', 'indexing', 'querying', 'evaluating', 'complete'];
        const currentIdx = stages.indexOf(s.stage);
        document.querySelectorAll('.stage-item').forEach(el => {
            const stageIdx = stages.indexOf(el.dataset.stage);
            el.classList.remove('active', 'done');
            if (stageIdx < currentIdx) el.classList.add('done');
            else if (stageIdx === currentIdx) el.classList.add('active');
        });

        if (s.stage === 'complete' || s.stage === 'error') {
            clearInterval(pollInterval);
            document.getElementById('runBtn').disabled = false;
            updateIndicator(s.stage === 'complete' ? 'ready' : 'error', s.stage === 'complete' ? 'Ready' : 'Error');
            if (s.stage === 'complete') { loadOverview(); refreshExperiments(); loadTradeoffs(); }
        }
    } catch (e) { console.error('Poll error:', e); }
}

function updateIndicator(state, text) {
    const el = document.getElementById('statusIndicator');
    el.className = 'status-indicator' + (state === 'running' ? ' running' : state === 'error' ? ' error' : '');
    el.querySelector('.status-text').textContent = text;
}

function escHtml(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

// ── Init ────────────────────────────────────────
loadOverview();
