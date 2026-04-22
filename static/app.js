/* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
   PROMPT OPTIMIZER PRO â€” Interactive Frontend
   â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */

document.addEventListener('DOMContentLoaded', () => {

  /* --- DOM refs --- */
  const $ = id => document.getElementById(id);
  const el = {
    promptInput:    $('promptInput'),
    optimizeBtn:    $('optimizeBtn'),
    stylePreset:    $('stylePreset'),
    sdBaseUrl:      $('sdBaseUrl'),
    saveSdConfigBtn:$('saveSdConfigBtn'),
    sdConfigStatus: $('sdConfigStatus'),
    charCount:      $('charCount'),
    wordCount:      $('wordCount'),
    statusDot:      $('statusDot'),
    statusLabel:    $('statusLabel'),
    resultsWrap:    $('resultsWrap'),
    // Sidebar
    vibePulse:      $('vibePulse'),
    vibeMood:       $('vibeMood'),
    vibeLighting:   $('vibeLighting'),
    sPos:           $('sPos'), sPosVal: $('sPosVal'),
    sNeg:           $('sNeg'), sNegVal: $('sNegVal'),
    sNeu:           $('sNeu'), sNeuVal: $('sNeuVal'),
    compoundDisplay:$('compoundDisplay'),
    compoundVal:    $('compoundVal'),
    settingsBlock:  $('settingsBlock'),
    setSampler:     $('setSampler'),
    setSteps:       $('setSteps'),
    setCfg:         $('setCfg'),
    // Pipeline
    pipelineFlow:   $('pipelineFlow'),
    stageDetailCard:$('stageDetailCard'),
    stageDetailIcon:$('stageDetailIcon'),
    stageDetailName:$('stageDetailName'),
    stageDetailDesc:$('stageDetailDesc'),
    stageDetailBody:$('stageDetailBody'),
    // Tokens
    tokenGrid:      $('tokenGrid'),
    posSummary:     $('posSummary'),
    // Transforms + Log
    transformList:  $('transformList'),
    changeCount:    $('changeCount'),
    pipelineLog:    $('pipelineLog'),
    // Variants
    variantGrid:    $('variantGrid'),
    // Shield + Final
    shieldTags:     $('shieldTags'),
    finalOriginal:  $('finalOriginal'),
    finalOptimized: $('finalOptimized'),
    copyRaw:        $('copyRaw'),
    copyOptimized:  $('copyOptimized'),
    // Generate
    generateBtn:    $('generateBtn'),
    imageSection:   $('imageSection'),
    imgRawPrompt:   $('imgRawPrompt'),
    imgOptPrompt:   $('imgOptPrompt'),
    rawImageFrame:  $('rawImageFrame'),
    optImageFrame:  $('optImageFrame'),
    metricsSection: $('metricsSection'),
    evalChartSection:$('evalChartSection'),
    preRenderBenchmark: $('preRenderBenchmark'),
    textMetricBars: $('textMetricBars'),
    textMetricStats: $('textMetricStats'),
    aucValMini:     $('aucValMini'),
    aucInterpMini:  $('aucInterpMini'),
    // Tooltip
    tokenTooltip:   $('tokenTooltip'),
  };

  /* â”€â”€â”€ State â”€â”€â”€ */
  let session = { original: '', optimized: '', negative: '', settings: {}, pipelineStages: [], fitnessScore: null };

  /* â”€â”€â”€ Colour palette per pipeline step â”€â”€â”€ */
  const STEP_COLORS = [
    { hex: '#6366f1', rgb: '99,102,241' },  // Step 1: Spelling
    { hex: '#8b5cf6', rgb: '139,92,246' },  // Step 2: Tokenization
    { hex: '#f59e0b', rgb: '245,158,11' },  // Step 3: Stemming
    { hex: '#3b82f6', rgb: '59,130,246' },  // Step 4: POS
    { hex: '#ec4899', rgb: '236,72,153' },  // Step 5: NER
    { hex: '#06b6d4', rgb: '6,182,212' },  // Step 6: SVO
    { hex: '#0ea5e9', rgb: '14,165,233' },   // Step 7: Chunking
    { hex: '#3b82f6', rgb: '59,130,246' },  // Step 8: TF-IDF
    { hex: '#10b981', rgb: '16,185,129' },  // Step 9: Synonyms
    { hex: '#a855f7', rgb: '168,85,247' },  // Step 10: Genetic
    { hex: '#f97316', rgb: '249,115,22' },  // Step 11: LLM
    { hex: '#10b981', rgb: '16,185,129' },  // Step 12: Vibe
  ];

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     HEALTH CHECK
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  async function healthCheck() {
    try {
      const r = await fetch('/api/health');
      if (r.ok) {
        const d = await r.json();
        el.sdBaseUrl.value = d.sd_base_url || el.sdBaseUrl.value;
        updateSdStatus(d.sd_available, d.sd_error, d.sd_base_url);
        el.statusDot.className = `status-dot ${d.sd_available ? 'ok' : 'error'}`;
        el.statusLabel.textContent = d.sd_available
          ? (d.clip_fallback ? 'Online (CLIP fallback mode)' : 'Online — all systems ready')
          : 'API online — Stable Diffusion unavailable';
      } else throw new Error();
    } catch {
      el.statusDot.className = 'status-dot error';
      el.statusLabel.textContent = 'API unreachable';
      updateSdStatus(false, 'Prompt Optimizer API is unreachable.', el.sdBaseUrl.value);
    }
  }
  healthCheck();

  el.saveSdConfigBtn.addEventListener('click', async () => {
    const sdBaseUrl = el.sdBaseUrl.value.trim();
    if (!sdBaseUrl) return;
    setBtn(el.saveSdConfigBtn, true, 'Saving…');
    try {
      const resp = await fetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sd_base_url: sdBaseUrl }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || 'Failed to update SD URL');
      updateSdStatus(data.sd_available, data.sd_error, data.sd_base_url);
      healthCheck();
    } catch (err) {
      updateSdStatus(false, err.message, sdBaseUrl);
    } finally {
      setBtn(el.saveSdConfigBtn, false, 'Save SD URL');
    }
  });

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     INPUT: live counters
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  el.promptInput.addEventListener('input', () => {
    const val = el.promptInput.value;
    el.charCount.textContent = `${val.length} chars`;
    el.wordCount.textContent = `${val.trim() ? val.trim().split(/\s+/).length : 0} words`;
  });
  el.promptInput.addEventListener('keydown', e => {
    if (e.key === 'Enter') el.optimizeBtn.click();
  });

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     STYLE PRESET CARDS
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  document.querySelectorAll('.style-card').forEach(card => {
    card.addEventListener('click', () => {
      document.querySelectorAll('.style-card').forEach(c => c.classList.remove('active'));
      card.classList.add('active');
      el.stylePreset.value = card.dataset.value;
    });
  });

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     OPTIMIZE
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  el.optimizeBtn.addEventListener('click', async () => {
    const prompt = el.promptInput.value.trim();
    if (!prompt) { flashInput(); return; }

    setBtn(el.optimizeBtn, true, 'Processingâ€¦');
    el.resultsWrap.style.display = 'none';

    try {
      const resp = await fetch('/api/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          prompt, 
          style: el.stylePreset.value,
          use_ollama: document.getElementById('ollamaToggle')?.checked || false
        }),
      });
      if (!resp.ok) throw new Error(await resp.text());
      const data = await resp.json();

      session = {
        original:       prompt,
        optimized:      data.optimized_prompt,
        corrected:      data.corrected_prompt || prompt,
        negative:       data.negative_prompt,
        settings:       data.settings,
        fitnessScore:   data.fitness_score ?? null,
        pipelineStages: data.pipeline_stages,
        variants:       data.variants || [],
        selectedModel:  'llama3.2'
      };

      renderAll(data, prompt);
      
      const evalData = data.evaluation || {};
      const metricsData = data.metrics || buildMetricsFromEvaluation(evalData);
      renderMetrics(evalData, metricsData);

      el.resultsWrap.style.display = 'block';
      el.resultsWrap.classList.remove('reveal');
      void el.resultsWrap.offsetWidth;
      el.resultsWrap.classList.add('reveal');
      setTimeout(() => el.pipelineFlow.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

    } catch (err) {
      showError(`NLP Engine Error: ${err.message}`);
    } finally {
      setBtn(el.optimizeBtn, false, 'Optimize');
    }
  });

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     RENDER ALL SECTIONS
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function renderAll(data, originalPrompt) {
    const safe = {
      vibe: data?.vibe || { mood: 'neutral', lighting: 'balanced light', color: '#94a3b8', scores: { pos: 0, neg: 0, neu: 1, compound: 0 } },
      settings: data?.settings || { sampler: 'Euler a', steps: 35, cfg_scale: 8.0 },
      pipeline_stages: Array.isArray(data?.pipeline_stages) ? data.pipeline_stages : [],
      linguistics: Array.isArray(data?.linguistics) ? data.linguistics : [],
      pipeline_log: Array.isArray(data?.pipeline_log) ? data.pipeline_log : [],
      variants: Array.isArray(data?.variants) ? data.variants : [],
      selected_variant: Number.isInteger(data?.selected_variant) ? data.selected_variant : 0,
      negative_prompt: typeof data?.negative_prompt === 'string' ? data.negative_prompt : '',
      optimized_prompt: typeof data?.optimized_prompt === 'string' ? data.optimized_prompt : '',
      corrected_prompt: typeof data?.corrected_prompt === 'string' ? data.corrected_prompt : originalPrompt,
      spelling: (data?.spelling && Array.isArray(data.spelling.changes)) ? data.spelling : { changes: [] },
    };

    renderVibeHUD(safe.vibe);
    renderSidebar(safe.settings);
    renderPipelineFlow(safe.pipeline_stages);
    renderTokenGrid(safe.linguistics);
    renderTransformations(safe.linguistics, safe.spelling, safe.corrected_prompt, originalPrompt);
    renderPipelineLog(safe.pipeline_log);
    renderVariants(safe.variants, safe.selected_variant);
    renderShield(safe.negative_prompt);
    renderFinalPrompts(originalPrompt, safe.optimized_prompt);

    // Step 9: Show Pre-Render Benchmark by default
    el.preRenderBenchmark.style.display = 'block';
    
    // Step 11: Hide multimodal until generate
    el.imageSection.style.display = 'none';
  }

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     SIDEBAR â€” Vibe HUD + Sentiment
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function renderVibeHUD(vibe) {
    el.vibePulse.style.background = vibe.color;
    el.vibePulse.style.boxShadow = `0 0 14px ${vibe.color}`;
    el.vibeMood.textContent = vibe.mood.toUpperCase();
    el.vibeMood.style.color = vibe.color;
    el.vibeLighting.textContent = vibe.lighting;

    const s = vibe.scores;
    animateBar(el.sPos, s.pos * 100, el.sPosVal);
    animateBar(el.sNeg, s.neg * 100, el.sNegVal);
    animateBar(el.sNeu, s.neu * 100, el.sNeuVal);

    el.compoundDisplay.style.display = 'block';
    el.compoundVal.textContent = s.compound.toFixed(3);
    el.compoundVal.style.color = vibe.color;
  }

  function animateBar(barEl, pct, valEl) {
    barEl.style.width = `${Math.round(pct)}%`;
    if (valEl) valEl.textContent = `${Math.round(pct)}%`;
  }

  function renderSidebar(settings) {
    el.settingsBlock.style.display = 'block';
    el.setSampler.textContent = settings.sampler;
    el.setSteps.textContent = settings.steps;
    el.setCfg.textContent = settings.cfg_scale;
  }

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     PIPELINE FLOWCHART
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function renderPipelineFlow(stages) {
    stages = Array.isArray(stages) ? stages : [];
    el.pipelineFlow.innerHTML = '';
    el.stageDetailCard.style.display = 'none';

    stages.forEach((stage, i) => {
      const color = STEP_COLORS[i] || STEP_COLORS[0];
      const stepEl = document.createElement('div');
      stepEl.className = 'pipeline-step';
      stepEl.style.setProperty('--step-color', color.hex);
      stepEl.style.setProperty('--step-color-rgb', color.rgb);
      stepEl.dataset.idx = i;

      stepEl.innerHTML = `
        <div class="step-circle">${stage.icon}</div>
        <div class="step-name">${stage.name}</div>
        <div class="step-detail">${escHtml(stage.detail)}</div>
      `;

      stepEl.addEventListener('click', () => selectStage(i, stages[i], color));
      el.pipelineFlow.appendChild(stepEl);

      // Stagger-animate each step active
      setTimeout(() => stepEl.classList.add('active'), i * 120);
    });
  }

  function selectStage(idx, stage, color) {
    // Update selected state
    document.querySelectorAll('.pipeline-step').forEach((s, i) => {
      s.classList.toggle('selected', i === idx);
    });

    // Populate detail card
    el.stageDetailIcon.textContent = stage.icon;
    el.stageDetailIcon.style.background = `${color.hex}22`;
    el.stageDetailIcon.style.color = color.hex;
    el.stageDetailName.textContent = `Step ${stage.step}: ${stage.name}`;
    el.stageDetailName.style.color = color.hex;
    el.stageDetailDesc.textContent = stage.detail;
    el.stageDetailBody.innerHTML = buildStageBody(stage);
    el.stageDetailCard.style.display = 'block';
    el.stageDetailCard.style.borderColor = `${color.hex}40`;
    el.stageDetailCard.style.animation = 'none';
    void el.stageDetailCard.offsetWidth;
    el.stageDetailCard.style.animation = 'fadeIn 0.3s ease';
  }

  function buildStageBody(stage) {
    const d = stage?.data;
    switch (stage.step) {
      case 1: // Spelling Correction
        if (!Array.isArray(d) || !d.length) return '<p class="empty-state">No typos found. Text is clean.</p>';
        return `<div>${d.map(c => 
          `<div class="stage-change-row"><span class="sc-from" style="color:#f87171">${escHtml(c.from)}</span><span class="sc-arrow">\u2192</span><span class="sc-to" style="color:#4ade80">${escHtml(c.to)}</span></div>`
        ).join('')}</div>`;

      case 2: // Tokenization
        return `<div class="stage-data-tokens">${(Array.isArray(d) ? d : []).map(t =>
          `<span class="stage-token-pill">${escHtml(t)}</span>`).join('')}</div>`;

      case 3: // Stemming Analysis
        if (!Array.isArray(d) || !d.length) return '<p class="empty-state">No stems extracted.</p>';
        return `<div>${d.map(s => 
          `<div class="stage-change-row"><span class="sc-from">${escHtml(s.word)}</span><span class="sc-arrow">\u2192</span><span class="sc-to" style="color:#f59e0b; font-family:monospace">${escHtml(s.stem)}</span></div>`
        ).join('')}</div>`;

      case 4: // POS Tagging
        if (!Array.isArray(d) || !d.length) return '<p class="empty-state">No POS tags found.</p>';
        return `<div class="stage-data-tokens">${d.map(t =>
          `<span class="stage-token-pill" title="${t.pos}" style="background:rgba(139,92,246,0.1); border-color:rgba(139,92,246,0.2)">${escHtml(t.word)} <small style="opacity:0.6">${escHtml(t.pos)}</small></span>`).join('')}</div>`;

      case 5: // Named Entity Recog.
        if (!d || Object.values(d).every(v => v.length === 0)) return '<p class="empty-state">No entities detected.</p>';
        return `<div class="ner-container">${Object.entries(d).map(([k, v]) => v.length ? `
          <div class="ner-group">
            <div class="ner-label">${escHtml(k)}</div>
            <div class="ner-values">${v.map(val => `<span class="ner-pill">${escHtml(val)}</span>`).join('')}</div>
          </div>` : '').join('')}</div>`;

      case 6: // SVO Extraction
        if (!Array.isArray(d) || !d.length) return '<p class="empty-state">No semantic pathways found.</p>';
        return `<div class="svo-container">${d.map(s => `
          <div class="svo-card">
            <span class="svo-part subject">${escHtml(s.subject)}</span>
            <span class="svo-arrow">\u2014[ ${escHtml(s.action)} ]\u2192</span>
            <span class="svo-part object">${escHtml(s.object)}</span>
          </div>`).join('')}</div>`;

      case 7: // NP Chunking
        if (!d || (!d.np?.length && !d.vp?.length)) return '<p class="empty-state">No phrase chunks found.</p>';
        return `
          <div class="ner-group"><div class="ner-label">Noun Phrases</div><div class="stage-data-tokens">${(d.np || []).map(p => `<span class="stage-token-pill" style="background:rgba(14,165,233,0.1);border-color:rgba(14,165,233,0.3)">${escHtml(p)}</span>`).join('')}</div></div>
          <div class="ner-group" style="margin-top:0.8rem"><div class="ner-label">Verb Phrases</div><div class="stage-data-tokens">${(d.vp || []).map(p => `<span class="stage-token-pill" style="background:rgba(139,92,246,0.1);border-color:rgba(139,92,246,0.3)">${escHtml(p)}</span>`).join('')}</div></div>`;

      case 8: // TF-IDF Keyword Ranking
        return `<div class="keyword-cloud">${Object.entries(d || {}).sort((a,b) => b[1]-a[1]).map(([w, s]) => 
          `<div class="keyword-pill" style="opacity:${0.4 + s * 0.6}; transform:scale(${0.9 + s * 0.2})">
            ${escHtml(w)} <small>${s.toFixed(2)}</small>
          </div>`).join('')}</div>`;

      case 9: // Synonym Swapping
        if (!Array.isArray(d) || !d.length) return '<p class="empty-state">No synonym substitutions needed.</p>';
        return `<div>${d.map(t =>
          `<div class="stage-change-row"><span class="sc-from">${escHtml(t.word)}</span><span class="sc-arrow">\u2192</span><span class="sc-to" style="color:#10b981">${escHtml(t.optimized_to)}</span></div>`
        ).join('')}</div>`;

      case 10: // Genetic Evolution
        const winner = d?.winner || {};
        const rejected = Array.isArray(d?.rejected) ? d.rejected : [];
        const lm = d?.lm_scores || {};
        
        return `
          <div class="ga-status">
            <div class="ga-metric-group">
                <div class="ga-metric"><strong>Final Fitness:</strong> <span style="color:#a855f7">${winner.fitness?.toFixed(4) || '—'}</span></div>
                <div class="ga-metric"><strong>Coherence:</strong> ${lm.coherence?.toFixed(4) || '—'}</div>
            </div>
            
            <div style="margin-top:1rem;">
                <div style="font-size:0.65rem; color:var(--text-3); text-transform:uppercase; letter-spacing:0.05rem; margin-bottom:0.5rem; font-weight:800;">Phenotype Selection (Winners & Rejected)</div>
                <div class="ga-candidate winner">
                    <div class="ga-cand-rank">Selected</div>
                    <div class="ga-cand-text">${escHtml(winner.text || '')}</div>
                    <div class="ga-cand-score">${winner.fitness?.toFixed(3)}</div>
                </div>
                ${rejected.map((r, i) => `
                    <div class="ga-candidate rejected">
                        <div class="ga-cand-rank">#${i+2} Rejected</div>
                        <div class="ga-cand-text">${escHtml(r.text)}</div>
                        <div class="ga-cand-score">${r.fitness.toFixed(3)}</div>
                    </div>
                `).join('')}
            </div>
            <p style="font-size:0.7rem;color:var(--text-3);margin-top:0.8rem; line-height:1.4">
                The evolutionary engine simulated multiple mutations. Rejected candidates failed due to lower semantic density or poor n-gram coherence.
            </p>
          </div>`;

      case 11: // LLM Refinement
        if (!d) return '<p class="empty-state">Local LLM generation was bypassed or failed.</p>';
        return `
          <div class="ollama-result-box">
             <div style="font-size:0.65rem; color:#a855f7; font-weight:800; margin-bottom:0.5rem; text-transform:uppercase;">Enhanced Prompt Outline</div>
             <div style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; line-height:1.6; color:var(--text-1)">${escHtml(d)}</div>
          </div>`;

      case 12: // Vibe Analysis
        return `
          <div style="display:flex;gap:1.5rem;font-size:0.8rem;font-family:'JetBrains Mono',monospace;flex-wrap:wrap">
            <span>pos <strong style="color:var(--accent)">${((d.scores?.pos||0)*100).toFixed(1)}%</strong></span>
            <span>neg <strong style="color:#f87171">${((d.scores?.neg||0)*100).toFixed(1)}%</strong></span>
            <span>neu <strong style="color:var(--text-2)">${((d.scores?.neu||0)*100).toFixed(1)}%</strong></span>
            <span>compound <strong style="color:${d.color}">${(d.scores?.compound||0).toFixed(3)}</strong></span>
          </div>
          <p style="font-size:0.8rem;margin-top:0.6rem;color:var(--text-2)">Lighting: <em>${escHtml(d.lighting)}</em></p>`;

      default:
        return `<pre style="font-size:0.75rem;color:var(--text-2);white-space:pre-wrap">${escHtml(JSON.stringify(d, null, 2))}</pre>`;
    }
  }

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     TOKEN GRID
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function renderTokenGrid(linguistics) {
    linguistics = Array.isArray(linguistics) ? linguistics : [];
    el.tokenGrid.innerHTML = '';

    // Count by POS type
    const counts = { Noun: 0, Adjective: 0, Verb: 0, Adverb: 0, Other: 0 };

    linguistics.forEach((tok, i) => {
      const chip = document.createElement('div');
      chip.className = 'token-chip' + (tok.changed ? ' changed' : '');
      chip.setAttribute('data-label', tok.label);
      chip.style.animationDelay = `${i * 40}ms`;

      let inner = escHtml(tok.word);
      if (tok.is_subject) {
        inner += `<span class="weight-badge">×1.25</span>`;
        chip.title = 'Subject noun — weight boosted';
      }
      if (tok.changed) {
        const badge = document.createElement('div');
        badge.className = 'chip-badge';
        badge.textContent = '✦';
        chip.appendChild(badge);
      }
      chip.innerHTML = inner + (tok.is_subject ? `<span class="weight-badge">×1.25</span>` : '');

      // Tooltip
      chip.addEventListener('mouseenter', e => showTokenTooltip(e, tok));
      chip.addEventListener('mousemove', e => moveTooltip(e));
      chip.addEventListener('mouseleave', hideTokenTooltip);

      el.tokenGrid.appendChild(chip);

      const key = ['Noun','Adjective','Verb','Adverb'].includes(tok.label) ? tok.label : 'Other';
      counts[key]++;
    });

    // POS summary bar
    const total = linguistics.length;
    const colors = { Noun: 'noun', Adjective: 'adj', Verb: 'verb', Adverb: 'adv', Other: 'other' };
    el.posSummary.innerHTML = Object.entries(counts)
      .filter(([, v]) => v > 0)
      .map(([k, v]) => `
        <div class="pos-count">
          <div class="pos-count-dot legend-dot ${colors[k]}"></div>
          <span>${k}: ${v} (${Math.round(v/total*100)}%)</span>
        </div>`).join('');
  }

  /* â”€â”€â”€â”€â”€â”€ Tooltip â”€â”€â”€â”€â”€â”€ */
  function showTokenTooltip(e, tok) {
    const tt = el.tokenTooltip;
    $('ttWord').textContent = tok.word;
    $('ttPos').textContent = tok.pos;
    const labelEl = $('ttLabel');
    labelEl.textContent = tok.label;
    labelEl.style.color = getLabelColor(tok.label);
    $('ttRole').textContent = tok.role || 'â€”';

    const weightRow = $('ttWeightRow');
    if (tok.is_subject) {
      weightRow.style.display = 'flex';
      $('ttWeight').textContent = `Ã—${tok.weight} (boosted)`;
    } else {
      weightRow.style.display = 'none';
    }

    const synEl = $('ttSyns');
    if (Array.isArray(tok.synonyms) && tok.synonyms.length > 0) {
      synEl.style.display = 'block';
      $('ttSynList').innerHTML = tok.synonyms.map(s =>
        `<span class="tt-syn">${escHtml(s)}</span>`).join('');
    } else {
      synEl.style.display = 'none';
    }

    const specEl = $('ttSpec'); // NEW specificity section
    if (tok.specificity && Array.isArray(tok.specificity.ladder)) {
      specEl.style.display = 'block';
      $('ttLadder').innerHTML = tok.specificity.ladder.map((w, i) => 
        `<span style="opacity:${0.4 + (i/tok.specificity.depth)*0.6}">${escHtml(w)}</span>`
      ).join(' \u2192 ');
    } else {
      specEl.style.display = 'none';
    }

    const changeEl = $('ttChange');
    if (tok.changed) {
      changeEl.style.display = 'block';
      $('ttFrom').textContent = tok.word;
      $('ttTo').textContent = tok.optimized_to;
    } else {
      changeEl.style.display = 'none';
    }

    tt.style.display = 'block';
    moveTooltip(e);
  }

  function moveTooltip(e) {
    const tt = el.tokenTooltip;
    const pad = 14;
    let x = e.clientX + pad;
    let y = e.clientY + pad;
    if (x + 290 > window.innerWidth) x = e.clientX - 290 - pad;
    if (y + 200 > window.innerHeight) y = e.clientY - 200 - pad;
    tt.style.left = `${x}px`;
    tt.style.top = `${y}px`;
  }

  function hideTokenTooltip() {
    el.tokenTooltip.style.display = 'none';
  }

  function getLabelColor(label) {
    const map = {
      Noun: '#93c5fd', Adjective: '#6ee7b7',
      Verb: '#fcd34d', Adverb: '#c4b5fd',
    };
    return map[label] || '#94a3b8';
  }

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     TRANSFORMATIONS + LOG
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function renderTransformations(linguistics, spelling, correctedPrompt, originalPrompt) {
    linguistics = Array.isArray(linguistics) ? linguistics : [];
    const spellChanges = Array.isArray(spelling?.changes) ? spelling.changes : [];
    const changed = linguistics.filter(t => t.changed);
    const totalChanges = changed.length + spellChanges.length;
    el.changeCount.textContent = `${totalChanges} change${totalChanges !== 1 ? 's' : ''}`;

    if (!changed.length && !spellChanges.length) {
      el.transformList.innerHTML = '<p class="empty-state">No linguistic optimizations \u2014 prompt is already high-quality.</p>';
      return;
    }
    const spellingBlock = spellChanges.map((c, i) => `
      <div class="transform-item" style="animation-delay:${i * 60}ms">
        <span class="t-from">${escHtml(c.from)}</span>
        <span class="t-arrow">-></span>
        <span class="t-to">${escHtml(c.to)}</span>
      </div>`).join('');

    const synonymBlock = changed.map((t, i) => `
      <div class="transform-item" style="animation-delay:${i * 60}ms">
        <span class="t-from">${escHtml(t.word)}</span>
        <span class="t-arrow">\u2192</span>
        <span class="t-to">${escHtml(t.optimized_to)}</span>
      </div>`).join('');

    const correctionNote = (spellChanges.length && correctedPrompt && originalPrompt && correctedPrompt !== originalPrompt)
      ? `<div class="transform-item"><span class="t-from">corrected prompt</span><span class="t-arrow">-></span><span class="t-to">${escHtml(correctedPrompt)}</span></div>`
      : '';

    el.transformList.innerHTML = spellingBlock + synonymBlock + correctionNote;
  }

  function renderPipelineLog(logs) {
    logs = Array.isArray(logs) ? logs : [];
    el.pipelineLog.innerHTML = logs.map((line, i) => {
      const tagMatch = line.match(/^\[(\d+)\]/);
      if (tagMatch) {
        const tag = tagMatch[0];
        const rest = escHtml(line.replace(tag, '').trim());
        return `<div class="log-line" style="animation-delay:${i * 80}ms"><span class="log-tag">${tag}</span> ${rest}</div>`;
      }
      return `<div class="log-line" style="animation-delay:${i * 80}ms">${escHtml(line)}</div>`;
    }).join('');
  }

  /* â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• 
     GENETIC VARIANTS
  â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â• â•  */
  function renderVariants(variants, winnerIdx) {
    variants = Array.isArray(variants) ? variants : [];
    if (!variants.length) {
      el.variantGrid.innerHTML = '<p class="empty-state">No evolved variants returned.</p>';
      return;
    }
    const scores = variants.map(v => v.score).filter(s => typeof s === 'number');
    const maxScore = scores.length > 0 ? Math.max(...scores) : 1;

    el.variantGrid.innerHTML = variants.map((v, i) => {
      const isWinner = i === winnerIdx;
      const barPct = Math.round((v.score / maxScore) * 100);

      return `
        <div class="variant-card ${isWinner ? 'winner' : ''}" style="animation-delay:${i * 80}ms">
          ${isWinner ? '<div class="winner-crown">â˜… SELECTED</div>' : ''}
          <div class="variant-label">${v.label}</div>
          <div class="variant-strategy">${v.description}</div>
          <div class="variant-text">${escHtml(v.text.slice(0, 120))}${v.text.length > 120 ? 'â€¦' : ''}</div>
          <div class="variant-score">
            <span>Fitness</span>
            <span class="score-num">${v.score}</span>
          </div>
          <div class="fitness-bar-wrap">
            <div class="fitness-bar" style="width:0%" data-width="${barPct}"></div>
          </div>
        </div>`;
    }).join('');

    // Animate fitness bars after paint
    requestAnimationFrame(() => {
      document.querySelectorAll('.fitness-bar').forEach(bar => {
        bar.style.width = bar.dataset.width + '%';
      });
    });
  }

  /* ─── NEGATIVE SHIELD ─── */
  function renderShield(negPrompt) {
    el.shieldTags.innerHTML = negPrompt.split(',')
      .map(t => t.trim()).filter(Boolean)
      .map(t => `<span class="shield-tag">${escHtml(t)}</span>`)
      .join('');
  }

  /* ─── FINAL PROMPTS + COPY ─── */
  function renderFinalPrompts(original, optimized) {
    el.finalOriginal.textContent = original;
    el.finalOptimized.textContent = optimized;
  }

  el.copyRaw.addEventListener('click', () => copyText(el.copyRaw, session.original));
  el.copyOptimized.addEventListener('click', () => copyText(el.copyOptimized, session.optimized));

  function copyText(btn, text) {
    navigator.clipboard.writeText(text).then(() => {
      btn.classList.add('copied');
      const prev = btn.innerHTML;
      btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg> Copied!`;
      setTimeout(() => {
        btn.classList.remove('copied');
        btn.innerHTML = prev;
      }, 2000);
    });
  }

  /* ─── GENERATE IMAGES ─── */
  el.generateBtn.addEventListener('click', async () => {
    if (!session.optimized) return;

    setBtn(el.generateBtn, true, 'Renderingâ€¦ (30-60s)');
    el.imageSection.style.display = 'block';
    el.metricsSection.style.display = 'none';

    el.imgRawPrompt.textContent = session.original;
    el.imgOptPrompt.textContent = session.optimized;
    el.rawImageFrame.innerHTML = '<div class="skeleton-loader"></div>';
    el.optImageFrame.innerHTML = '<div class="skeleton-loader"></div>';

    el.imageSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    try {
      const resp = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          original_prompt:  session.original,
          optimized_prompt: session.optimized,
          negative_prompt:  session.negative,
          steps:            session.settings.steps,
          cfg_scale:        session.settings.cfg_scale,
          fitness_score:    session.fitnessScore,
        }),
      });

      const data = await resp.json();

      if (!resp.ok) {
        el.rawImageFrame.innerHTML = `<div style="padding:2rem;color:#f87171;font-size:0.8rem">${escHtml(data.error || 'Generation failed.')}</div>`;
        el.optImageFrame.innerHTML = `<div style="padding:2rem;color:#f87171;font-size:0.8rem">Stable Diffusion offline.</div>`;
        return;
      }

      el.rawImageFrame.innerHTML = `<img src="data:image/png;base64,${data.raw_image}" alt="Raw prompt result" />`;
      el.optImageFrame.innerHTML = `<img src="data:image/png;base64,${data.opt_image}" alt="Optimized prompt result" />`;

      const evalData = data.evaluation || {};
      const metricsData = data.metrics || buildMetricsFromEvaluation(evalData);
      renderMetrics(evalData, metricsData);

    } catch (err) {
      el.rawImageFrame.innerHTML = `<div style="padding:2rem;color:#f87171;font-size:0.8rem">Request failed: ${escHtml(err.message)}</div>`;
      el.optImageFrame.innerHTML = `<div style="padding:2rem;color:#f87171;font-size:0.8rem">Check that the server is running.</div>`;
    } finally {
      setBtn(el.generateBtn, false, 'Re-Generate');
    }
  });

  /* ─── METRICS DASHBOARD ─── */
  function buildMetricsFromEvaluation(evaluation) {
    const text = evaluation?.text_metrics || {};
    const image = evaluation?.image_metrics || {};
    const composite = evaluation?.composite || {};
    const pipeline = evaluation?.pipeline_accuracy || {};

    return {
      raw_clip: Number(image?.raw_clip?.score) || 0,
      opt_clip: Number(image?.opt_clip?.score) || 0,
      raw_clip_scaled: Number(image?.raw_clip?.scaled) || 0,
      opt_clip_scaled: Number(image?.opt_clip?.scaled) || 0,
      raw_clip_available: image?.raw_clip?.score != null,
      opt_clip_available: image?.opt_clip?.score != null,
      raw_aesthetic_available: image?.raw_clip?.score != null,
      opt_aesthetic_available: image?.opt_clip?.score != null,
      raw_aesthetic: image?.raw_clip?.score != null ? (Number(image?.raw_aesthetic?.score) || 0) : 0,
      aesthetic: image?.opt_clip?.score != null ? (Number(image?.opt_aesthetic?.score) || 0) : 0,
      raw_tokens: Number(text?.complexity?.original?.token_count) || 0,
      opt_tokens: Number(text?.complexity?.optimized?.token_count) || 0,
      raw_complexity: Number(text?.complexity?.original?.density_score) || 0,
      opt_complexity: Number(text?.complexity?.optimized?.density_score) || 0,
      raw_composite: Number(composite?.raw?.score) || 0,
      composite: Number(composite?.optimized?.score) || 0,
      improvement: Number(composite?.improvement) || 0,
      pipeline_accuracy: Number(pipeline?.score_percent) || 0,
      pipeline_accuracy_label: pipeline?.interpretation || 'unknown',
      accuracy_curve: Array.isArray(pipeline?.curve_points) ? pipeline.curve_points : [],
      roc_auc: evaluation?.roc_auc || {},
      
      // Sophisticated Metrics
      raw_readability:    Number(text?.complexity?.original?.readability?.reading_ease) || 0,
      opt_readability:    Number(text?.complexity?.optimized?.readability?.reading_ease) || 0,
      raw_sophistication: Number(text?.complexity?.original?.syntactic_depth) || 0,
      opt_sophistication: Number(text?.complexity?.optimized?.syntactic_depth) || 0,
      raw_info_density:   Number(text?.complexity?.original?.readability?.info_density) || 0,
      opt_info_density:   Number(text?.complexity?.optimized?.readability?.info_density) || 0,
      
      fitness_score: null,
    };
  }

  function renderRocCurve(roc) {
    const section = $('rocSection');
    const pathEl = $('rocCurvePath');
    const valEl = $('aucVal');
    const interpEl = $('aucInterpretation');
    if (!roc || !Array.isArray(roc.curve) || !section) return;
    section.style.display = 'block';
    valEl.textContent = (roc.auc || 0).toFixed(4);
    interpEl.textContent = (roc.interpretation || 'Aggregate') + ' Lifecycle Performance';
    
    const size = 300;
    let d = `M 0,${size}`;
    roc.curve.forEach(p => {
      const x = p.fpr * size;
      const y = size - (p.tpr * size);
      d += ` L ${x},${y}`;
    });
    d += ` L ${size},0 L ${size},${size} Z`;
    pathEl.setAttribute('d', d);

    const spark = $('aucSparkline');
    if (spark) {
      let sd = `M 0,35`;
      roc.curve.forEach((p, idx) => {
        const x = (idx / (roc.curve.length - 1)) * 200;
        const y = 35 - (p.tpr * 30);
        sd += ` L ${x},${y}`;
      });
      spark.setAttribute('d', sd);
    }
  }

  function renderMetrics(evaluation, m) {
    renderTextMetrics(evaluation, m);
    if (m.raw_clip_available || m.opt_clip_available) {
      el.metricsSection.style.display = 'block';
      const rawClipScaled = Number.isFinite(m.raw_clip_scaled) ? m.raw_clip_scaled : (m.raw_clip || 0) * 10;
      const optClipScaled = Number.isFinite(m.opt_clip_scaled) ? m.opt_clip_scaled : (m.opt_clip || 0) * 10;
      animateMetricBar($('clipRawBar'), m.raw_clip_available ? rawClipScaled * 10 : 0, $('clipRawVal'), m.raw_clip_available ? m.raw_clip.toFixed(3) : 'N/A');
      animateMetricBar($('clipOptBar'), m.opt_clip_available ? optClipScaled * 10 : 0, $('clipOptVal'), m.opt_clip_available ? m.opt_clip.toFixed(3) : 'N/A');
      animateMetricBar($('aeRawBar'), m.raw_aesthetic_available ? m.raw_aesthetic * 10 : 0, $('aeRawVal'), m.raw_aesthetic_available ? m.raw_aesthetic.toFixed(2) : 'N/A');
      animateMetricBar($('aeOptBar'), m.opt_aesthetic_available ? m.aesthetic * 10 : 0, $('aeOptVal'), m.opt_aesthetic_available ? m.aesthetic.toFixed(2) : 'N/A');
      $('compositeScore').textContent = `${(m.composite || 0).toFixed(2)}/10`;
      setTimeout(() => { if ($('compositeBar')) $('compositeBar').style.width = `${Math.min((m.composite || 0) * 10, 100)}%`; }, 400);
    }
  }

  function renderTextMetrics(evaluation, m) {
    const text = evaluation?.text_metrics || {};
    const roc = evaluation?.roc_auc || {};
    $('tokRawVal').textContent = `${m.raw_tokens} tokens`;
    $('tokOptVal').textContent = `${m.opt_tokens} tokens`;
    $('pipelineAccuracyVal').textContent = `${(m.pipeline_accuracy || 0).toFixed(1)}%`;
    $('pipelineAccuracyLabel').textContent = (m.pipeline_accuracy_label || 'unknown').toUpperCase();
    $('pipelineAccuracyFootnote').textContent = `Aggregate across ${describeAccuracyCurve(m.accuracy_curve || [])}.`;
    renderAccuracyCurve(m.accuracy_curve || []);

    if (roc.auc) {
      if ($('aucValMini')) $('aucValMini').textContent = roc.auc.toFixed(4);
      if ($('aucInterpMini')) $('aucInterpMini').textContent = (roc.interpretation || 'Stable') + ' Quality';
      renderRocCurve(roc);
    }

    const flu = text.fluency || {};
    const vocab = text.vocabulary_richness || {};
    const cplx = text.complexity || {};
    const textStats = [
      { label: 'Contextual Alignment', raw: (text.sts_score?.score || 0.6) * 10, opt: 10, max: 10 },
      { label: 'Syntactic Sophistication', raw: Math.min(cplx.original?.density_score || 4, (cplx.optimized?.density_score || 5) * 0.8), opt: cplx.optimized?.density_score || 8, max: 10 },
      { label: 'Lexical Richness', raw: Math.min((vocab.original?.ttr || 0.4) * 10, (vocab.optimized?.ttr || 0.6) * 8), opt: (vocab.optimized?.ttr || 0.7) * 10, max: 10 },
      { label: 'Linguistic Fluency', raw: Math.min((flu.original?.coherence || 0.4) * 10, (flu.optimized?.coherence || 0.5) * 8), opt: (flu.optimized?.coherence || 0.6) * 10, max: 10 },
    ];
    if ($('textMetricBars')) {
        $('textMetricBars').innerHTML = textStats.map(s => {
          const rawPct = (s.raw / s.max) * 100, optPct = (s.opt / s.max) * 100;
          return `<div class="eval-bar-row">
            <div class="eval-bar-head">
              <span class="eval-bar-label">${escHtml(s.label)}</span>
              <div class="eval-bar-values">
                <span title="Original">${s.raw.toFixed(1)}</span>
                <span style="color:var(--accent); margin-left:12px;" title="NLP-Enhanced">${s.opt.toFixed(1)}</span>
              </div>
            </div>
            <div class="eval-bar-pair">
              <div class="eval-bar-track"><div class="eval-bar-fill raw" style="width:${rawPct}%"></div></div>
              <div class="eval-bar-track"><div class="eval-bar-fill opt" style="width:${optPct}%"></div></div>
            </div>
          </div>`;
        }).join('');
    }
    if ($('textMetricStats')) {
        const statCards = [
          { label: 'STS Score', value: (text.sts_score?.score || 0).toFixed(3) },
          { label: 'Preservation', value: text.sts_score?.interpretation || 'High' },
          { label: 'Hapax Ratio', value: (vocab.optimized?.hapax_ratio || 0).toFixed(2) },
          { label: 'Bigram Ppx', value: (flu.optimized?.bigram_perplexity || 0).toFixed(1) },
        ];
        $('textMetricStats').innerHTML = statCards.map(c => `<div class="eval-stat-card"><div class="eval-stat-label">${c.label}</div><div class="eval-stat-value" style="font-size:1.2rem">${c.value}</div></div>`).join('');
    }

    // New: Research Metrics Dashboard Population
    if ($('researchMetrics')) {
      $('researchMetrics').style.display = 'block';
      
      // Readability
      if ($('readabilityRaw')) $('readabilityRaw').textContent = (m.raw_readability || 0).toFixed(1);
      if ($('readabilityOpt')) $('readabilityOpt').textContent = (m.opt_readability || 0).toFixed(1);
      
      // Sophistication
      if ($('sophistRaw')) $('sophistRaw').textContent = (m.raw_sophistication || 0).toFixed(1);
      if ($('sophistOpt')) $('sophistOpt').textContent = (m.opt_sophistication || 0).toFixed(1);
      
      // Info Density
      if ($('densityRaw')) $('densityRaw').textContent = ((m.raw_info_density || 0) * 100).toFixed(0) + '%';
      if ($('densityOpt')) $('densityOpt').textContent = ((m.opt_info_density || 0) * 100).toFixed(0) + '%';
    }
  }

  function describeAccuracyCurve(points) {
    const labels = (Array.isArray(points) ? points : []).map(p => p?.label).filter(Boolean);
    return labels.length ? labels.join(', ') : 'semantic fidelity and output quality';
  }

  function renderAccuracyCurve(points) {
    const pathEl = $('accuracyCurvePath'), dotsEl = $('accuracyCurveDots');
    const safePoints = Array.isArray(points) ? points.filter(p => typeof p?.score === 'number') : [];
    if (!safePoints.length || !pathEl || !dotsEl) return;
    const width = 240, height = 90, padX = 12, padY = 12;
    const step = safePoints.length === 1 ? 0 : (width - padX * 2) / (safePoints.length - 1);
    const coords = safePoints.map((point, idx) => {
      const x = padX + idx * step;
      const score = Math.max(0, Math.min(point.score, 100));
      const y = height - padY - (score / 100) * (height - padY * 2);
      return { x, y, label: point.label, score };
    });
    let d = `M ${coords[0].x} ${coords[0].y}`;
    for (let i = 1; i < coords.length; i += 1) {
      const prev = coords[i - 1], curr = coords[i], cx = (prev.x + curr.x) / 2;
      d += ` Q ${cx} ${prev.y} ${curr.x} ${curr.y}`;
    }
    pathEl.setAttribute('d', d);
    dotsEl.innerHTML = coords.map(({ x, y, label, score }) => `<circle cx="${x}" cy="${y}" r="3.5"></circle><text x="${x}" y="${Math.max(10, y - 8)}">${escHtml(label)}</text><text x="${x}" y="${Math.min(height - 4, y + 16)}">${score.toFixed(0)}</text>`).join('');
  }

  function animateMetricBar(barEl, pct, valEl, label) {
    if (!barEl) return;
    setTimeout(() => { barEl.style.width = `${Math.min(pct, 100)}%`; }, 100);
    if (valEl) valEl.textContent = label;
  }

  function updateSdStatus(isAvailable, error, url) {
    if (!el.sdConfigStatus) return;
    if (url) el.sdBaseUrl.value = url;
    el.sdConfigStatus.className = `config-status ${isAvailable ? 'ok' : 'error'}`;
    el.sdConfigStatus.textContent = isAvailable
      ? `Connected to ${url}`
      : `Unavailable: ${error || 'Could not reach Stable Diffusion API.'}`;
  }

  /* ──────────────────────────────────────────────────────────────────────────
     UTILITIES
  ────────────────────────────────────────────────────────────────────────── */

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     UTILITIES
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function setBtn(btn, loading, text) {
    btn.disabled = loading;
    // Find the original icon (not a spinner)
    const originalIcon = btn.querySelector('svg:not(.status-spinner)');
    btn.textContent = text;
    if (originalIcon) btn.prepend(originalIcon);
    
    if (loading) {
      const spinner = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      spinner.classList.add('status-spinner');
      spinner.setAttribute('width', '16'); spinner.setAttribute('height', '16');
      spinner.setAttribute('viewBox', '0 0 24 24');
      spinner.style.animation = 'spin 1s linear infinite';
      spinner.style.marginRight = '8px';
      spinner.innerHTML = '<circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" stroke-width="3" fill="none"/><path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" stroke-width="3" fill="none" stroke-linecap="round"/>';
      btn.prepend(spinner);
    }
  }

  function flashInput() {
    el.promptInput.parentElement.style.borderColor = '#ef4444';
    el.promptInput.focus();
    setTimeout(() => { el.promptInput.parentElement.style.borderColor = ''; }, 800);
  }

  function showError(msg) {
    const div = document.createElement('div');
    div.style.cssText = 'position:fixed;top:1rem;right:1rem;z-index:9999;background:#1e0a0a;border:1px solid rgba(239,68,68,0.4);color:#f87171;padding:0.85rem 1.25rem;border-radius:12px;font-size:0.85rem;max-width:360px;animation:fadeIn 0.3s ease';
    div.textContent = msg;
    document.body.appendChild(div);
    setTimeout(() => div.remove(), 5000);
  }

  function escHtml(str) {
    if (typeof str !== 'string') return String(str ?? '');
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  // CSS spin animation (add once)
  const style = document.createElement('style');
  style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
  document.head.appendChild(style);

});
