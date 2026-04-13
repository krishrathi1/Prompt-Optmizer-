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
    // Tooltip
    tokenTooltip:   $('tokenTooltip'),
  };

  /* â”€â”€â”€ State â”€â”€â”€ */
  let session = { original: '', optimized: '', negative: '', settings: {}, pipelineStages: [] };

  /* â”€â”€â”€ Colour palette per pipeline step â”€â”€â”€ */
  const STEP_COLORS = [
    { hex: '#6366f1', rgb: '99,102,241' },
    { hex: '#8b5cf6', rgb: '139,92,246' },
    { hex: '#3b82f6', rgb: '59,130,246' },
    { hex: '#10b981', rgb: '16,185,129' },
    { hex: '#f59e0b', rgb: '245,158,11' },
    { hex: '#06b6d4', rgb: '6,182,212' },
    { hex: '#ec4899', rgb: '236,72,153' },
    { hex: '#ef4444', rgb: '239,68,68' },
  ];

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     HEALTH CHECK
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  async function healthCheck() {
    try {
      const r = await fetch('/api/health');
      if (r.ok) {
        const d = await r.json();
        el.statusDot.className = 'status-dot ok';
        el.statusLabel.textContent = d.clip_fallback
          ? 'Online (CLIP fallback mode)'
          : 'Online â€” all systems ready';
      } else throw new Error();
    } catch {
      el.statusDot.className = 'status-dot error';
      el.statusLabel.textContent = 'API unreachable';
    }
  }
  healthCheck();

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
        pipelineStages: data.pipeline_stages,
        variants:       data.variants || [],
        selectedModel:  'llama3.2'
      };

      renderAll(data, prompt);

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
    // Reset image section
    el.imageSection.style.display = 'none';
    el.metricsSection.style.display = 'none';
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

      case 3: // SVO Extraction
        if (!Array.isArray(d) || !d.length) return '<p class="empty-state">No semantic pathways found.</p>';
        return `<div class="svo-container">${(Array.isArray(d) ? d : []).map(s => `
          <div class="svo-card">
            <span class="svo-part subject">${escHtml(s.subject)}</span>
            <span class="svo-arrow">\u2014[ ${escHtml(s.action)} ]\u2192</span>
            <span class="svo-part object">${escHtml(s.object)}</span>
          </div>`).join('')}</div>`;

      case 4: // Keyword Ranking
        return `<div class="keyword-cloud">${Object.entries(d || {}).sort((a,b) => b[1]-a[1]).map(([w, s]) => 
          `<div class="keyword-pill" style="opacity:${0.4 + s * 0.6}; transform:scale(${0.9 + s * 0.2})">
            ${escHtml(w)} <small>${s.toFixed(2)}</small>
          </div>`).join('')}</div>`;

      case 5: // NP Chunking
        return `<div class="stage-data-tokens">${(Array.isArray(d) ? d : []).map(p =>
          `<span class="stage-token-pill" style="background:rgba(6,182,212,0.1);border-color:rgba(6,182,212,0.3)">${escHtml(p)}</span>`).join('')}</div>`;

      case 6: // Specificity
        return `<div class="ladder-container">${(Array.isArray(d) ? d : []).map(l => `
          <div class="ladder-item">
            <div class="ladder-word">${escHtml(l.word)}</div>
            <div class="ladder-path">${l.specificity.ladder.join(' \u2192 ')}</div>
            <div class="ladder-meta">${l.specificity.depth} levels deep | ${l.specificity.is_generic ? '\u26A0 Generic' : '\u2705 Specific'}</div>
          </div>`).join('')}</div>`;

      case 7: // Synonym Swapping
        return `<div>${(Array.isArray(d) ? d : []).map(t =>
          `<div class="stage-change-row"><span class="sc-from">${escHtml(t.word)}</span><span class="sc-arrow">\u2192</span><span class="sc-to">${escHtml(t.optimized_to)}</span></div>`
        ).join('')}</div>`;

      case 8: // Genetic Evolution
        return `
          <div class="ga-status">
            <div class="ga-metric"><strong>Final Fitness:</strong> ${d?.final_score?.toFixed(2) || 'â€”'}</div>
            <p style="font-size:0.75rem;color:var(--text-3);margin-top:0.4rem">Evolutionary algorithm optimized for semantic density over 3 generations.</p>
          </div>`;

      case 9: // Ollama Brainstorm
        if (!d) return '<p class="empty-state">Local LLM generation was bypassed or failed.</p>';
        return `
          <div class="ollama-result-box">
             <div style="font-size:0.65rem; color:#a855f7; font-weight:800; margin-bottom:0.5rem; text-transform:uppercase;">Expanded Description</div>
             <div style="font-family:'JetBrains Mono', monospace; font-size:0.8rem; line-height:1.6; color:var(--text-1)">${escHtml(d)}</div>
          </div>`;

      case 10: // Vibe Analysis
        return `
          <div style="display:flex;gap:1.5rem;font-size:0.8rem;font-family:'JetBrains Mono',monospace;flex-wrap:wrap">
            <span>pos <strong style="color:var(--accent)">${(d.scores.pos*100).toFixed(1)}%</strong></span>
            <span>neg <strong style="color:#f87171">${(d.scores.neg*100).toFixed(1)}%</strong></span>
            <span>neu <strong style="color:var(--text-2)">${(d.scores.neu*100).toFixed(1)}%</strong></span>
            <span>compound <strong style="color:${d.color}">${d.scores.compound.toFixed(3)}</strong></span>
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
        inner += `<span class="weight-badge">Ã—1.25</span>`;
        chip.title = 'Subject noun â€” weight boosted';
      }
      if (tok.changed) {
        const badge = document.createElement('div');
        badge.className = 'chip-badge';
        badge.textContent = 'âœ¦';
        chip.appendChild(badge);
      }
      chip.innerHTML = inner + (tok.is_subject ? `<span class="weight-badge">Ã—1.25</span>` : '');

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

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     GENETIC VARIANTS
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     NEGATIVE SHIELD
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function renderShield(negPrompt) {
    el.shieldTags.innerHTML = negPrompt.split(',')
      .map(t => t.trim()).filter(Boolean)
      .map(t => `<span class="shield-tag">${escHtml(t)}</span>`)
      .join('');
  }

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     FINAL PROMPTS + COPY
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     GENERATE IMAGES
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
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

      renderMetrics(data.metrics);

    } catch (err) {
      el.rawImageFrame.innerHTML = `<div style="padding:2rem;color:#f87171;font-size:0.8rem">Request failed: ${escHtml(err.message)}</div>`;
      el.optImageFrame.innerHTML = `<div style="padding:2rem;color:#f87171;font-size:0.8rem">Check that the server is running.</div>`;
    } finally {
      setBtn(el.generateBtn, false, 'Re-Generate');
    }
  });

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     METRICS DASHBOARD
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function renderMetrics(m) {
    el.metricsSection.style.display = 'block';

    // CLIP bars (score is already Ã—10 from server, max ~10)
    animateMetricBar($('clipRawBar'), m.raw_clip * 10, $('clipRawVal'), m.raw_clip.toFixed(3));
    animateMetricBar($('clipOptBar'), m.opt_clip * 10, $('clipOptVal'), m.opt_clip.toFixed(3));

    // Aesthetic (score 0-10)
    animateMetricBar($('aeRawBar'), m.raw_aesthetic * 10, $('aeRawVal'), m.raw_aesthetic.toFixed(2));
    animateMetricBar($('aeOptBar'), m.aesthetic * 10, $('aeOptVal'), m.aesthetic.toFixed(2));

    // Token count
    $('tokRawVal').textContent = `${m.raw_tokens} tokens`;
    $('tokOptVal').textContent = `${m.opt_tokens} tokens`;

    // Composite score
    const pct = Math.min(m.composite * 10, 100);
    $('compositeScore').textContent = `${m.composite.toFixed(2)}/10`;
    setTimeout(() => { $('compositeBar').style.width = `${pct}%`; }, 100);

    // Improvement banner
    const banner = $('improvementBanner');
    const clipDiff = m.opt_clip - m.raw_clip;
    if (clipDiff > 0.01) {
      banner.className = 'improvement-banner positive';
      banner.textContent = `â˜… CLIP score improved by +${clipDiff.toFixed(3)} â€” NLP enhancement measurably increased semantic alignment.`;
    } else {
      banner.className = 'improvement-banner neutral';
      banner.textContent = `Scores are comparable. The visual complexity and prompt richness are significantly higher in the enhanced version.`;
    }
    banner.style.display = 'block';
  }

  function animateMetricBar(barEl, pct, valEl, label) {
    setTimeout(() => { barEl.style.width = `${Math.min(pct, 100)}%`; }, 100);
    if (valEl) valEl.textContent = label;
  }

  /* â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
     UTILITIES
  â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â• */
  function setBtn(btn, loading, text) {
    btn.disabled = loading;
    // keep the SVG icon â€” just swap text node
    const svg = btn.querySelector('svg');
    btn.textContent = text;
    if (svg) btn.prepend(svg);
    if (loading && btn === el.optimizeBtn) {
      const spinner = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      spinner.setAttribute('width', '16'); spinner.setAttribute('height', '16');
      spinner.setAttribute('viewBox', '0 0 24 24');
      spinner.style.animation = 'spin 1s linear infinite';
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

