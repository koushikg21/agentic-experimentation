const state = {
  records: [],
  winTallies: [],
  pointTallies: [],
  displayedGames: 0,
  timer: null,
  timelineTiles: [],
  metric: 'wins',
  playerALabel: 'Player A',
  playerBLabel: 'Player B',
};

const dashboardEl = document.querySelector('.dashboard');
const scoreSvg = document.getElementById('score-chart');
const timelineContainer = document.getElementById('timeline');
const startBtn = document.getElementById('start-btn');
const seriesLabel = document.getElementById('series-label');
const modelLabel = document.getElementById('model-label');
const statusEl = document.getElementById('status');
const finalSummaryEl = document.getElementById('final-summary');
const legendTextEl = document.getElementById('legend-text');
const metricButtons = document.querySelectorAll('[data-metric]');

startBtn.addEventListener('click', () => {
  startBtn.disabled = true;
  startBtn.textContent = 'Playing...';
  restartAnimation();
});
metricButtons.forEach((button) =>
  button.addEventListener('click', () => setMetric(button.dataset.metric))
);

init();

async function init() {
  try {
    const response = await fetch('../game_history.csv', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error('Unable to fetch game_history.csv');
    }
    const text = await response.text();
    const records = parseCsv(text);
    const latestSeries = pickLatestSeries(records);
    if (!latestSeries.length) {
      statusEl.textContent = 'No series detected. Ensure game_history.csv has data.';
      return;
    }
    state.records = latestSeries;
    state.playerALabel = latestSeries[0].modelA || 'Player A';
    state.playerBLabel = latestSeries[0].modelB || 'Player B';
    updateLegendText();
    state.winTallies = computeWinTallies(latestSeries);
    state.pointTallies = computePointTallies(latestSeries);
    state.metric = 'wins';
    renderTimeline(latestSeries);
    seriesLabel.textContent = `Series ${latestSeries[0].seriesId}`;
    modelLabel.textContent = `${latestSeries[0].modelA} (A) · ${latestSeries[0].modelB} (B)`;
    updateMetricButtons();
    statusEl.textContent = 'Ready to animate. Tap start when ready.';
  } catch (error) {
    console.error(error);
    statusEl.textContent = 'Failed to load CSV. Serve this folder via http (python -m http.server).';
  }
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length <= 1) return [];
  const records = [];
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    const cells = [];
    let current = '';
    let inQuotes = false;
    for (let j = 0; j < line.length; j++) {
      const char = line[j];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === ',' && !inQuotes) {
        cells.push(current);
        current = '';
      } else {
        current += char;
      }
    }
    cells.push(current);
    if (cells.length < 15) continue;
    records.push({
      timestamp: parseTimestamp(cells[0]),
      seriesId: cells[1],
      seriesGamesConfigured: Number.parseInt(cells[2], 10) || 0,
      modelA: cells[3],
      modelB: cells[4],
      gameNumber: Number.parseInt(cells[5], 10) || 0,
      winner: cells[6],
      starter: cells[7],
      moves: Number.parseInt(cells[8], 10) || 0,
      aUsedMinimax: cells[9].trim().toLowerCase() === 'true',
      bUsedMinimax: cells[10].trim().toLowerCase() === 'true',
      aTokens: Number.parseInt(cells[11], 10) || 0,
      bTokens: Number.parseInt(cells[12], 10) || 0,
      aPoints: Number.parseInt(cells[13], 10) || 0,
      bPoints: Number.parseInt(cells[14], 10) || 0,
      incidents: cells[15] ?? '',
    });
  }
  return records;
}

function pickLatestSeries(records) {
  if (!records.length) return [];
  let latest = records[0];
  for (const record of records) {
    if (record.timestamp > latest.timestamp) {
      latest = record;
    }
  }
  const latestSeries = records.filter((r) => r.seriesId === latest.seriesId);
  return latestSeries.sort((a, b) => a.gameNumber - b.gameNumber);
}

function restartAnimation() {
  if (!state.records.length) return;
  clearInterval(state.timer);
  state.displayedGames = 0;
  exitFinalMode();
  hideFinalSummary();
  drawScoreChart();
  updateTimeline();
  updateStatus(-1);
  state.timer = setInterval(() => {
    if (state.displayedGames >= state.records.length) {
      clearInterval(state.timer);
      updateStatus(state.records.length - 1, true);
      startBtn.disabled = false;
      startBtn.textContent = 'Replay animation';
      renderFinalSummary();
      return;
    }
    state.displayedGames += 1;
    drawScoreChart();
    updateTimeline();
    updateStatus(state.displayedGames - 1);
  }, 220);
}

function drawScoreChart() {
  const totalWidth = 1020;
  const totalHeight = 420;
  const margin = { top: 36, right: 70, bottom: 60, left: 80 };
  const width = totalWidth - margin.left - margin.right;
  const height = totalHeight - margin.top - margin.bottom;
  scoreSvg.innerHTML = '';

  const background = createSvgElement('rect', {
    x: margin.left - 40,
    y: margin.top - 20,
    width: width + 80,
    height: height + 60,
    rx: 20,
    fill: '#111622',
  });
  scoreSvg.appendChild(background);

  const g = createSvgElement('g', { transform: `translate(${margin.left}, ${margin.top})` });
  scoreSvg.appendChild(g);

  const totalGames = state.records.length;
  if (!totalGames) return;
  const tallies = getTalliesForMetric(state.metric);
  const maxValue = Math.max(...tallies.map((t) => Math.max(t.a, t.b)), 1);
  const displayed = Math.max(state.displayedGames, 1);
  const showLabels = state.displayedGames >= state.records.length;
  const xStep = totalGames === 1 ? 0 : width / (totalGames - 1);
  const yScale = (value) => height - (value / maxValue) * height;

  const axis = createSvgElement('path', {
    d: `M0 ${height} H${width} M0 ${height} V0`,
    stroke: '#30363d',
    'stroke-width': 1.5,
  });
  g.appendChild(axis);

  const yLabel = createSvgElement('text', {
    x: -50,
    y: -10,
    fill: '#8b949e',
    'font-size': 12,
  });
  yLabel.textContent = state.metric === 'wins' ? 'Wins' : 'Points';
  g.appendChild(yLabel);

  const gridCount = 4;
  for (let i = 1; i <= gridCount; i++) {
    const y = (height / gridCount) * i;
    const gridLine = createSvgElement('line', {
      x1: 0,
      y1: y,
      x2: width,
      y2: y,
      stroke: 'rgba(255,255,255,0.05)',
    });
    g.appendChild(gridLine);
  }

  const minimaxLines = createSvgElement('g', {});
  const minimaxMarkers = createSvgElement('g', {});
  const labelA = state.playerALabel || 'Player A';
  const labelB = state.playerBLabel || 'Player B';
  for (let i = 0; i < displayed && i < state.records.length; i++) {
    const record = state.records[i];
    if (record.aUsedMinimax || record.bUsedMinimax) {
      const x = i * xStep;
      const stroke = record.aUsedMinimax && record.bUsedMinimax
        ? '#f4d675'
        : record.aUsedMinimax
          ? '#ff7b72'
          : '#539bf5';
      minimaxLines.appendChild(createSvgElement('line', {
        x1: x,
        y1: 0,
        x2: x,
        y2: height,
        stroke,
        opacity: 0.45,
        'stroke-width': 2.5,
        'stroke-dasharray': '6 6',
        'stroke-dashoffset': '2',
      }));
      if (record.aUsedMinimax) {
        minimaxMarkers.appendChild(drawMinimaxMarker(x, -14, '#ff7b72', labelA));
      }
      if (record.bUsedMinimax) {
        minimaxMarkers.appendChild(drawMinimaxMarker(x, -34, '#539bf5', labelB));
      }
    }
  }
  g.appendChild(minimaxLines);
  g.appendChild(minimaxMarkers);

  const pathA = buildPath(displayed, xStep, tallies, (record) => record?.a ?? 0, yScale);
  const pathB = buildPath(displayed, xStep, tallies, (record) => record?.b ?? 0, yScale);

  const pathNodeA = createSvgElement('path', {
    d: pathA,
    fill: 'none',
    stroke: '#ff7b72',
    'stroke-width': 3,
    'stroke-linecap': 'round',
  });
  g.appendChild(pathNodeA);

  const pathNodeB = createSvgElement('path', {
    d: pathB,
    fill: 'none',
    stroke: '#539bf5',
    'stroke-width': 3,
    'stroke-linecap': 'round',
  });
  g.appendChild(pathNodeB);

  if (showLabels && tallies.length) {
    const finalIndex = tallies.length - 1;
    const recordValues = tallies[finalIndex] ?? { a: 0, b: 0 };
    const x = finalIndex * xStep;
    const aY = yScale(recordValues.a);
    const bY = yScale(recordValues.b);
    g.appendChild(drawScoreLabel(x, aY, '#ff7b72', recordValues.a, 'below'));
    g.appendChild(drawScoreLabel(x, bY, '#539bf5', recordValues.b, 'above'));
  }

  const axisLabels = createSvgElement('text', {
    x: width / 2 - 20,
    y: height + 40,
    fill: '#8b949e',
    'font-size': 12,
  });
  axisLabels.textContent = 'Games';
  g.appendChild(axisLabels);
}

function buildPath(displayed, xStep, tallies, valueAccessor, yScale) {
  if (!displayed) return '';
  let d = '';
  for (let i = 0; i < displayed && i < tallies.length; i++) {
    const x = i * xStep;
    const value = valueAccessor(tallies[i], i);
    const y = yScale(value);
    d += i === 0 ? `M${x},${y}` : ` L${x},${y}`;
  }
  return d;
}

function drawScoreLabel(x, y, color, label, position = 'above') {
  const group = createSvgElement('g', { transform: `translate(${x}, ${y})` });
  const backing = createSvgElement('rect', {
    x: -18,
    y: position === 'above' ? -26 : 10,
    width: 36,
    height: 16,
    rx: 6,
    fill: 'rgba(0,0,0,0.45)',
    stroke: color,
    'stroke-width': 1,
  });
  const text = createSvgElement('text', {
    x: 0,
    y: position === 'above' ? -14 : 22,
    'font-size': 11,
    'text-anchor': 'middle',
    fill: '#e6edf3',
  });
  text.textContent = label;
  group.appendChild(backing);
  group.appendChild(text);
  return group;
}

function drawMinimaxMarker(x, y, color, label) {
  const group = createSvgElement('g', { transform: `translate(${x}, ${y})` });
  const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
  title.textContent = `${label} used minimax`;
  group.appendChild(title);
  group.appendChild(createMinimaxSymbol(color, 5));
  return group;
}

function createMinimaxSymbol(color, size = 6) {
  return createSvgElement('circle', {
    cx: 0,
    cy: 0,
    r: size,
    fill: color,
    stroke: '#0e1117',
    'stroke-width': 2,
  });
}

function createSvgElement(tag, props) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  Object.entries(props).forEach(([key, value]) => node.setAttribute(key, value));
  return node;
}

function updateLegendText() {
  if (!legendTextEl) return;
  const labelA = state.playerALabel || 'Player A';
  const labelB = state.playerBLabel || 'Player B';
  legendTextEl.innerHTML = `
    <span class="legend-pill pill-a">${labelA}</span> wins line
    &middot;
    <span class="legend-pill pill-b">${labelB}</span> wins line
    &middot;
    <span class="legend-helper" style="color:#ff7b72">
      ${labelA} helper
      <span class="dash"></span>
      <span class="dot"></span>
    </span>
    &middot;
    <span class="legend-helper" style="color:#539bf5">
      ${labelB} helper
      <span class="dash"></span>
      <span class="dot"></span>
    </span>
  `;
}

function renderTimeline(records) {
  timelineContainer.innerHTML = '';
  state.timelineTiles = records.map((record) => {
    const tile = document.createElement('div');
    tile.className = 'timeline-tile';

    const sparkA = document.createElement('div');
    sparkA.className = 'spark spark-a';
    tile.appendChild(sparkA);
    const sparkB = document.createElement('div');
    sparkB.className = 'spark spark-b';
    tile.appendChild(sparkB);

    const heading = document.createElement('h3');
    heading.textContent = `Game ${record.gameNumber}`;
    tile.appendChild(heading);

    const winner = document.createElement('p');
    winner.textContent = `Winner: ${formatPlayerCode(record.winner)}`;
    tile.appendChild(winner);

    const starter = document.createElement('p');
    starter.textContent = `Starter: ${formatPlayerCode(record.starter)}`;
    tile.appendChild(starter);

    const badges = document.createElement('div');
    const badgeA = createBadge(`${state.playerALabel || 'Player A'} minimax`, 'a');
    const badgeB = createBadge(`${state.playerBLabel || 'Player B'} minimax`, 'b');
    const badgeNone = createBadge('No minimax', 'none');
    badges.appendChild(badgeA);
    badges.appendChild(badgeB);
    badges.appendChild(badgeNone);
    tile.appendChild(badges);

    timelineContainer.appendChild(tile);
    return {
      tile,
      badgeA,
      badgeB,
      badgeNone,
      record,
    };
  });
}

function createBadge(text, variant) {
  const span = document.createElement('span');
  span.className = `badge ${variant}`;
  span.textContent = text;
  return span;
}

function updateTimeline() {
  state.timelineTiles.forEach((tileObj, index) => {
    const { tile, badgeA, badgeB, badgeNone, record } = tileObj;
    const active = index < state.displayedGames;
    tile.style.opacity = active ? '1' : '0.25';
    tile.classList.toggle('active-a', active && record.aUsedMinimax);
    tile.classList.toggle('active-b', active && record.bUsedMinimax);
    badgeA.style.display = active && record.aUsedMinimax ? 'inline-flex' : 'none';
    badgeB.style.display = active && record.bUsedMinimax ? 'inline-flex' : 'none';
    badgeNone.style.display = active && !record.aUsedMinimax && !record.bUsedMinimax ? 'inline-flex' : 'none';
  });
}

function setMetric(metric) {
  if (!metric || metric === state.metric) {
    return;
  }
  state.metric = metric;
  updateMetricButtons();
  drawScoreChart();
  if (state.displayedGames > 0) {
    updateStatus(state.displayedGames - 1);
  }
}

function updateMetricButtons() {
  metricButtons.forEach((button) => {
    button.classList.toggle('active', button.dataset.metric === state.metric);
  });
}

function getTalliesForMetric(metric) {
  if (metric === 'points') {
    if (state.pointTallies.length !== state.records.length) {
      state.pointTallies = computePointTallies(state.records);
    }
    return state.pointTallies;
  }
  if (state.winTallies.length !== state.records.length) {
    state.winTallies = computeWinTallies(state.records);
  }
  return state.winTallies;
}

function updateStatus(index, finished = false) {
  if (index < 0) {
    statusEl.textContent = 'Animating series...';
    return;
  }
  const game = state.records[index];
  const wins = state.winTallies[index] || { a: 0, b: 0 };
  const points = state.pointTallies[index] || { a: 0, b: 0 };
  const labelA = state.playerALabel || 'Player A';
  const labelB = state.playerBLabel || 'Player B';
  const winnerName = formatPlayerCode(game.winner);
  let text = `Game ${game.gameNumber}: winner ${winnerName} · Wins ${labelA} ${wins.a} - ${labelB} ${wins.b} · Score ${labelA} ${points.a} - ${labelB} ${points.b}`;
  if (finished) {
    text += ' · Animation complete. Tap replay to watch again.';
  }
  statusEl.textContent = text;
}

function hideFinalSummary() {
  finalSummaryEl.classList.add('hidden');
  finalSummaryEl.innerHTML = '';
  startBtn.disabled = false;
  startBtn.textContent = 'Start animation';
}

function renderFinalSummary() {
  if (!state.records.length) return;
  const lastIndex = state.records.length - 1;
  const wins = state.winTallies[lastIndex] || { a: 0, b: 0 };
  const points = state.pointTallies[lastIndex] || { a: 0, b: 0 };
  const labelA = state.playerALabel || 'Player A';
  const labelB = state.playerBLabel || 'Player B';
  const winnerLabel = determineSeriesWinner(wins, points, labelA, labelB);
  const winnerText = winnerLabel ? `${winnerLabel} wins the series` : 'Series tied';
  finalSummaryEl.innerHTML = `
    <span class="final-summary__badge">Series result</span>
    <div class="final-summary__winner">${winnerText}</div>
    <div class="final-summary__scoreboard">
      ${renderScoreBlock('Wins', wins, labelA, labelB)}
      ${renderScoreBlock('Score', points, labelA, labelB)}
    </div>
  `;
  finalSummaryEl.classList.remove('hidden');
}

function renderScoreBlock(title, values, labelA, labelB) {
  return `
    <div class="score-block">
      <h4>${title}</h4>
      <div class="value">${labelA} ${values.a} - ${values.b} ${labelB}</div>
      <p>Final ${title.toLowerCase()} total</p>
    </div>
  `;
}

function enterFinalMode() {
  dashboardEl.classList.add('final-mode');
}

function exitFinalMode() {
  dashboardEl.classList.remove('final-mode');
}

function parseTimestamp(raw) {
  if (!raw) return new Date(0);
  const normalized = raw.replace(/\.(\d{3})\d+/, '.$1');
  const parsed = new Date(normalized);
  return Number.isNaN(parsed.getTime()) ? new Date(0) : parsed;
}

function computeWinTallies(records) {
  const tallies = [];
  let aWins = 0;
  let bWins = 0;
  for (const record of records) {
    if (record.winner === 'A') {
      aWins += 1;
    } else if (record.winner === 'B') {
      bWins += 1;
    }
    tallies.push({ a: aWins, b: bWins });
  }
  return tallies;
}

function computePointTallies(records) {
  return records.map((record) => ({
    a: record.aPoints || 0,
    b: record.bPoints || 0,
  }));
}

function determineSeriesWinner(_wins, points, labelA, labelB) {
  if (points.a !== points.b) {
    return points.a > points.b ? labelA : labelB;
  }
  return null;
}

function formatPlayerCode(code) {
  if (code === 'A') return state.playerALabel || 'Player A';
  if (code === 'B') return state.playerBLabel || 'Player B';
  return code;
}
