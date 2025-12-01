document.addEventListener('DOMContentLoaded', function () {
  // Helper: navigate to server routes
  function go(path) { window.location.href = path; }

  // Map common link texts to routes
  const linkMap = {
    'home': '/Index.html',
    'predictions': '/Index.html',
    'standings': '/Standings.html',
    'leaderboard': '/Leaderboard.html',
    'teams': '/Teams.html',
    'user profile': '/user_profile',
    'profile': '/user_profile',
    'matches': '/Index.html'
  };

  document.querySelectorAll('a').forEach(a => {
    const txt = (a.textContent || '').trim().toLowerCase();
    if (linkMap[txt]) a.href = linkMap[txt];
  });

  // Wire buttons: generic handlers
  document.querySelectorAll('button').forEach(btn => {
    const txt = (btn.textContent || '').trim().toLowerCase();

    // Navigation buttons
    if (txt.includes('view full table') || txt === 'view full table') {
      btn.addEventListener('click', () => go('/Standings.html'));
    }
    if (txt === 'see details' || txt === 'details' || txt === 'view details') {
      btn.addEventListener('click', (e) => {
        // find team names from nearby text
        const card = btn.closest('div');
        let text = '';
        if (card) text = card.innerText || '';
        // try to find pattern 'Team A vs Team B'
        const vsMatch = text.match(/([A-Za-zÀ-ÖØ-öø-ÿ0-9 .'-]+)\s+vs\s+([A-Za-zÀ-ÖØ-öø-ÿ0-9 .'-]+)/i);
        if (vsMatch) {
          const home = vsMatch[1].trim();
          const away = vsMatch[2].trim();
          go(`/Match_detail.html?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
          return;
        }
        // fallback: open generic match detail
        go('/Match_detail.html');
      });
    }

    // Make Prediction / Place your prediction buttons
    if (txt.includes('make prediction') || txt.includes('place your prediction') || txt.includes('place your prediction')) {
      btn.addEventListener('click', (e) => {
        // try to extract teams like above and redirect to match detail
        const card = btn.closest('div');
        let text = '';
        if (card) text = card.innerText || '';
        const vsMatch = text.match(/([A-Za-zÀ-ÖØ-öø-ÿ0-9 .'-]+)\s+vs\s+([A-Za-zÀ-ÖØ-öø-ÿ0-9 .'-]+)/i);
        if (vsMatch) {
          const home = vsMatch[1].trim();
          const away = vsMatch[2].trim();
          go(`/Match_detail.html?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
          return;
        }
        // default action: open profile
        go('/user_profile');
      });
    }

    // Sign Up / Login buttons -> go to profile or trigger modal
    if (txt.includes('sign up') || txt.includes('signup') || txt.includes('sign-up')) {
      btn.addEventListener('click', () => go('/user_profile'));
    }
    if (txt.includes('login') || txt.includes('log in') || txt.includes('log-in')) {
      btn.addEventListener('click', () => go('/user_profile'));
    }

    // Post Comment buttons: append comment locally
    if (txt.includes('post comment') || txt.includes('post')) {
      btn.addEventListener('click', (e) => {
        const container = btn.closest('div');
        if (!container) return;
        const textarea = container.querySelector('textarea');
        if (!textarea) return;
        const text = textarea.value.trim();
        if (!text) return alert('Please enter a comment');
        const feed = document.querySelector('.space-y-5');
        if (!feed) return;
        const el = document.createElement('div');
        el.className = 'flex items-start gap-3';
        el.innerHTML = `<div class="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-9" style="background-image: url('/assets/logos/default.svg')"></div><div><div class="flex items-center gap-2"><p class="text-sm font-semibold text-white">You</p><p class="text-xs text-white/50">just now</p></div><p class="text-sm text-white/80 mt-1">${text}</p></div>`;
        feed.prepend(el);
        textarea.value = '';
      });
    }
  });

  // Standings page rendering
  async function loadStandingsIfPresent() {
    if (!window.location.pathname.toLowerCase().includes('standings')) return;
    const season = new URLSearchParams(window.location.search).get('season') || '2025-2026';
    const out = document.querySelector('tbody');
    if (!out) return;
    out.innerHTML = '<tr><td colspan="10" class="p-4 text-center">Loading standings...</td></tr>';
    try {
      const resp = await fetch(`/api/standings?season=${encodeURIComponent(season)}`);
      if (!resp.ok) {
        out.innerHTML = `<tr><td colspan="10" class="p-4 text-center">Standings not found for season ${season}. Use /api/standings/scrape to fetch.</td></tr>`;
        return;
      }
      const data = await resp.json();
      if (!Array.isArray(data) || data.length === 0) {
        out.innerHTML = `<tr><td colspan="10" class="p-4 text-center">No standings data available for ${season}.</td></tr>`;
        return;
      }
      const rows = data.map((r, i) => {
        const team = r.team || '';
        const p = r.played != null ? r.played : '';
        const w = r.wins != null ? r.wins : '';
        const d = r.draws != null ? r.draws : '';
        const l = r.losses != null ? r.losses : '';
        const gf = r.goals_for != null ? r.goals_for : '';
        const ga = r.goals_against != null ? r.goals_against : '';
        const gd = (r.goals_for != null && r.goals_against != null) ? (r.goals_for - r.goals_against) : '';
        const pts = r.points != null ? r.points : '';
        const img = `<img class="w-6 h-6" src="/assets/logos/${team.replace(/\s+/g, '-').toLowerCase()}.svg" onerror="this.onerror=null;this.src='/assets/logos/default.svg'"/>`;
        return `\n            <tr class="hover:bg-gray-100 dark:hover:bg-white/5">\n              <td class="h-[60px] px-4 py-2 text-gray-800 dark:text-gray-300 text-sm font-medium leading-normal">${i+1}</td>\n              <td class="h-[60px] px-4 py-2 text-gray-900 dark:text-white text-sm font-medium leading-normal flex items-center gap-3">${img}${team}</td>\n              <td class="h-[60px] px-4 py-2 text-center text-gray-700">${p}</td>\n              <td class="h-[60px] px-4 py-2 text-center text-gray-700">${w}</td>\n              <td class="h-[60px] px-4 py-2 text-center text-gray-700">${d}</td>\n              <td class="h-[60px] px-4 py-2 text-center text-gray-700">${l}</td>\n              <td class="hidden sm:table-cell h-[60px] px-4 py-2 text-center text-gray-700">${gf}</td>\n              <td class="hidden sm:table-cell h-[60px] px-4 py-2 text-center text-gray-700">${ga}</td>\n              <td class="hidden sm:table-cell h-[60px] px-4 py-2 text-center text-gray-700">${gd}</td>\n              <td class="h-[60px] px-4 py-2 text-center text-gray-900 text-sm font-bold leading-normal">${pts}</td>\n            </tr>`;
      }).join('\n');
      out.innerHTML = rows;
    } catch (e) {
      out.innerHTML = `<tr><td colspan="10" class="p-4 text-center">Error loading standings: ${e.message}</td></tr>`;
    }
  }

  loadStandingsIfPresent();

  // Leaderboard: connect to WebSocket and update table/podium if present
  function initLeaderboardWebSocket() {
    if (!window.location.pathname.toLowerCase().includes('leaderboard')) return;
    const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
    const wsUrl = `${proto}://${location.host}/ws/leaderboard`;
    try {
      const ws = new WebSocket(wsUrl);
      ws.addEventListener('open', () => console.log('Leaderboard WS connected'));
      ws.addEventListener('message', (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === 'leaderboard_update' && Array.isArray(msg.data)) {
            renderLeaderboard(msg.data);
          }
        } catch (e) { console.warn('Invalid WS message', e); }
      });
      ws.addEventListener('close', () => console.log('Leaderboard WS closed'));
    } catch (e) {
      console.warn('WS not available', e);
    }
  }

  function renderLeaderboard(data) {
    // Podium selectors
    const podiumNames = [data[0]?.username || '', data[1]?.username || '', data[2]?.username || ''];
    const podiumPoints = [data[0]?.total_points || '', data[1]?.total_points || '', data[2]?.total_points || ''];
    // Update podium cards if present
    document.querySelectorAll('[data-podium-name]').forEach((el, i) => { if (data[i]) el.textContent = data[i].username; });
    document.querySelectorAll('[data-podium-points]').forEach((el, i) => { if (data[i]) el.textContent = data[i].total_points; });

    // Update table body
    const tbody = document.querySelector('tbody');
    if (!tbody) return;
    // keep header row if sticky by checking first child
    const rows = data.map((r, idx) => {
      return `\n<tr class="border-b border-black/10 dark:border-white/10">\n<td class="p-4 font-medium text-black/60 dark:text-white/60">${r.position}</td>\n<td class="p-4"><div class="flex items-center gap-3"><div class="bg-center bg-no-repeat aspect-square bg-cover rounded-full size-10" style="background-image: url('/assets/logos/${r.username.replace(/\s+/g,'-').toLowerCase()}.svg')"></div><span class="font-medium text-black dark:text-white">${r.username}</span></div></td>\n<td class="p-4 font-medium text-black dark:text-white text-right">${r.total_points}</td>\n<td class="p-4 text-black/80 dark:text-white/80 text-center hidden sm:table-cell">${r.accuracy || ''}%</td>\n<td class="p-4 text-center hidden sm:table-cell">${r.change || ''}</td>\n</tr>`;
    }).join('\n');
    tbody.innerHTML = rows;
  }

  initLeaderboardWebSocket();

  // Auth forms handling on user_profile page
  function initAuthForms() {
    const authSection = document.getElementById('auth-section');
    if (!authSection) return;
    const tabLogin = document.getElementById('auth-tab-login');
    const tabReg = document.getElementById('auth-tab-register');
    const loginForm = document.getElementById('login-form');
    const regForm = document.getElementById('register-form');
    const loginMsg = document.getElementById('login-msg');
    const regMsg = document.getElementById('reg-msg');

    function showLogin() { loginForm.classList.remove('hidden'); regForm.classList.add('hidden'); tabLogin.classList.add('bg-primary','text-white'); tabReg.classList.remove('bg-primary','text-white'); }
    function showReg() { regForm.classList.remove('hidden'); loginForm.classList.add('hidden'); tabReg.classList.add('bg-primary','text-white'); tabLogin.classList.remove('bg-primary','text-white'); }

    tabLogin.addEventListener('click', showLogin);
    tabReg.addEventListener('click', showReg);

    document.getElementById('login-submit').addEventListener('click', async () => {
      const username = document.getElementById('login-username').value.trim();
      const password = document.getElementById('login-password').value;
      loginMsg.textContent = '';
      if (!username || !password) { loginMsg.textContent = 'Enter username and password'; return; }
      try {
        const resp = await fetch('/api/auth/token', {method:'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({username, password})});
        if (!resp.ok) {
          const txt = await resp.text(); loginMsg.textContent = `Login failed: ${txt}`; return;
        }
        const data = await resp.json();
        localStorage.setItem('access_token', data.access_token || data.accessToken || '');
        loginMsg.textContent = 'Login successful';
        setTimeout(()=>{ location.reload(); }, 800);
      } catch (e) { loginMsg.textContent = 'Login error: '+e.message; }
    });

    document.getElementById('reg-submit').addEventListener('click', async () => {
      const username = document.getElementById('reg-username').value.trim();
      const email = document.getElementById('reg-email').value.trim();
      const password = document.getElementById('reg-password').value;
      regMsg.textContent = '';
      if (!username || !email || !password) { regMsg.textContent = 'Fill all fields'; return; }
      try {
        const resp = await fetch('/api/auth/register', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({username, email, password})});
        if (!resp.ok) { regMsg.textContent = 'Register failed'; return; }
        const user = await resp.json();
        regMsg.textContent = 'Account created — signing in...';
        // auto-login
        const tokenResp = await fetch('/api/auth/token', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({username, password})});
        if (tokenResp.ok) { const tok = await tokenResp.json(); localStorage.setItem('access_token', tok.access_token || tok.accessToken || ''); setTimeout(()=>location.reload(),800); }
      } catch (e) { regMsg.textContent = 'Register error: '+e.message; }
    });
  }

  initAuthForms();

  // Match detail: fetch prediction if query params provided
  async function loadMatchPredictionIfPresent() {
    if (!window.location.pathname.toLowerCase().includes('match_detail')) return;
    const params = new URLSearchParams(window.location.search);
    const home = params.get('home');
    const away = params.get('away');
    const date = params.get('date') || '';
    // find the prediction card by header text
    let predictionCard = null;
    document.querySelectorAll('h3').forEach(h => { if ((h.textContent||'').toLowerCase().includes('match prediction')) predictionCard = h.parentElement; });
    if (!predictionCard) return;
    predictionCard.innerHTML = '<h3 class="text-lg font-bold text-white mb-4">Match Prediction</h3><div class="p-4 text-center text-white/70">Loading prediction...</div>';
    if (!home || !away) return;
    try {
      const resp = await fetch('/api/predict', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({home_team: home, away_team: away, date: date})
      });
      if (!resp.ok) {
        const txt = await resp.text();
        predictionCard.innerHTML = `<h3 class="text-lg font-bold text-white mb-4">Match Prediction</h3><div class="p-4 text-center text-white/70">Prediction not available: ${txt}</div>`;
        return;
      }
      const data = await resp.json();
      // render prediction
      const ph = data.pred_home_goals != null ? Number(data.pred_home_goals).toFixed(1) : '-';
      const pa = data.pred_away_goals != null ? Number(data.pred_away_goals).toFixed(1) : '-';
      const pr = data.pred_result != null ? data.pred_result : '-';
      predictionCard.innerHTML = `<h3 class="text-lg font-bold text-white mb-4">Match Prediction</h3><div class="grid grid-cols-3 gap-4"><div class="flex flex-col items-center p-3 rounded-lg bg-white/5"><p class="text-white/70 text-sm">${home} goals</p><p class="text-3xl font-bold text-white mt-1">${ph}</p></div><div class="flex flex-col items-center p-3 rounded-lg bg-primary/20 border-2 border-primary"><p class="text-white/70 text-sm">Result (model)</p><p class="text-3xl font-bold text-white mt-1">${pr}</p></div><div class="flex flex-col items-center p-3 rounded-lg bg-white/5"><p class="text-white/70 text-sm">${away} goals</p><p class="text-3xl font-bold text-white mt-1">${pa}</p></div></div>`;
    } catch (e) {
      predictionCard.innerHTML = `<h3 class="text-lg font-bold text-white mb-4">Match Prediction</h3><div class="p-4 text-center text-white/70">Error: ${e.message}</div>`;
    }
  }

  loadMatchPredictionIfPresent();
});
