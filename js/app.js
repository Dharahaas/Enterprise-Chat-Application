/* ============================================================
   App Shell — Sidebar, Header, Auth Guard, API Client
   Server-validated sessions and shared backend API client
   ============================================================ */

const APP = {
  name: "Enterprise Knowledge Assistant",
  short: "EKA",
  version: "Knowledge workspace",
};

const CHAT_USER_SCOPE_KEY = "eka-chat-user-scope-v1";
const DEMO_CHAT_SCOPE_KEY = "eka-demo-chat-scope-v1";

function newChatScope() {
  return globalThis.crypto?.randomUUID
    ? globalThis.crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function escapeHTML(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ============================================================
// Backend API client — all chat and administration calls require the live backend.
// ============================================================

const API = (() => {
  // Try multiple possible backend URLs:
  // 1. Explicit override via ?api= query param
  // 2. The local API when the frontend is served on localhost
  // 3. Same origin (when served by FastAPI behind a shared origin)
  // 4. Same-machine backend fallback for local frontend development
  const override = new URLSearchParams(location.search).get('api');
  const isLoopbackHost = location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  const isStaticLocalFrontend = isLoopbackHost && ['5500', '5501'].includes(location.port);
  const localBackend = isLoopbackHost
    ? `http://${location.hostname}:${location.port === '5501' ? '8766' : '8000'}`
    : null;
  const candidates = [...new Set([
    override,
    ...(isStaticLocalFrontend ? [] : [location.origin]),  // Prefer the API proxied by the current origin.
    localBackend,
  ].filter(base => base !== null && base !== undefined))];

  let backendUrl = null;
  let detection = null;
  let csrfToken = null;

  async function detect() {
    if (backendUrl) return backendUrl;
    if (detection) return detection;

    detection = (async () => {
      for (const base of candidates) {
        const ctrl = new AbortController();
        const timeout = setTimeout(() => ctrl.abort(), 2000);
        try {
          const res = await fetch(`${base}/api/health`, { signal: ctrl.signal });
          if (res.ok) {
            const data = await res.json();
            if (data.status === 'ok') {
              backendUrl = base;
              console.info(`[EKA] Backend connected: ${base}`);
              return backendUrl;
            }
          }
        } catch (err) {
          if (err.name !== 'AbortError') {
            console.debug(`[EKA] Backend check failed for ${base || 'same origin'}.`);
          }
        } finally {
          clearTimeout(timeout);
        }
      }
      return null;
    })();
    try {
      return await detection;
    } finally {
      detection = null;
    }
  }

  async function request(path, options = {}) {
    const base = await detect();
    if (!base) throw new Error('The assistant service is unavailable. Check that the backend is running and try again.');
    const { skipCsrf, ...fetchOptions } = options;
    const headers = new Headers(options.headers || {});
    const method = (options.method || 'GET').toUpperCase();
    if (!(options.body instanceof FormData) && options.body && !headers.has('Content-Type')) {
      headers.set('Content-Type', 'application/json');
    }
    if (!skipCsrf && !['GET', 'HEAD', 'OPTIONS'].includes(method) && csrfToken) {
      headers.set('X-CSRF-Token', csrfToken);
    }
    const requestStartedAt = performance.now();
    const res = await fetch(`${base}${path}`, {
      ...fetchOptions,
      headers,
      credentials: 'include',
    });
    if (path === '/api/chat') {
      console.info('[EKA] Chat HTTP response', {
        requestId: res.headers.get('X-Request-ID'),
        status: res.status,
        networkWaitMs: Math.round(performance.now() - requestStartedAt),
      });
    }
    if (!res.ok) {
      let detail = '';
      try {
        const payload = await res.json();
        detail = typeof payload.detail === 'string' ? payload.detail : '';
      } catch (_) {
        // Keep provider and infrastructure response bodies out of the UI.
      }
      const error = new Error(
        res.status >= 500
          ? 'The service could not complete this request. Please try again.'
          : detail || `The request failed (${res.status}). Please try again.`,
      );
      error.status = res.status;
      throw error;
    }
    return res.status === 204 ? null : res.json();
  }

  return {
    detect,
    isLive: async () => !!(await detect()),
    baseUrl: () => backendUrl,

    health: () => request('/api/health'),
    authOptions: () => request('/api/auth/options'),
    login: async (username, password, role) => {
      const session = await request('/api/auth/login', {
        method: 'POST',
        body: JSON.stringify({ username, password, role }),
        skipCsrf: true,
      });
      const isSharedDemoIdentity = session.user.username === 'demo-admin@demo.invalid' ||
        session.user.username === 'demo-employee@demo.invalid';
      const chatScope = isSharedDemoIdentity ? newChatScope() : session.user.username;
      if (isSharedDemoIdentity) {
        localStorage.setItem(DEMO_CHAT_SCOPE_KEY, chatScope);
      } else {
        localStorage.removeItem(DEMO_CHAT_SCOPE_KEY);
      }
      localStorage.setItem(CHAT_USER_SCOPE_KEY, chatScope);
      csrfToken = session.csrf_token;
      return session.user;
    },
    session: async () => {
      const session = await request('/api/auth/session');
      csrfToken = session.csrf_token;
      appUser = session.user;
      if (
        (session.user.username === 'demo-admin@demo.invalid' ||
          session.user.username === 'demo-employee@demo.invalid') &&
        !localStorage.getItem(CHAT_USER_SCOPE_KEY)
      ) {
        const chatScope = newChatScope();
        localStorage.setItem(DEMO_CHAT_SCOPE_KEY, chatScope);
        localStorage.setItem(CHAT_USER_SCOPE_KEY, chatScope);
      }
      return session.user;
    },
    logout: async () => {
      try {
        await request('/api/auth/logout', { method: 'POST' });
      } finally {
        csrfToken = null;
        appUser = null;
        localStorage.removeItem(CHAT_USER_SCOPE_KEY);
        localStorage.removeItem(DEMO_CHAT_SCOPE_KEY);
      }
    },
    chat: (question, agent, conversationId, conversationHistory) =>
      request('/api/chat', {
        method: 'POST',
        body: JSON.stringify({
          question,
          agent,
          conversation_id: conversationId,
          conversation_history: conversationHistory,
        }),
      }),
    chatStream: async (question, agent, conversationId, conversationHistory, onToken, onStatus) => {
      if (typeof WebSocket === 'undefined') {
        return API.chat(question, agent, conversationId, conversationHistory);
      }
      const base = await detect();
      if (!base) throw new Error('The assistant service is unavailable. Check that the backend is running and try again.');
      const endpoint = new URL(`${base}/api/chat/ws`);
      endpoint.protocol = endpoint.protocol === 'https:' ? 'wss:' : 'ws:';
      return new Promise((resolve, reject) => {
        const requestId = globalThis.crypto?.randomUUID
          ? globalThis.crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
        const socket = new WebSocket(endpoint);
        let settled = false;
        const fail = message => {
          if (settled) return;
          settled = true;
          socket.close();
          reject(new Error(message));
        };
        socket.addEventListener('open', () => {
          socket.send(JSON.stringify({
            v: 1,
            type: 'chat',
            request_id: requestId,
            csrf_token: csrfToken,
            payload: {
              question,
              agent,
              conversation_id: conversationId,
              conversation_history: conversationHistory,
            },
          }));
        });
        socket.addEventListener('message', event => {
          let message;
          try { message = JSON.parse(event.data); } catch (_) { return fail('The assistant returned an invalid stream event.'); }
          if (message.v !== 1 || message.request_id !== requestId) return;
          if (message.type === 'status') {
            onStatus?.(message.payload || {});
          } else if (message.type === 'token') {
            onToken?.(message.payload?.text || '');
          } else if (message.type === 'complete') {
            settled = true;
            resolve({
              answer: message.payload.answer,
              agent: message.payload.agent,
              sources: message.payload.sources || [],
              latency_ms: message.payload.total_ms,
              llm_provider: message.payload.provider,
              llm_model: message.payload.model_id,
              conversation_id: message.payload.conversation_id,
              time_to_first_token_ms: message.payload.time_to_first_token_ms,
            });
            socket.close();
          } else if (message.type === 'error') {
            fail(message.error?.message || 'The assistant could not complete this request.');
          } else if (message.type === 'cancel') {
            fail('The response was cancelled.');
          }
        });
        socket.addEventListener('error', () => fail('The streaming connection failed. Please try again.'));
        socket.addEventListener('close', () => {
          if (!settled) fail('The stream ended before the response completed. Please retry.');
        });
      });
    },
    listDocuments: () => request('/api/documents'),
    uploadDocument: (file, agent) => {
      const fd = new FormData();
      fd.append('file', file);
      fd.append('agent', agent);
      return request('/api/documents/upload', { method: 'POST', body: fd });
    },
    deleteDocument: (agent, name) =>
      request(`/api/documents/${agent}/${encodeURIComponent(name)}`, { method: 'DELETE' }),
    analytics: () => request('/api/analytics'),
    seed: () => request('/api/seed', { method: 'POST' }),
    evalSummary: () => request('/api/eval/summary'),
    evalRun: () => request('/api/eval/run', { method: 'POST' }),
    evalLatest: () => request('/api/eval/latest'),
    metrics: async () => {
      const base = await detect();
      if (!base) throw new Error('The assistant service is unavailable.');
      const response = await fetch(`${base}/api/metrics`, { credentials: 'include' });
      if (!response.ok) throw new Error(`Metrics request failed (${response.status}).`);
      return response.text();
    },
  };
})();

const AGENTS = [
  {
    id: "hr",
    name: "HR Agent",
    desc: "Leaves, payroll, onboarding, benefits",
    initial: "HR",
    placeholder: "Ask about leaves, payroll, onboarding...",
    suggested: [
      { title: "Leave policy", sub: "How many casual leaves am I entitled to?" },
      { title: "Payroll cycle", sub: "When is the next payroll cut-off?" },
      { title: "Onboarding", sub: "What documents are needed for new hires?" },
      { title: "Benefits", sub: "What is included in the health plan?" },
    ],
  },
  {
    id: "it",
    name: "IT Agent",
    desc: "Access, VPN, hardware, software, tickets",
    initial: "IT",
    placeholder: "Ask about VPN, passwords, hardware, tickets...",
    suggested: [
      { title: "Reset password", sub: "How do I reset my corporate password?" },
      { title: "VPN setup", sub: "How to configure VPN on macOS?" },
      { title: "New laptop", sub: "Request a hardware upgrade" },
      { title: "Software install", sub: "Can I install Python on my machine?" },
    ],
  },
  {
    id: "policy",
    name: "Policy Agent",
    desc: "Code of conduct, compliance, security",
    initial: "PL",
    placeholder: "Ask about company policies and compliance...",
    suggested: [
      { title: "Data classification", sub: "What are the data classification levels?" },
      { title: "Remote work", sub: "Is hybrid work allowed in my role?" },
      { title: "Code of conduct", sub: "What is the gift acceptance policy?" },
      { title: "Information security", sub: "How do I report a phishing email?" },
    ],
  },
];

/* ---------- Server-validated auth ---------- */
let appUser = null;

async function requireAuth(requiredRole = null) {
  try {
    const user = await API.session();
    if (requiredRole === 'admin' && user.role !== 'admin') {
      window.location.replace('portal.html?access=denied');
      return null;
    }
    appUser = user;
    return user;
  } catch (error) {
    if (error.status !== 401) {
      console.error('[EKA] Could not validate the sign-in session:', error);
      window.location.replace('index.html?error=unavailable');
      return null;
    }
    window.location.replace('index.html');
    return null;
  }
}

function isDemoAdminUser(user) {
  return user?.role === 'admin' && user.username === 'demo-admin@demo.invalid';
}

function homePageForUser(user) {
  return 'portal.html';
}

async function logout() {
  try {
    await API.logout();
  } catch (error) {
    console.error('[EKA] Sign-out could not reach the server:', error);
    if (error.status !== 401) {
      window.alert(error.message || 'Sign-out could not be completed. Please try again.');
      return;
    }
  }
  window.location.replace('index.html');
}

/* ---------- Sidebar render ---------- */
function renderSidebar(activePage) {
  const user = appUser || {};
  const realAdmin = user.role === 'admin' && !isDemoAdminUser(user);
  const selectedAgent = new URLSearchParams(location.search).get('agent') || 'hr';
  const navItems = [
    { id: "portal", label: "Employee portal", href: "portal.html", icon: chartIcon() },
    { id: "chat", label: "Chat", href: "chat.html", icon: chatIcon() },
    ...(realAdmin ? [
      { id: "documents", label: "Documents", href: "documents.html", icon: docIcon() },
      { id: "analytics", label: "Analytics", href: "analytics.html", icon: chartIcon() },
    ] : []),
  ];

  return `
    <aside class="sidebar" id="appSidebar" aria-label="Primary navigation">
      <div class="sidebar-brand">
        <div class="brand-mark">${APP.short.charAt(0)}</div>
        <div>
          <div class="brand-name">${APP.name}</div>
          <div class="brand-sub">EMPLOYEE KNOWLEDGE PORTAL</div>
        </div>
      </div>
      <nav class="sidebar-nav">
        <div class="nav-section-title">Workspace</div>
        ${navItems.map(item => `
          <a href="${item.href}" class="nav-item ${item.id === activePage ? 'active' : ''}" ${item.id === activePage ? 'aria-current="page"' : ''}>
            ${item.icon}
            <span>${item.label}</span>
          </a>
        `).join('')}
        <div class="nav-section-title">Agents</div>
        ${AGENTS.map(a => `
          <a href="chat.html?agent=${a.id}" class="nav-item agent-nav-item ${activePage === 'chat' && selectedAgent === a.id ? 'active' : ''}" ${activePage === 'chat' && selectedAgent === a.id ? 'aria-current="page"' : ''}>
            <span class="agent-nav-mark ${a.id}" aria-hidden="true">${a.initial.slice(0, 1)}</span>
            <span>${a.name}</span>
          </a>
        `).join('')}
      </nav>
      <div class="sidebar-footer">
        <button class="user-chip" type="button" onclick="logout()" title="Sign out">
          <div class="user-avatar">${escapeHTML((user.username || 'U').charAt(0).toUpperCase())}</div>
          <div class="user-info">
            <div class="user-name">${escapeHTML(user.username || 'User')}</div>
            <div class="user-id">${escapeHTML(user.role || 'employee')} · Sign out</div>
          </div>
          <svg aria-hidden="true" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
            <polyline points="16 17 21 12 16 7"/>
            <line x1="21" y1="12" x2="9" y2="12"/>
          </svg>
        </button>
      </div>
    </aside>
  `;
}

function renderHeader(title, breadcrumb, actions = "") {
  return `
    <header class="header">
      <div class="header-title">
        <button class="mobile-nav-toggle" type="button" aria-label="Open navigation" aria-controls="appSidebar" aria-expanded="false" onclick="toggleSidebar()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>
        </button>
        <div>
          <h2>${title}</h2>
          <div class="header-breadcrumb">${breadcrumb}</div>
        </div>
      </div>
      <div class="header-actions">${actions}</div>
    </header>
  `;
}

/* ---------- Icons ---------- */
function chatIcon() {
  return `<svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
  </svg>`;
}
function docIcon() {
  return `<svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
    <line x1="16" y1="13" x2="8" y2="13"/>
    <line x1="16" y1="17" x2="8" y2="17"/>
  </svg>`;
}
function chartIcon() {
  return `<svg class="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <line x1="18" y1="20" x2="18" y2="10"/>
    <line x1="12" y1="20" x2="12" y2="4"/>
    <line x1="6" y1="20" x2="6" y2="14"/>
  </svg>`;
}

/* ---------- Mount shell ---------- */
async function mountShell(activePage, title, breadcrumb, mainContent, headerActions = "", requiredRole = null) {
  const user = await requireAuth(requiredRole);
  if (!user) return;

  document.body.innerHTML = `
    <div class="app-shell">
      <button class="sidebar-backdrop" type="button" aria-label="Close navigation" onclick="toggleSidebar(false)"></button>
      ${renderSidebar(activePage)}
      ${renderHeader(title, breadcrumb, headerActions)}
      <main class="main">
        ${mainContent}
      </main>
    </div>
  `;
  return true;
}

function toggleSidebar(forceOpen) {
  const open = typeof forceOpen === 'boolean'
    ? forceOpen
    : !document.body.classList.contains('sidebar-open');
  document.body.classList.toggle('sidebar-open', open);
  const toggle = document.querySelector('.mobile-nav-toggle');
  if (toggle) toggle.setAttribute('aria-expanded', String(open));
}

document.addEventListener('keydown', event => {
  if (event.key === 'Escape') toggleSidebar(false);
});
