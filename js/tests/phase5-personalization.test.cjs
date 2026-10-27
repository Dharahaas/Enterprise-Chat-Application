const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const read = relative => fs.readFileSync(path.join(root, relative), 'utf8');

const appSource = read('js/app.js');
const chatSource = read('chat.html');
const portalSource = read('portal.html');
const indexSource = read('index.html');
const documentsSource = read('documents.html');
const cssSource = read('css/pages.css');

function extractFunction(source, name) {
  const start = source.indexOf(`function ${name}(`);
  assert.ok(start >= 0, `expected to find function ${name}`);
  const open = source.indexOf('{', start);
  assert.ok(open >= 0, `expected a body for ${name}`);
  // Brace-count to the matching close; template-literal `${}` pairs stay balanced.
  let depth = 0;
  for (let i = open; i < source.length; i += 1) {
    if (source[i] === '{') depth += 1;
    else if (source[i] === '}') {
      depth -= 1;
      if (depth === 0) return source.slice(start, i + 1);
    }
  }
  throw new Error(`unterminated body for ${name}`);
}

function extractConst(source, name) {
  const match = source.match(new RegExp(`const ${name} = [^;]+;`));
  assert.ok(match, `expected to find const ${name}`);
  return match[0];
}

function evaluate(source, names, globals = {}, consts = []) {
  const context = vm.createContext({ Array, String, Boolean, Date, Number, Math, ...globals });
  const body = [
    ...consts.map(name => extractConst(source, name)),
    ...names.map(name => extractFunction(source, name)),
  ].join('\n');
  vm.runInContext(body, context);
  return context;
}

// ---------- 1. Personalized greeting ----------

function greetingContext() {
  return evaluate(
    appSource,
    ['isDemoAdminUser', 'greetingDisplayName', 'timeOfDayGreeting', 'personalizedGreeting'],
  );
}

test('the greeting uses the session display name', () => {
  const context = greetingContext();
  assert.equal(
    context.greetingDisplayName({ username: 'dharahaas@example.test', role: 'employee' }),
    'dharahaas',
  );
  assert.equal(
    context.greetingDisplayName({ username: 'dharahaas', role: 'employee' }),
    'dharahaas',
  );
  assert.equal(context.personalizedGreeting(
    { username: 'employee@example.test', role: 'employee' },
    new Date('2026-06-15T08:30:00'),
  ), 'Good morning, employee');
});

test('the greeting is time-appropriate and falls back safely', () => {
  const context = greetingContext();
  assert.equal(context.timeOfDayGreeting(new Date('2026-06-15T08:00:00')), 'Good morning');
  assert.equal(context.timeOfDayGreeting(new Date('2026-06-15T12:00:00')), 'Good afternoon');
  assert.equal(context.timeOfDayGreeting(new Date('2026-06-15T16:59:00')), 'Good afternoon');
  assert.equal(context.timeOfDayGreeting(new Date('2026-06-15T17:00:00')), 'Good evening');
  assert.equal(context.timeOfDayGreeting(new Date('2026-06-15T23:30:00')), 'Good evening');
  // Unavailable clock or time zone → safe generic fallback.
  assert.equal(context.timeOfDayGreeting(new Date('not-a-date')), 'Good day');
  class BrokenDate extends Date {
    getHours() { throw new Error('time zone unavailable'); }
  }
  assert.equal(context.timeOfDayGreeting(new BrokenDate()), 'Good day');
});

test('shared and missing identities are never personalized', () => {
  const context = greetingContext();
  assert.equal(context.greetingDisplayName(null), '');
  assert.equal(context.greetingDisplayName({}), '');
  assert.equal(context.greetingDisplayName({ username: '   ' }), '');
  assert.equal(
    context.greetingDisplayName({ username: 'demo-admin@demo.invalid', role: 'admin' }),
    '',
  );
  assert.equal(
    context.greetingDisplayName({ username: 'demo-employee@demo.invalid', role: 'employee' }),
    '',
  );
  // Without a usable name the time-of-day is kept, but no personal data shows.
  assert.equal(
    context.personalizedGreeting({ username: 'demo-employee@demo.invalid', role: 'employee' },
      new Date('2026-06-15T19:00:00')),
    'Good evening.',
  );
  assert.equal(context.personalizedGreeting(null, new Date('2026-06-15T08:00:00')), 'Good morning.');
});

test('the portal and the assistant both use the personalized greeting', () => {
  assert.match(portalSource, /<h1 id="portalHeading">\$\{escapeHTML\(personalizedGreeting\(user\)\)\}<\/h1>/);
  assert.ok(!portalSource.includes('Good to have you here.'));
  assert.match(chatSource, /<h2 class="welcome-title">\$\{escapeHTML\(personalizedGreeting\(appUser\)\)\}<\/h2>/);
  assert.ok(!chatSource.includes('How can we help you today?'));
});

// ---------- 2. Recent questions and continuity ----------

function recentContext(conversations = []) {
  return evaluate(
    chatSource,
    ['recentQuestions', 'recentQuestionAgentLabel'],
    {
      conversations,
      timeAgo: value => `ago:${value}`,
      AGENTS: [
        { id: 'hr', name: 'HR Specialist' },
        { id: 'it', name: 'IT Specialist' },
        { id: 'policy', name: 'Policy Specialist' },
      ],
    },
  );
}

const conversation = (id, question, updatedAt, agent = 'hr') => ({
  id,
  agent,
  updatedAt,
  messages: question
    ? [{ role: 'user', content: question }, { role: 'agent', content: 'Answer' }]
    : [],
});

test('recent questions come from the existing conversation store, newest first', () => {
  const context = recentContext([
    conversation('a', 'Old question', '2026-10-01T10:00:00'),
    conversation('b', 'Newest question', '2026-10-27T10:00:00', 'it'),
    conversation('c', 'Middle question', '2026-10-15T10:00:00', 'policy'),
  ]);

  const recent = context.recentQuestions();
  // Arrays created inside the VM realm are compared by value, not prototype.
  assert.deepEqual(Array.from(recent, item => item.question), [
    'Newest question', 'Middle question', 'Old question',
  ]);
  assert.equal(recent[0].agent, 'it');
  assert.equal(recent[0].time, 'ago:2026-10-27T10:00:00');
  assert.equal(context.recentQuestionAgentLabel('it'), 'IT Specialist');
  assert.equal(context.recentQuestionAgentLabel(null), 'Ask Cathy');
});

test('recent questions are bounded, skip empty conversations and handle no history', () => {
  const many = Array.from({ length: 8 }, (_, index) =>
    conversation(`c${index}`, `Question ${index}`, `2026-10-0${(index % 9) + 1}T10:00:00`));
  const context = recentContext(many);
  assert.equal(context.recentQuestions().length, 5);
  assert.equal(context.recentQuestions(2).length, 2);

  const withEmpty = recentContext([
    conversation('empty', null, '2026-10-27T10:00:00'),
    conversation('real', 'A real question', '2026-10-26T10:00:00'),
  ]);
  const entries = withEmpty.recentQuestions();
  assert.deepEqual(Array.from(entries, item => item.id), ['real']);

  assert.equal(recentContext([]).recentQuestions().length, 0);
});

test('the assistant shows recent questions with an empty state and a new-conversation action', () => {
  for (const marker of [
    '<h3 id="recentQuestionsHeading">Recent questions</h3>',
    '<section class="recent-questions" aria-labelledby="recentQuestionsHeading">',
    'data-start-new>New conversation<',
    'No recent questions yet — ask Cathy anything',
    'data-recent-conversation="${escapeHtml(item.id)}"',
    'aria-labelledby="recentQuestionsHeading"',
  ]) {
    assert.ok(chatSource.includes(marker), `missing recent-questions marker: ${marker}`);
  }
  for (const wiring of [
    "event.target.closest('[data-recent-conversation]')",
    'openConversation(recentButton.dataset.recentConversation)',
    "event.target.closest('[data-start-new]')",
    'startNewConversation();',
  ]) {
    assert.ok(chatSource.includes(wiring), `missing wiring: ${wiring}`);
  }
  // Resuming reuses the existing conversation loader (routing + follow-ups).
  assert.match(chatSource, /function openConversation\(id\) \{/);
  assert.match(chatSource, /activeAgent = AGENTS\.find\(agent => agent\.id === conversation\.agent\) \|\| null;/);
});

test('conversation history stays scoped to the signed-in user', () => {
  const storage = () => {
    const values = {};
    return {
      getItem: key => (key in values ? values[key] : null),
      setItem: (key, value) => { values[key] = String(value); },
    };
  };
  const context = evaluate(
    chatSource,
    ['chatStorageKey'],
    {
      CHAT_USER_SCOPE_KEY: 'eka-chat-scope',
      localStorage: storage(),
      appUser: { username: 'alice@example.test', role: 'employee' },
    },
  );

  const aliceKey = context.chatStorageKey();
  assert.ok(aliceKey.includes('alice@example.test'));
  assert.ok(aliceKey.includes('employee'));

  // A different account resolves to a different storage key.
  context.appUser = { username: 'bob@example.test', role: 'employee' };
  const bobKey = context.chatStorageKey();
  assert.notEqual(aliceKey, bobKey);

  // ...and so does a different role for the same name.
  context.appUser = { username: 'alice@example.test', role: 'admin' };
  assert.notEqual(context.chatStorageKey(), aliceKey);

  // A missing profile falls back to a non-throwing, non-matching key.
  context.appUser = null;
  assert.equal(typeof context.chatStorageKey(), 'string');
});

// ---------- 3. What's new since your last visit ----------

function whatsNewContext(globals = {}) {
  return evaluate(
    portalSource,
    ['collectAnnouncements', 'readLastVisit', 'recordVisit', 'formatVisitDate',
      'announcementLine', 'buildWhatsNewQuestion'],
    globals,
    ['PORTAL_LAST_VISIT_KEY', 'WHATS_NEW_LIMIT'],
  );
}

const announcementRoot = titles => ({
  querySelectorAll: () => titles.map(title => ({
    querySelector: selector => {
      if (selector === 'h3') return title ? { textContent: title } : null;
      if (selector === 'time[datetime]') return null;
      if (selector === 'a[href]') return null;
      return null;
    },
  })),
});

test('announcements are read from the rendered portal and bounded', () => {
  const context = whatsNewContext();
  const found = context.collectAnnouncements(
    announcementRoot(['First announcement', 'Second announcement', 'Third announcement', 'Fourth announcement']),
  );
  assert.equal(found.length, 3, 'bounded set only');
  assert.deepEqual(Array.from(found, item => item.title), [
    'First announcement', 'Second announcement', 'Third announcement',
  ]);
  // Cards without a real title are never summarised.
  const withBlank = context.collectAnnouncements(announcementRoot([null, 'Only real one']));
  assert.deepEqual(Array.from(withBlank, item => item.title), ['Only real one']);
  // No dates or links are invented when the announcement has none.
  assert.equal(found[0].date, null);
  assert.equal(found[0].href, null);
});

test('the what\'s-new question summarises only what exists', () => {
  const context = whatsNewContext();
  const announcements = [
    { title: 'Welcome to the refreshed employee hub', date: null, href: null },
    { title: 'Explore the team resource library', date: '2026-10-20', href: '/news/resources' },
  ];

  const withHistory = context.buildWhatsNewQuestion(announcements, '26 Oct 2026');
  assert.ok(withHistory.startsWith('What has changed since my last visit on 26 Oct 2026?'));
  assert.ok(withHistory.includes('1. Welcome to the refreshed employee hub'));
  assert.ok(withHistory.includes('2. Explore the team resource library (2026-10-20) — /news/resources'));
  // Dates and links appear only when the announcement really carries them.
  assert.ok(!withHistory.includes('1. Welcome to the refreshed employee hub ('));

  const withoutHistory = context.buildWhatsNewQuestion(announcements, null);
  assert.ok(withoutHistory.includes("I don't have a record of my previous visit"));
  assert.ok(withoutHistory.includes('2 most recent announcements'));

  const empty = context.buildWhatsNewQuestion([], null);
  assert.ok(empty.includes('no announcements'));
  assert.equal((empty.match(/\d\. /g) || []).length, 0, 'never invents announcements');

  // Bounded even if more announcements are passed in.
  const many = Array.from({ length: 6 }, (_, index) => ({
    title: `Announcement ${index + 1}`, date: null, href: null,
  }));
  const bounded = context.buildWhatsNewQuestion(many, '1 Oct 2026');
  assert.equal((bounded.match(/\d\. /g) || []).length, 3);
  assert.ok(!bounded.includes('Announcement 4'));
});

test('visit history degrades safely when storage is unavailable', () => {
  const context = whatsNewContext();
  const working = {
    getItem: () => '2026-10-26T09:00:00.000Z',
    setItem: () => {},
  };
  assert.equal(context.readLastVisit(working).toISOString(), '2026-10-26T09:00:00.000Z');

  const empty = { getItem: () => null, setItem: () => {} };
  assert.equal(context.readLastVisit(empty), null);

  const broken = {
    getItem: () => { throw new Error('storage blocked'); },
    setItem: () => { throw new Error('storage blocked'); },
  };
  assert.equal(context.readLastVisit(broken), null);
  assert.doesNotThrow(() => context.recordVisit(broken));
  assert.equal(context.readLastVisit({ getItem: () => 'not-a-date', setItem: () => {} }), null);

  // The timestamp records a visit only — never any content.
  const values = {};
  context.recordVisit({
    getItem: key => values[key] ?? null,
    setItem: (key, value) => { values[key] = value; },
  });
  assert.equal(Object.keys(values)[0], 'eka-portal-last-visit-v1');
  assert.equal(Object.keys(values).length, 1);
  assert.match(values['eka-portal-last-visit-v1'], /^\d{4}-\d{2}-\d{2}T/);
});

test('the what\'s-new action is wired to the assistant hand-off', () => {
  assert.match(
    portalSource,
    /<button type="button" class="whats-new-button" data-whats-new>What's new since your last visit\?<\/button>/,
  );
  assert.match(portalSource, /lastVisitBeforeThisVisit = readLastVisit\(\);/);
  assert.match(portalSource, /recordVisit\(\);/);
  assert.match(portalSource, /buildWhatsNewQuestion\(collectAnnouncements\(document\), label\)/);
  assert.match(portalSource, /window\.location\.href = assistantQuestionUrl\(question\);/);
  // The hand-off reuses the Phase 2 assistant route.
  assert.match(portalSource, /return `chat\.html\?q=\$\{encodeURIComponent\(trimmed\)\}`;/);
});

// ---------- 4. Accessibility and UI polish ----------

test('every page offers a skip link that targets real content', () => {
  assert.match(portalSource, /<body class="employee-portal-page">\s*<a class="skip-link" href="#top">Skip to content<\/a>/);
  assert.match(portalSource, /<main id="top"/);
  assert.match(indexSource, /<a class="skip-link" href="#main-content">Skip to content<\/a>/);
  assert.match(appSource, /<a class="skip-link" href="#main-content">Skip to content<\/a>/);
  // Skip-link targets are focusable so activating the link actually moves
  // focus into the content (fragment targets without tabindex only move the
  // tab sequence starting point).
  assert.match(portalSource, /<main id="top" class="portal-main" tabindex="-1">/);
  assert.match(indexSource, /<main class="portal-login-shell" id="main-content" tabindex="-1">/);
  assert.match(appSource, /<main class="main" id="main-content" tabindex="-1">/);
  // The skip link appears before the navigation and main landmarks.
  const skip = appSource.indexOf('skip-link');
  const shell = appSource.indexOf('<div class="app-shell">');
  const sidebar = appSource.indexOf('${renderSidebar(activePage)}');
  const header = appSource.indexOf('${renderHeader(title, breadcrumb, headerActions)}');
  const main = appSource.indexOf('<main class="main" id="main-content"');
  assert.ok(skip > 0 && shell > skip && sidebar > shell && header > sidebar && main > header,
    'skip link must precede navigation and main landmarks');
  assert.match(cssSource, /\.skip-link \{/);
  assert.match(cssSource, /\.skip-link:focus \{ transform: translateY\(0\); \}/);
});

test('headings have a single h1 and a clean order', () => {
  assert.equal((chatSource.match(/<h1/g) || []).length, 1, 'one h1: the active assistant');
  assert.ok(chatSource.includes('<h2 class="welcome-title">'));
  assert.ok(chatSource.includes('<h3 id="recentQuestionsHeading">'));
  assert.match(portalSource, /<h1 id="portalHeading">/);
});

test('decorative marks are hidden and interactive icons are named', () => {
  // Decorative glyphs/initials hidden from screen readers.
  assert.match(documentsSource, /document-file-mark[^>]*aria-hidden="true"/);
  assert.match(appSource, /brand-mark" aria-hidden="true"/);
  assert.match(appSource, /user-avatar" aria-hidden="true"/);
  assert.match(chatSource, /agent-mark \$\{view\.id \|\| 'automatic'\}" aria-hidden="true"/);
  assert.match(portalSource, /portal-brand-mark" aria-hidden="true">C<\/span>/);
  assert.match(portalSource, /<span aria-hidden="true">ⓘ<\/span>/);
  // Icon-only controls carry accessible names.
  assert.match(portalSource, /aria-label="Ask Cathy, the workplace assistant"/);
  assert.match(portalSource, /aria-label="Minimize assistant"/);
  assert.match(portalSource, /aria-label="Close assistant"/);
  assert.match(chatSource, /aria-label="Send message"/);
  assert.match(chatSource, /aria-label="Close ticket draft"/);
  assert.match(documentsSource, /aria-label="Delete \$\{escapeHtml\(document\.name\)\}"/);
  assert.match(appSource, /aria-label="Open navigation"/);
});

test('focus is clearly visible and reduced motion is respected', () => {
  assert.match(cssSource, /outline: 2px solid #5C3FD0;/);
  assert.match(cssSource, /button:focus-visible, a:focus-visible/);
  const reducedBlocks = cssSource.match(/@media \(prefers-reduced-motion: reduce\) \{[\s\S]*?\n\}/g) || [];
  assert.ok(reducedBlocks.length >= 3, 'existing plus new reduced-motion coverage');
  const joined = reducedBlocks.join('\n');
  for (const selector of ['.skip-link', '.feedback-button', '.recent-question', '.whats-new-button', '.portal-announcement-card']) {
    assert.ok(joined.includes(selector), `reduced motion should cover ${selector}`);
  }
});

test('redundant sample labels are gone while the disclosure remains', () => {
  assert.ok(!portalSource.includes('SAMPLE CONTENT'), 'redundant chip removed');
  assert.ok(!portalSource.includes('· SAMPLE'), 'redundant per-card labels removed');
  assert.equal((portalSource.match(/portal-demo-notice/g) || []).length, 1, 'single disclosure kept');
  assert.match(portalSource, /Northstar is a fictional company, and the announcements below are sample demonstration content\./);
  assert.match(portalSource, /Fictional demonstration company/);
  assert.match(indexSource, /Northstar is a fictional demonstration company\./);
  assert.match(chatSource, /AI-generated responses can be inaccurate/);
});