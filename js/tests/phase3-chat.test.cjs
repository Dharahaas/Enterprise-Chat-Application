const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const read = relative => fs.readFileSync(path.join(root, relative), 'utf8');

const chatSource = read('chat.html');
const cssSource = read('css/pages.css');
const appSource = read('js/app.js');
const routesSource = read('backend/app/api/routes.py');

function extractFunction(source, name) {
  const match = source.match(new RegExp(`function ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n    \\}`));
  assert.ok(match, `expected to find function ${name}`);
  return match[0];
}

function runFunction(source, name, globals = {}) {
  const context = vm.createContext({ Array, String, Boolean, ...globals });
  vm.runInContext(
    `${extractFunction(source, name)}\nglobalThis.fn = ${name};`,
    context,
  );
  return context.fn;
}

// Evaluates helpers together so they can call each other inside one realm.
function runFunctions(source, names, globals = {}) {
  const context = vm.createContext({ Array, String, Boolean, ...globals });
  const body = names.map(name => extractFunction(source, name)).join('\n');
  vm.runInContext(`${body}\nglobalThis.fn = ${names[names.length - 1]};`, context);
  return context.fn;
}

// ---------- A: unified copy and automatic-mode identity ----------

test('the welcome copy no longer tells users to choose a specialist', () => {
  assert.ok(!chatSource.includes('Choose a specialist, then ask a question'));
  assert.match(
    chatSource,
    /Ask Cathy anything about HR, IT or company policies/,
  );
});

test('automatic-mode messages use the deliberate automatic avatar class', () => {
  assert.match(chatSource, /agent-avatar \$\{agent\.id \|\| 'automatic'\}/);
  assert.match(chatSource, /agent-avatar \$\{requestAgent\.id \|\| 'automatic'\}/);
  assert.match(cssSource, /\.agent-avatar\.automatic \{/);
  assert.match(cssSource, /\.agent-dot\.automatic \{/);
});

// ---------- C: only valid retrieved sources are shown ----------

test('source labels require a real document name', () => {
  const sourceLabel = runFunction(chatSource, 'sourceLabel');

  assert.equal(
    sourceLabel({ document: 'Leave_Policy.pdf', page: 2, section: 'Leave' }),
    'Leave_Policy.pdf · p. 2 · § Leave',
  );
  assert.equal(sourceLabel({ document: 'IT_Network_Guide.pdf' }), 'IT_Network_Guide.pdf');
  // Unusable citations are dropped instead of being shown as "Document".
  assert.equal(sourceLabel({ document: 'unknown', snippet: 'text' }), '');
  assert.equal(sourceLabel({ document: '', snippet: 'text' }), '');
  assert.equal(sourceLabel({ snippet: 'text' }), '');
  assert.equal(sourceLabel(null), '');
  assert.equal(sourceLabel('Leave_Policy.pdf'), '');
});

test('invalid and duplicate sources are filtered before rendering', () => {
  const validSourceLabels = runFunctions(chatSource, ['sourceLabel', 'validSourceLabels']);

  const labels = validSourceLabels([
    { document: 'Leave_Policy.pdf', page: 2 },
    { document: 'unknown', snippet: 'no citation' },
    { document: '', snippet: 'no name' },
    null,
    { document: 'Leave_Policy.pdf', page: 2 },
    { document: 'Hybrid_Work_Policy.pdf', section: 'Hybrid' },
  ]);
  // The array comes from the VM realm, so compare values, not prototypes.
  assert.deepEqual(Array.from(labels), [
    'Leave_Policy.pdf · p. 2',
    'Hybrid_Work_Policy.pdf · § Hybrid',
  ]);
  assert.equal(validSourceLabels(undefined).length, 0);
});

test('sources render only when valid labels exist', () => {
  assert.match(chatSource, /const labels = validSourceLabels\(sources\);/);
  assert.match(chatSource, /if \(!labels\.length\) return '';/);
  assert.ok(!chatSource.includes("source.document || source.name || source.source || 'Document'"));
});

// ---------- C: no-source support actions ----------

test('the support block renders only for messages without evidence', () => {
  assert.match(chatSource, /\$\{message\.noEvidence \? renderSupportActions\(\) : ''\}/);
  assert.match(
    chatSource,
    /noEvidence: response\.error_code === 'insufficient_evidence'/,
  );
  assert.match(chatSource, /noEvidence: error\.code === 'insufficient_evidence'/);
});

test('the support action offers a working document link for each role', () => {
  const render = runFunction(chatSource, 'renderSupportActions', {
    appUser: { role: 'admin' },
    isDemoAdminUser: () => false,
  });
  const employee = runFunction(chatSource, 'renderSupportActions', {
    appUser: { role: 'employee' },
    isDemoAdminUser: () => false,
  });
  const demoAdmin = runFunction(chatSource, 'renderSupportActions', {
    appUser: { role: 'admin' },
    isDemoAdminUser: () => true,
  });

  // The Documents page is admin-only, so other roles get a link that works.
  assert.match(render(), /href="documents\.html"/);
  assert.doesNotMatch(employee(), /href="documents\.html"/);
  assert.match(employee(), /href="portal\.html#resources"/);
  assert.doesNotMatch(demoAdmin(), /href="documents\.html"/);
  assert.match(demoAdmin(), /href="portal\.html#resources"/);
  for (const html of [render(), employee(), demoAdmin()]) {
    assert.ok(html.includes('Still need help? Raise an IT/HR ticket'));
    assert.ok(html.includes('data-raise-ticket'));
  }
});

test('retrieval failures stay distinct from missing evidence in the UI', () => {
  // Only the insufficient_evidence code flips on the support block; the
  // retrieval_error path renders a plain error message without it.
  assert.match(appSource, /if \(code\) error\.code = code;/);
  assert.match(appSource, /message\.error\?\.code/);
  assert.ok(!chatSource.includes("noEvidence: error.code === 'retrieval_error'"));
});

// ---------- D: escalation draft ----------

test('no ticket backend exists, so escalation must stay a labelled draft', () => {
  assert.doesNotMatch(routesSource, /ticket/i);
  assert.match(
    chatSource,
    /Draft only — no ticket system is connected yet, so <strong>nothing has been submitted<\/strong>/,
  );
  // The draft is never transmitted by the page itself.
  assert.doesNotMatch(chatSource, /fetch\(/);
  assert.ok(!chatSource.includes('API.createTicket'));
});

test('the ticket draft is built from the conversation and department', () => {
  const buildTicketDraft = runFunction(chatSource, 'buildTicketDraft');
  const turns = [
    { role: 'user', content: 'My VPN stopped connecting after the update.' },
    { role: 'agent', content: 'Restart AnyConnect and reconnect.' },
  ];

  const itDraft = buildTicketDraft(turns, 'IT');
  assert.ok(itDraft.subject.startsWith('IT support request: '));
  assert.ok(itDraft.subject.includes('My VPN stopped connecting'));
  assert.match(itDraft.body, /^Department: IT$/m);
  assert.ok(itDraft.body.includes('My VPN stopped connecting after the update.'));
  assert.ok(itDraft.body.includes('Restart AnyConnect and reconnect.'));
  assert.ok(itDraft.body.includes('has not been submitted to any ticket system'));

  const hrDraft = buildTicketDraft(turns, 'HR');
  assert.ok(hrDraft.subject.startsWith('HR support request: '));
  assert.match(hrDraft.body, /^Department: HR$/m);

  // Empty conversations still produce a usable draft.
  const empty = buildTicketDraft([], 'HR');
  assert.ok(empty.subject.includes('Assistance needed'));
  assert.match(empty.body, /^Department: HR$/m);

  // Long assistant replies are excerpted, not dumped in full.
  const longReply = buildTicketDraft(
    [{ role: 'agent', content: 'x'.repeat(900) }],
    'IT',
  );
  assert.ok(longReply.body.includes(`${'x'.repeat(600)}…`));
  assert.ok(!longReply.body.includes('x'.repeat(601)));
});

test('the ticket department follows the current specialist', () => {
  const department = runFunction(chatSource, 'ticketDepartmentForCurrentView', {
    activeAgent: { id: 'it' },
  });
  const automatic = runFunction(chatSource, 'ticketDepartmentForCurrentView', {
    activeAgent: null,
  });
  const hr = runFunction(chatSource, 'ticketDepartmentForCurrentView', {
    activeAgent: { id: 'hr' },
  });

  assert.equal(department(), 'IT');
  assert.equal(automatic(), 'HR');
  assert.equal(hr(), 'HR');
});

test('the draft dialog lets the user review before anything is sent', () => {
  for (const marker of [
    '<dialog class="ticket-draft" id="ticketDraft"',
    'id="ticketDepartment"',
    '<option value="HR">HR</option>',
    '<option value="IT">IT</option>',
    'id="ticketSubject"',
    'id="ticketBody"',
    'id="ticketDraftCopy">Copy draft<',
    'role="status" aria-live="polite"',
  ]) {
    assert.ok(chatSource.includes(marker), `missing dialog marker: ${marker}`);
  }
  for (const wiring of [
    "if (event.target.closest('[data-raise-ticket]')) {",
    "          openTicketDraft();",
    "getElementById('ticketDraftCopy')?.addEventListener('click', copyTicketDraft)",
    "getElementById('ticketDepartment')?.addEventListener('change', renderTicketDraft)",
    "getElementById('chatTranscript')?.addEventListener('click'",
  ]) {
    assert.ok(chatSource.includes(wiring), `missing wiring: ${wiring}`);
  }
  assert.match(chatSource, /Nothing has been sent/);
});

test('conversation context stays behind the existing session check', () => {
  // The draft is assembled from the signed-in user's own conversation only.
  assert.match(chatSource, /await requireAuth\(\)/);
  assert.ok(!chatSource.includes('localStorage.setItem(\'ticket'));
});