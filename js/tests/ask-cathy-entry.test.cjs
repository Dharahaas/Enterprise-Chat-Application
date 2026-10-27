const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const read = relative => fs.readFileSync(path.join(root, relative), 'utf8');

const portalSource = read('portal.html');
const chatSource = read('chat.html');
const appSource = read('js/app.js');

const EXAMPLE_QUESTIONS = [
  'How many leave days do I have?',
  'My VPN is not connecting.',
  'What is the work-from-home policy?',
  'How do I reset my password?',
];

function extractFunction(source, name) {
  const match = source.match(new RegExp(`function ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n    \\}`));
  assert.ok(match, `expected to find function ${name}`);
  return match[0];
}

function runFunction(source, name) {
  const context = { URLSearchParams, String, encodeURIComponent };
  vm.runInNewContext(
    `${extractFunction(source, name)}\nglobalThis.fn = ${name};`,
    context,
  );
  return context.fn;
}

// ---------- Unified naming (FR-01) ----------

test('every assistant entry point on the homepage is called Ask Cathy', () => {
  for (const expected of [
    'class="portal-hero-ask-submit" id="heroAskSubmit" type="submit">Ask Cathy<',
    'class="portal-header-assistant" aria-controls="assistantPanel"',
    '>Ask Cathy</button>',
    'class="portal-support-button" onclick="openAssistant()">Ask Cathy <',
    'aria-label="Ask Cathy, the workplace assistant"',
    '<strong>Ask Cathy</strong><small>WORKPLACE ASSISTANT',
    '<strong>Ask Cathy</strong><small>Ask anything about HR, IT or company policies',
  ]) {
    assert.ok(portalSource.includes(expected), `missing unified label: ${expected}`);
  }
});

test('inconsistent assistant labels are gone from the homepage', () => {
  for (const stale of [
    'Ask the company assistant',
    'Open AI help',
    'Choose a specialist assistant',
  ]) {
    assert.ok(!portalSource.includes(stale), `stale label still present: ${stale}`);
  }
});

test('the hero has exactly one primary entry point and the header has a persistent one', () => {
  assert.equal(portalSource.match(/class="portal-hero-ask"/g).length, 1);
  assert.equal(portalSource.match(/class="portal-header-assistant"/g).length, 1);
  assert.ok(portalSource.includes('id="assistantFab"'), 'the floating entry point stays');
});

// ---------- Hero question input (FR-02) ----------

test('the hero exposes an accessible question input with the four example questions', () => {
  assert.match(portalSource, /<form class="portal-hero-ask" id="heroAskForm" novalidate>/);
  assert.match(
    portalSource,
    /<label class="sr-only" for="heroAskInput">Ask Cathy a question<\/label>/,
  );
  assert.match(
    portalSource,
    /id="heroAskInput" name="q" type="text" maxlength="2000"[^>]*placeholder="Ask anything about HR, IT or company policies…"/,
  );
  assert.match(portalSource, /id="heroAskError" role="alert" aria-live="polite"/);
  for (const question of EXAMPLE_QUESTIONS) {
    assert.ok(
      portalSource.includes(`data-hero-example="${question}"`),
      `missing example question: ${question}`,
    );
  }
  assert.equal(portalSource.match(/data-hero-example=/g).length, 4);
});

test('the hero question is handed to Ask Cathy with a query parameter', () => {
  const questionUrl = runFunction(portalSource, 'assistantQuestionUrl');

  assert.equal(questionUrl('   '), null);
  assert.equal(questionUrl(''), null);
  assert.equal(questionUrl(undefined), null);
  assert.equal(
    questionUrl('  How many leave days do I have?  '),
    'chat.html?q=How%20many%20leave%20days%20do%20I%20have%3F',
  );
  assert.equal(questionUrl('VPN & network down?'), 'chat.html?q=VPN%20%26%20network%20down%3F');
});

test('the hero submits on Enter and blocks empty or duplicate submissions', () => {
  assert.match(
    portalSource,
    /getElementById\('heroAskForm'\)\?\.addEventListener\('submit', event => \{\s*\n\s*event\.preventDefault\(\);/,
  );
  assert.match(portalSource, /if \(heroAskSubmitting\) return;/);
  assert.match(portalSource, /heroAskSubmitting = true;/);
  assert.match(portalSource, /submit\.disabled = true;\s*\n\s*submit\.textContent = 'Opening Ask Cathy…';/);
  assert.match(portalSource, /window\.addEventListener\('pageshow', resetHeroQuestionForm\)/);
});

// ---------- ?q= hand-off into the existing assistant flow ----------

test('the assistant reads and consumes the homepage question exactly once', () => {
  const readInitialQuestion = runFunction(chatSource, 'readInitialQuestion');

  assert.equal(readInitialQuestion(new URLSearchParams('')), '');
  assert.equal(readInitialQuestion(new URLSearchParams('q=%20%20')), '');
  assert.equal(
    readInitialQuestion(new URLSearchParams('q=My%20VPN%20is%20not%20connecting.')),
    'My VPN is not connecting.',
  );

  assert.match(chatSource, /readInitialQuestion\(chatParams\)/);
  assert.match(chatSource, /if \(question\) submitQuestion\(question\);/);
  assert.match(chatSource, /url\.searchParams\.delete\('q'\);/);
  assert.match(chatSource, /if \(initialQuestionHandled\) return;/);
});

// ---------- Topic handling and automatic routing (FR-03 / requirement 6) ----------

test('the shared AGENTS list still holds only the three specialists', () => {
  const match = appSource.match(/const AGENTS = (\[[\s\S]*?\n\]);/);
  assert.ok(match, 'expected app.js to declare AGENTS');
  const agents = vm.runInNewContext(match[1], {});
  // The array comes from another realm, so compare values, not prototypes.
  assert.deepEqual(Array.from(agents, agent => agent.id), ['hr', 'it', 'policy']);
  assert.ok(!agents.some(agent => /ask cathy/i.test(agent.name || '')));
  assert.ok(!appSource.includes('"auto"') && !appSource.includes("'auto'"));
});

test('a general entry point starts in automatic mode instead of forcing HR', () => {
  assert.match(chatSource, /const AUTOMATIC_ASSISTANT = \{/);
  assert.match(chatSource, /name: 'Ask Cathy'/);
  const resolve = runFunction(chatSource, 'resolveRequestedAgent');
  const agents = [{ id: 'hr' }, { id: 'it' }, { id: 'policy' }];

  assert.equal(resolve(new URLSearchParams(''), agents), null);
  assert.equal(resolve(new URLSearchParams('q=hello'), agents), null);
  assert.equal(resolve(new URLSearchParams('agent=nope'), agents), null);
  // An explicit topic still guides the initial routing.
  assert.equal(resolve(new URLSearchParams('topic=hr'), agents).id, 'hr');
  assert.equal(resolve(new URLSearchParams('topic=it'), agents).id, 'it');
  assert.equal(resolve(new URLSearchParams('topic=policy'), agents).id, 'policy');

  // The general entry points no longer select a specialist.
  assert.match(portalSource, /let selectedAssistantAgent = null;/);
  assert.match(portalSource, /if \(!selectedAssistantAgent\) return;/);
  assert.match(portalSource, /src="chat\.html\?embedded=1"/);
  assert.doesNotMatch(portalSource, /src="chat\.html\?embedded=1&/);
  assert.doesNotMatch(portalSource, /openAssistant\('hr'\)/);
});

test('an explicit topic guides the fallback but the request still carries the specialist', () => {
  assert.match(chatSource, /createConversation\(activeAgent \? activeAgent\.id : null\)/);
  assert.match(chatSource, /API\.chatStream\(\s*\n\s*question,\s*\n\s*conversation\.agent,/);
  assert.doesNotMatch(chatSource, /API\.chatStream\(\s*\n\s*question,\s*\n\s*'hr',/);
});

// ---------- Compatibility with saved conversations (requirement 7) ----------

test('saved conversations from earlier releases stay loadable', () => {
  const isSupported = runFunction(chatSource, 'isSupportedConversationAgent');
  const agents = [{ id: 'hr' }, { id: 'it' }, { id: 'policy' }];

  assert.equal(isSupported('hr', agents), true);
  assert.equal(isSupported('it', agents), true);
  assert.equal(isSupported('policy', agents), true);
  assert.equal(isSupported(null, agents), true);
  assert.equal(isSupported('finance', agents), false);
  assert.equal(isSupported(undefined, agents), false);

  assert.match(chatSource, /isSupportedConversationAgent\(conversation\.agent, AGENTS\)/);
});

test('specialist switching and embedded chat keep working', () => {
  assert.match(chatSource, /function switchAgent\(agentId\)/);
  assert.match(chatSource, /function switchToAutomatic\(\)/);
  assert.match(chatSource, /if \(!nextAgent \|\| \(activeAgent && nextAgent\.id === activeAgent\.id\)\) return;/);
  assert.match(chatSource, /url\.searchParams\.set\('agent', activeAgent\.id\);/);
  assert.match(
    chatSource,
    /event\.data\?\.type !== 'eka:select-agent'/,
  );
  assert.match(chatSource, /switchAgent\(event\.data\.agent\)/);
  assert.match(chatSource, /event\.origin !== location\.origin/);
  assert.match(chatSource, /chatParams\.get\('embedded'\) === '1'/);
  // Backward compatibility: ?agent= links still select the same specialist.
  assert.match(chatSource, /params\.get\('agent'\), params\.get\('topic'\)/);
});