const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const read = relative => fs.readFileSync(path.join(root, relative), 'utf8');

const chatSource = read('chat.html');
const analyticsSource = read('analytics.html');
const appSource = read('js/app.js');
const routesSource = read('backend/app/api/routes.py');

function extractFunction(source, name) {
  const match = source.match(new RegExp(`function ${name}\\([^)]*\\) \\{[\\s\\S]*?\\n    \\}`));
  assert.ok(match, `expected to find function ${name}`);
  return match[0];
}

// Evaluates the feedback helpers together so they can reference each other and
// the script-level reason list.
function feedbackContext(globals = {}) {
  const source = chatSource;
  const declarations = [
    (source.match(/const FEEDBACK_REASONS = \[[\s\S]*?\];/) || [null])[0],
    'function escapeHtml(value) { return String(value ?? "").replace(/&/g, "&").replace(/</g, "<").replace(/>/g, ">"); }',
    ...[
      'answerKeyFor',
      'messageForAnswerKey',
      'questionForMessage',
      'sourceStatusFor',
      'reasonLabel',
      'renderReasonChips',
      'renderFeedbackControls',
    ].map(name => extractFunction(source, name)),
  ].filter(Boolean);
  const context = vm.createContext({ Array, String, Boolean, Math, ...globals });
  vm.runInContext(declarations.join('\n'), context);
  return context;
}

// ---------- Controls rendered on assistant answers ----------

test('every assistant answer offers thumbs-up and thumbs-down controls', () => {
  assert.match(chatSource, /\$\{!isUser \? renderFeedbackControls\(message\) : ''\}/);
  const controls = feedbackContext({ messages: [] }).renderFeedbackControls({ content: 'Answer', time: '10:00' });

  assert.equal((controls.match(/data-feedback=/g) || []).length, 2);
  assert.ok(controls.includes('data-feedback="positive"'));
  assert.ok(controls.includes('data-feedback="negative"'));
  assert.ok(controls.includes('aria-label="This answer was helpful"'));
  assert.ok(controls.includes('aria-label="This answer was not helpful"'));
  assert.ok(controls.includes('data-answer-key="answer-'));
  assert.ok(controls.includes('Was this helpful?'));
});

test('the reason list matches the required negative-feedback reasons', () => {
  const context = feedbackContext({ messages: [] });
  vm.runInContext('globalThis.reasons = FEEDBACK_REASONS;', context);
  assert.deepEqual(Array.from(context.reasons, ([value]) => value), [
    'incorrect_answer',
    'missing_information',
    'irrelevant_sources',
    'outdated_document',
    'other',
  ]);
});

test('feedback states cover pending, saved, failed and reasons', () => {
  const context = feedbackContext({ messages: [] });

  const idle = context.renderFeedbackControls({ content: 'Answer' });
  assert.ok(!idle.includes('Saving…'));
  assert.ok(!idle.includes('feedback-error'));

  const pending = context.renderFeedbackControls({ content: 'Answer', feedbackPending: true });
  assert.ok(pending.includes('Saving…'));
  assert.ok(/data-feedback="positive"[^>]*disabled/.test(pending));
  assert.ok(/data-feedback="negative"[^>]*disabled/.test(pending));

  const saved = context.renderFeedbackControls({
    content: 'Answer',
    feedback: { rating: 'positive', reason: null },
  });
  assert.ok(saved.includes('Marked helpful'));
  assert.ok(saved.includes('aria-pressed="true"'));

  const failed = context.renderFeedbackControls({
    content: 'Answer',
    feedbackError: 'Your feedback could not be saved. Please try again.',
  });
  assert.ok(failed.includes('role="alert"'));
  assert.ok(failed.includes('could not be saved'));

  const negative = context.renderFeedbackControls({
    content: 'Answer',
    feedback: { rating: 'negative', reason: 'missing_information' },
  });
  assert.ok(negative.includes('data-reason='));
  assert.ok(negative.includes('Missing information'));
  assert.ok((negative.match(/data-reason=/g) || []).length === 5);
  assert.ok(negative.includes('selected'));
  // Reasons are only offered for negative feedback.
  assert.ok(!saved.includes('data-reason='));
});

// ---------- Answer identity and payload construction ----------

test('the answer key is stable, deterministic and content-free', () => {
  const context = feedbackContext({ messages: [] });
  const message = { time: '10:00:00', agent: 'hr', content: 'Casual leave is 12 days.' };

  const key = context.answerKeyFor(message);
  assert.equal(context.answerKeyFor(message), key, 'same answer → same key');
  assert.notEqual(
    key,
    context.answerKeyFor({ ...message, content: 'Different answer.' }),
  );
  assert.match(key, /^answer-[a-z0-9]+$/);
  assert.ok(key.length >= 8);
  assert.match(key, /^[A-Za-z0-9_-]+$/, 'key must satisfy the API pattern');
  // The key derives from the answer itself; no conversation payload is embedded.
  assert.ok(!key.includes('Casual'));
});

test('the question sent with feedback is the preceding user question, capped', () => {
  const question = 'How many casual leaves can I take?';
  const answer = { role: 'agent', content: 'You have 12 days.', time: '10:00' };
  const context = feedbackContext({
    messages: [
      { role: 'user', content: question, time: '09:59' },
      answer,
    ],
  });

  assert.equal(context.questionForMessage(answer), question);
  // A stored question wins, and long text is capped to the API limit.
  assert.equal(context.questionForMessage({ question: 'Stored question' }), 'Stored question');
  assert.equal(context.questionForMessage(answer).length <= 300, true);
  // A message with no preceding user turn still yields a usable string.
  const orphan = feedbackContext({ messages: [answer] });
  assert.equal(orphan.questionForMessage(answer), '');
});

test('the source status distinguishes no-source, retrieval failures and errors', () => {
  const context = feedbackContext({ messages: [] });

  assert.equal(context.sourceStatusFor({ noEvidence: true, isError: true }), 'no_sources');
  assert.equal(context.sourceStatusFor({ isError: true, errorCode: 'retrieval_error' }), 'retrieval_error');
  assert.equal(context.sourceStatusFor({ isError: true, errorCode: 'network' }), 'error');
  assert.equal(context.sourceStatusFor({ sources: [{ document: 'a.pdf' }] }), 'sources');
  assert.equal(context.sourceStatusFor({}), 'no_sources');
});

test('the submission posts only validated feedback fields', () => {
  const submit = extractFunction(chatSource, 'submitFeedback');

  for (const marker of [
    'API.submitFeedback({',
    'rating,',
    'answer_key: answerKey',
    'question: questionForMessage(message).slice(0, 300)',
    "source_status: sourceStatusFor(message)",
    'reason: nextReason',
    'message.feedbackPending = true;',
    'message.feedbackError = ',
    'persistConversations();',
  ]) {
    assert.ok(submit.includes(marker), `missing in submitFeedback: ${marker}`);
  }
  // No conversation history, tokens or assistant answers are transmitted.
  for (const forbidden of ['conversation_history', 'csrf', 'password', 'localStorage.getItem']) {
    assert.ok(!submit.includes(forbidden), `feedback must not send ${forbidden}`);
  }
  assert.match(appSource, /submitFeedback: payload =>\s*\n\s*request\('\/api\/feedback'/);
});

test('feedback clicks are delegated from the transcript', () => {
  for (const marker of [
    "event.target.closest('[data-feedback]')",
    "event.target.closest('[data-reason]')",
    "feedbackButton.closest('.message-feedback')",
    "reasonButton.dataset.reason",
  ]) {
    assert.ok(chatSource.includes(marker), `missing wiring: ${marker}`);
  }
});

test('recorded feedback survives a reload of saved conversations', () => {
  // The rating is written onto the message object and persisted with it.
  assert.match(extractFunction(chatSource, 'submitFeedback'), /message\.feedback = \{ rating: response\.rating/);
  assert.match(chatSource, /persistConversations\(\);/);
});

// ---------- Analytics panel (administrator only) ----------

test('the analytics page renders feedback aggregates for administrators', () => {
  for (const marker of [
    'function renderFeedbackPanel(feedback)',
    'function renderFeedbackTrend(days)',
    'Helpful votes',
    'Not-helpful votes',
    'Questions without sources',
    'Unanswered questions',
    'Negative feedback reasons',
    'Feedback by topic',
    'Reported knowledge gaps',
    'Feedback trend',
    '${renderFeedbackPanel(analyticsData.feedback)}',
  ]) {
    assert.ok(analyticsSource.includes(marker), `missing analytics marker: ${marker}`);
  }
  assert.match(analyticsSource, /mountShell\('analytics', 'Analytics', 'Workspace \/ Analytics', mainContent, '', 'admin'\)/);
});

test('the analytics report exposes aggregates only, never conversation content', () => {
  const panel = extractFunction(analyticsSource, 'renderFeedbackPanel');
  // Only aggregate fields are read from the API payload.
  for (const field of [
    'feedback.total_positive',
    'feedback.total_negative',
    'feedback.negative_reasons',
    'feedback.feedback_by_agent',
    'feedback.feedback_by_day',
    'feedback.knowledge_gaps',
    'feedback.no_source_questions',
    'feedback.unanswered_questions',
  ]) {
    assert.ok(panel.includes(field), `panel should read ${field}`);
  }
  for (const forbidden of ['username', 'conversation', 'answer_text']) {
    assert.ok(!panel.includes(forbidden), `panel must not read ${forbidden}`);
  }
});

test('the analytics navigation stays administrator-only in the shell', () => {
  assert.match(
    appSource,
    /\.\.\.\(realAdmin \? \[\s*\n\s*\{ id: "documents".*\n\s*\{ id: "analytics"/s,
  );
  assert.match(appSource, /const realAdmin = user\.role === 'admin' && !isDemoAdminUser\(user\);/);
});