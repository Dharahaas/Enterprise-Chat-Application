const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '../..');
const portalSource = fs.readFileSync(path.join(root, 'portal.html'), 'utf8');
const chatSource = fs.readFileSync(path.join(root, 'chat.html'), 'utf8');
const pagesCss = fs.readFileSync(path.join(root, 'css', 'pages.css'), 'utf8');

const hrefs = [...portalSource.matchAll(/href="([^"]+)"/g)].map(match => match[1]);

function quickLinkCard(label) {
  const labelIndex = portalSource.indexOf(`<strong>${label}</strong>`);
  assert.ok(labelIndex > -1, `expected a homepage quick link for "${label}"`);
  const cardStart = portalSource.lastIndexOf('<a class="portal-resource-card"', labelIndex);
  assert.ok(cardStart > -1, `expected "${label}" to be a link, not a button`);
  return portalSource.slice(cardStart, portalSource.indexOf('>', cardStart) + 1);
}

test('every homepage link resolves to a real destination or an in-page anchor', () => {
  assert.ok(hrefs.length > 0, 'expected the homepage to declare links');
  for (const href of hrefs) {
    if (href.startsWith('#')) {
      const id = href.slice(1);
      assert.ok(id, `anchor "${href}" has no target id`);
      assert.ok(
        portalSource.includes(`id="${id}"`),
        `homepage link ${href} points at an anchor that does not exist`,
      );
      continue;
    }
    const [target] = href.split(/[?#]/);
    assert.ok(
      fs.existsSync(path.join(root, target)),
      `homepage link ${href} points at a missing file`,
    );
  }
});

test('People & HR, IT Help Desk and Policies & Compliance open the assistant with a topic', () => {
  assert.match(quickLinkCard('People & HR'), /href="chat\.html\?topic=hr"/);
  assert.match(quickLinkCard('IT help desk'), /href="chat\.html\?topic=it"/);
  assert.match(quickLinkCard('Policies & compliance'), /href="chat\.html\?topic=policy"/);
});

test('quick links are real anchors so they survive direct navigation and refresh', () => {
  const resourceCards = [...portalSource.matchAll(/<a class="portal-resource-card" href="([^"]+)"/g)];
  assert.equal(resourceCards.length, 4, 'expected four functional quick links');
  assert.doesNotMatch(
    portalSource,
    /<button class="portal-resource-card"/,
    'quick links must not be inert buttons',
  );
});

test('internal navigation indicators use → and reserve ↗ for external links', () => {
  const arrows = [...portalSource.matchAll(/<span class="portal-resource-arrow"[^>]*>([^<]*)</g)]
    .map(match => match[1]);
  assert.equal(arrows.length, 4);
  assert.ok(arrows.every(arrow => arrow.trim() === '→'), `unexpected indicators: ${arrows}`);
  assert.match(
    portalSource,
    /<button type="button" class="portal-support-button"[^>]*>[^<]*<span aria-hidden="true">→<\/span><\/button>/,
  );
});

test('announcement cards are rendered beneath their section heading', () => {
  const sectionStart = portalSource.indexOf('<section id="announcements"');
  const headingIndex = portalSource.indexOf('<h2 id="announcementsHeading">');
  const gridIndex = portalSource.indexOf('<div class="portal-announcement-grid">');
  const sectionEnd = portalSource.indexOf('</section>', gridIndex);

  assert.ok(sectionStart > -1, 'expected an announcements section');
  assert.ok(headingIndex > sectionStart, 'expected the heading inside the announcements section');
  assert.ok(gridIndex > headingIndex, 'announcement cards must come after the heading');
  assert.ok(sectionEnd > gridIndex, 'expected the announcement grid inside the section');

  const cards = portalSource.slice(gridIndex, sectionEnd).match(/class="portal-announcement-card/g);
  assert.ok(cards && cards.length > 0, 'expected announcement cards inside the grid');
  assert.ok(
    portalSource.includes('aria-labelledby="announcementsHeading"'),
    'the announcements section must be labelled for screen readers',
  );
});

test('announcement layout keeps the heading and grid in normal document flow', () => {
  const gridRule = pagesCss.match(/\.portal-announcement-grid\s*\{[^}]*\}/);
  assert.ok(gridRule, 'expected styles for the announcement grid');
  assert.doesNotMatch(gridRule[0], /position\s*:\s*(absolute|fixed)/);
  const headingRule = pagesCss.match(/\.portal-section-heading\s*\{[^}]*\}/);
  assert.ok(headingRule, 'expected styles for section headings');
  assert.doesNotMatch(headingRule[0], /position\s*:\s*(absolute|fixed)/);
});

test('the assistant resolves the homepage topic parameter', () => {
  const helper = chatSource.match(
    /function resolveRequestedAgent\(params, agents\) \{[\s\S]*?\n    \}/,
  );
  assert.ok(helper, 'expected chat.html to resolve the requested assistant');

  const context = { URLSearchParams };
  vm.runInNewContext(
    `${helper[0]}\nglobalThis.resolveRequestedAgent = resolveRequestedAgent;`,
    context,
  );
  const agents = [
    { id: 'hr' },
    { id: 'it' },
    { id: 'policy' },
  ];
  const resolve = query => context.resolveRequestedAgent(new URLSearchParams(query), agents);

  assert.equal(resolve('topic=hr').id, 'hr');
  assert.equal(resolve('topic=it').id, 'it');
  assert.equal(resolve('topic=policy').id, 'policy');
  // Existing entry points keep working unchanged.
  assert.equal(resolve('agent=it&embedded=1').id, 'it');
  assert.equal(resolve('agent=policy').id, 'policy');
  // Unknown or missing hints start automatic routing instead of forcing HR.
  assert.equal(resolve('topic=finance'), null);
  assert.equal(resolve(''), null);
  assert.equal(resolve('agent=bogus&topic=policy').id, 'policy');
});

test('the homepage topic links are wired into the assistant page', () => {
  assert.match(chatSource, /let activeAgent = resolveRequestedAgent\(chatParams, AGENTS\);/);
  assert.match(chatSource, /url\.searchParams\.delete\('topic'\)/);
});