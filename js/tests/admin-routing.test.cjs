const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const appSource = fs.readFileSync(path.resolve(__dirname, '../app.js'), 'utf8');
const demoAdminHelper = appSource.match(/function isDemoAdminUser\(user\) \{[\s\S]*?\n\}/);
const homePageHelper = appSource.match(/function homePageForUser\(user\) \{[\s\S]*?\n\}/);
const sidebarRenderer = appSource.match(/function renderSidebar\(activePage\) \{[\s\S]*?\n\}/);

assert.ok(demoAdminHelper, 'expected the app to define its demo-admin identity helper');
assert.ok(homePageHelper, 'expected the app to define its home-page routing helper');
assert.ok(sidebarRenderer, 'expected the app to define its shared workspace navigation');

const routingContext = {
  APP: { short: 'N', name: 'Northstar' },
  AGENTS: [
    { id: 'hr', name: 'HR Agent', initial: 'HR' },
    { id: 'it', name: 'IT Agent', initial: 'IT' },
    { id: 'policy', name: 'Policy Agent', initial: 'PL' },
  ],
  appUser: null,
  location: { search: '' },
  URLSearchParams,
  chartIcon: () => '<svg></svg>',
  chatIcon: () => '<svg></svg>',
  docIcon: () => '<svg></svg>',
  escapeHTML: (value) => String(value),
};
vm.runInNewContext(
  `${demoAdminHelper[0]}\n${homePageHelper[0]}\n${sidebarRenderer[0]}\nglobalThis.homePageForUser = homePageForUser; globalThis.renderSidebar = renderSidebar;`,
  routingContext,
);

test('a real authenticated administrator opens the portal homepage', () => {
  assert.equal(
    routingContext.homePageForUser({ role: 'admin', username: 'local.admin@example.test' }),
    'portal.html',
  );
});

test('a real administrator can return from the workspace to the portal', () => {
  routingContext.appUser = { role: 'admin', username: 'local.admin@example.test' };
  const sidebar = routingContext.renderSidebar('documents');
  assert.match(sidebar, /href="portal\.html"[\s\S]*?<span>Employee portal<\/span>/);
  assert.match(sidebar, /href="documents\.html"[\s\S]*?<span>Documents<\/span>/);
  assert.match(sidebar, /href="analytics\.html"[\s\S]*?<span>Analytics<\/span>/);
  assert.match(sidebar, /href="chat\.html"[\s\S]*?<span>Chat<\/span>/);
});

test('the restricted demo administrator remains in the demo portal', () => {
  assert.equal(
    routingContext.homePageForUser({ role: 'admin', username: 'demo-admin@demo.invalid' }),
    'portal.html',
  );
  routingContext.appUser = { role: 'admin', username: 'demo-admin@demo.invalid' };
  const sidebar = routingContext.renderSidebar('chat');
  assert.match(sidebar, /href="portal\.html"/);
  assert.doesNotMatch(sidebar, /href="documents\.html"|href="analytics\.html"/);
});

test('employees remain in the employee portal', () => {
  assert.equal(
    routingContext.homePageForUser({ role: 'employee', username: 'employee@example.test' }),
    'portal.html',
  );
  routingContext.appUser = { role: 'employee', username: 'employee@example.test' };
  const sidebar = routingContext.renderSidebar('chat');
  assert.match(sidebar, /href="portal\.html"/);
  assert.doesNotMatch(sidebar, /href="documents\.html"|href="analytics\.html"/);
});
