const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function ui() {
  const elements = new Map();
  const defaults = { prompt: 'portrait', cfg: '1', quality: '95', refRes: '1024',
    steps: '25', format: 'png', seed: '', negative: '', loraNsfw: '1',
    loraPenis: '0', loraVagina: '0' };
  const element = () => ({ value: '', children: [], checked: false, appendChild(row) {
    this.children.push(row);
  }, set innerHTML(value) {
    this.children = value ? [{ value: '' }, { value: '' }, {}] : [];
  } });
  const get = (id) => {
    if (!elements.has(id)) {
      const el = element(); el.value = defaults[id] || ''; elements.set(id, el);
    }
    return elements.get(id);
  };
  const saved = new Map();
  const context = vm.createContext({
    document: { getElementById: get, addEventListener() {}, createElement: element },
    localStorage: { setItem: (k, v) => saved.set(k, v), getItem: (k) => saved.get(k) },
    Blob, setTimeout, clearTimeout,
  });
  const html = fs.readFileSync(path.join(__dirname, '../tools/webui/index.html'), 'utf8');
  // Execute the real application functions; leave startup timers/event wiring out.
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1].split('/* ---------- wiring ---------- */')[0];
  vm.runInContext(script + '\nrenderSizes = () => {}; updatePayloadSize = () => {};', context);
  const call = (expression) => {
    const result = vm.runInContext(`JSON.stringify(${expression})`, context);
    return result === undefined ? undefined : JSON.parse(result);
  };
  return { get, saved, call, context };
}

test('custom LoRA zero remains zero in the submitted payload', () => {
  const app = ui();
  app.get('loras').children = [{ children: [{ value: 'style.safetensors' }, { value: '0' }] }];
  assert.deepEqual(app.call('buildInput().loras'), [{ name: 'style.safetensors', strength: 0 }]);
});

test('submitted defaults and independently changed strengths reach the API', () => {
  const app = ui();
  assert.deepEqual(app.call('buildInput().lora_strengths'), { nsfw: 1, penis: 0, vagina: 0 });
  app.get('loraNsfw').value = '0';
  app.get('loraPenis').value = '0.65';
  app.get('loraVagina').value = '-0.2';
  assert.deepEqual(app.call('buildInput().lora_strengths'), { nsfw: 0, penis: 0.65, vagina: -0.2 });
});

test('strengths survive saving, restoring and reusing a job', () => {
  const app = ui();
  app.get('loraNsfw').value = '0';
  app.get('loraPenis').value = '0.65';
  vm.runInContext('saveForm()', app.context);
  app.get('loraNsfw').value = '1';
  app.get('loraPenis').value = '0';
  vm.runInContext('restoreForm()', app.context);
  assert.deepEqual(app.call('buildInput().lora_strengths'), { nsfw: 0, penis: 0.65, vagina: 0 });
  vm.runInContext('applyParams(inputToParams({prompt: "repeat", lora_strengths: {nsfw: 0.3, penis: 0, vagina: 1}}))', app.context);
  assert.deepEqual(app.call('buildInput().lora_strengths'), { nsfw: 0.3, penis: 0, vagina: 1 });
});

test('legacy builtin filenames migrate to dedicated controls without double application', () => {
  const app = ui();
  vm.runInContext('applyParams({loras: [{name: "NSFW Qwen by TheseAlpacas V2.safetensors", strength: 0.3}]})', app.context);
  assert.deepEqual(app.call('buildInput().lora_strengths'), { nsfw: 0.3, penis: 0, vagina: 0 });
  assert.equal(app.call('readLoras().length'), 0);
});
