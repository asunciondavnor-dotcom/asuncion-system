/* Prepare approved models through short, authenticated requests before submitting documents. */
(() => {
  'use strict';
  const selector = 'form[data-model-preparation]';
  const requested = new Set();
  let waiting = new WeakMap();
  const replaying = new WeakSet();

  function endpoint(form) {
    const url = new URL(form.dataset.modelPreparation, document.baseURI);
    url.search = '';
    return url.href;
  }

  function selectedOrder(form) {
    return form.querySelector('select[name="po_id"]')?.selectedOptions?.[0];
  }

  async function status(url, start, retry = true) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 60000);
    let response;
    try {
      response = await window.PSMSCSRF.fetch(url, {
        method: 'POST', body: new URLSearchParams({start: start ? 'yes' : 'no'}),
        headers: {'Accept': 'application/json'}, cache: 'no-store', signal: controller.signal,
      });
    } catch (error) {
      if (retry) return status(url, start, false);
      throw error;
    } finally {
      clearTimeout(timer);
    }
    if (retry && [502, 504].includes(response.status)) return status(url, start, false);
    if (!response.ok || response.redirected) throw new Error('Preparation unavailable');
    const {state} = await response.json();
    if (!['ready', 'preparing', 'idle'].includes(state)) throw new Error('Invalid preparation state');
    return state;
  }

  function notice(form) {
    let element = form.querySelector('[data-model-preparation-status]');
    if (!element) {
      element = document.createElement('p');
      element.dataset.modelPreparationStatus = '';
      element.className = 'notice';
      element.hidden = true;
      element.setAttribute('role', 'status');
      form.append(element);
    }
    return element;
  }

  function prepareAhead(form) {
    if (form.dataset.modelPreparationAction) {
      const order = selectedOrder(form);
      if (!order?.value || order.dataset.savedPrId) return;
    }
    const url = endpoint(form);
    if (requested.has(url)) return;
    requested.add(url);
    status(url, true).catch(() => requested.delete(url));
  }

  document.querySelectorAll(selector).forEach(prepareAhead);
  document.addEventListener('focusin', event => {
    const form = event.target.closest?.(selector);
    if (form) prepareAhead(form);
  });
  document.addEventListener('change', event => {
    const form = event.target.closest?.(selector);
    if (form) {
      const ticket = waiting.get(form);
      if (ticket) ticket.changed = true;
      else prepareAhead(form);
    }
  });

  document.addEventListener('submit', async event => {
    const form = event.target, button = event.submitter;
    if (!form.matches?.(selector) || replaying.has(form) || event.defaultPrevented) return;
    if (form.dataset.modelPreparationAction && button?.value !== form.dataset.modelPreparationAction) return;
    if (form.dataset.modelPreparationAction && selectedOrder(form)?.dataset.savedPrId) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (waiting.has(form)) return;
    const ticket = {changed: false};
    waiting.set(form, ticket);
    const message = notice(form), url = endpoint(form);
    try {
      let state = await status(url, true);
      if (state !== 'ready') {
        message.hidden = false;
        message.textContent = 'Preparing the approved model. Keep this page open.';
      }
      const deadline = Date.now() + 720000;
      while (state !== 'ready' && Date.now() < deadline && form.isConnected && waiting.get(form) === ticket) {
        await new Promise(resolve => setTimeout(resolve, 6000));
        state = await status(url, state === 'idle');
      }
      if (!form.isConnected || waiting.get(form) !== ticket) return;
      if (state !== 'ready') throw new Error('Preparation incomplete');
      if (ticket.changed) {
        message.hidden = false;
        message.textContent = 'The form changed. Review the current values and submit again.';
        return;
      }
      message.textContent = '';
      message.hidden = true;
      replaying.add(form);
      try {
        form.requestSubmit(button || undefined);
      } finally {
        replaying.delete(form);
      }
    } catch (_) {
      message.hidden = false;
      message.textContent = 'The model could not finish preparing. You can try submitting again.';
    } finally {
      if (waiting.get(form) === ticket) waiting.delete(form);
    }
  }, true);
  window.addEventListener('pageshow', () => { waiting = new WeakMap(); });
})();
