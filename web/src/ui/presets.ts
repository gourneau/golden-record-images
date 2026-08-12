/**
 * The four ways of reading the same signal, and the A/B wipe that puts any two
 * of them side by side.
 *
 * These are settings, not separate code paths: every preset is a patch over
 * DEFAULT_SETTINGS, so the dials always show exactly what produced the picture
 * and you can drag away from a preset without leaving it.
 *
 * The values in Raw and Barry 2017 come from what those decoders actually did
 * (see the resample-mode comments in pipeline/decode.py). The percentiles,
 * gamma and deconvolution strengths in Restored and Max detail are rendering
 * taste, not measurements.
 */

import type { DecoderSettings, SettingsSlot } from '../types';
import type { Explainer } from './dials';

export interface Preset {
  id: string;
  name: string;
  /** One line, shown in the legend window on hover. */
  note: string;
  settings: Partial<DecoderSettings>;
}

export const PRESETS: Preset[] = [
  {
    id: 'raw',
    name: 'Raw',
    note: 'Bin the samples, stretch to the full range, stop. No porch clamp, no filtering — the picture as the stylus found it.',
    settings: {
      height: 384,
      resample: 'box',
      dcRestore: false,
      uncouple: false,
      tau: 0,
      deconv: 0,
      invert: true,
      blackPct: 0,
      whitePct: 100,
      gamma: 1,
      transfer: 'linear',
      despike: 0,
      destripe: false,
    },
  },
  {
    id: 'barry',
    name: 'Barry 2017',
    note: "Ron Barry's rendering: area binning, a back-porch clamp per trace, and his cosine transfer curve.",
    settings: {
      height: 384,
      resample: 'area',
      dcRestore: true,
      uncouple: false,
      tau: 0,
      deconv: 0,
      invert: true,
      blackPct: 1,
      whitePct: 99,
      gamma: 1,
      transfer: 'barry',
      despike: 0,
      destripe: false,
    },
  },
  {
    id: 'restored',
    name: 'Restored',
    note: 'The full reference chain: Lanczos resampling, porch clamp, the recording chain’s high-pass undone, clicks removed, a light deconvolution.',
    settings: {
      height: 384,
      resample: 'lanczos3',
      dcRestore: true,
      uncouple: true,
      tau: 0,
      deconv: 0.6,
      deconvNoise: 0.02,
      invert: true,
      blackPct: 0.5,
      whitePct: 99.5,
      gamma: 1.05,
      transfer: 'linear',
      despike: 4,
      destripe: true,
    },
  },
  {
    id: 'maxdetail',
    name: 'Max detail',
    note: 'Past the point of comfort: 512 pixels per trace and full-strength deconvolution against the measured impulse response.',
    settings: {
      height: 512,
      resample: 'lanczos3',
      dcRestore: true,
      uncouple: true,
      tau: 0,
      deconv: 1,
      deconvNoise: 0.008,
      invert: true,
      blackPct: 0.2,
      whitePct: 99.8,
      gamma: 1,
      transfer: 'linear',
      despike: 5,
      destripe: true,
    },
  },
];

export function presetById(id: string | null): Preset | undefined {
  return PRESETS.find((p) => p.id === id);
}

export function presetName(id: string | null): string {
  return presetById(id)?.name ?? 'Custom';
}

// ---------------------------------------------------------------------------
// The bar
// ---------------------------------------------------------------------------

export interface PresetBarSpec {
  explainer?: Explainer;
  /** Apply a preset to the slot currently being edited. */
  onPreset: (id: string) => void;
  /** Change which slot the dials and preset buttons write to. */
  onSlot: (slot: SettingsSlot) => void;
  /** 1 = slot A fills the canvas, 0 = slot B fills it. */
  onWipe: (t: number) => void;
  /** Exchange the two slots, so the comparison can be read either way round. */
  onSwap: () => void;
}

export interface PresetBarHandle {
  el: HTMLElement;
  setSlot(slot: SettingsSlot): void;
  setWipe(t: number): void;
  /** Highlight the preset a slot currently matches, or null for Custom. */
  setActive(slot: SettingsSlot, id: string | null): void;
}

export function createPresetBar(spec: PresetBarSpec): PresetBarHandle {
  let slot: SettingsSlot = 'a';
  const active: Record<SettingsSlot, string | null> = { a: null, b: null };

  const el = document.createElement('div');
  el.className = 'presets';

  const set = document.createElement('div');
  set.className = 'presets__set';
  set.setAttribute('role', 'group');
  set.setAttribute('aria-label', 'Decoder presets');

  const buttons = PRESETS.map((p) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'preset';
    b.textContent = p.name;
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => spec.onPreset(p.id));
    b.addEventListener('pointerenter', () => spec.explainer?.show(p.name, p.note));
    b.addEventListener('focus', () => spec.explainer?.show(p.name, p.note));
    b.addEventListener('pointerleave', () => spec.explainer?.clear());
    b.addEventListener('blur', () => spec.explainer?.clear());
    set.append(b);
    return b;
  });

  const ab = document.createElement('div');
  ab.className = 'presets__ab';
  ab.innerHTML = `
    <span class="eng">A/B</span>
    <span class="slot" role="radiogroup" aria-label="Settings slot being edited">
      <button type="button" class="slot__opt" role="radio" data-slot="a" aria-checked="true">A</button>
      <button type="button" class="slot__opt" role="radio" data-slot="b" aria-checked="false">B</button>
    </span>
    <input class="wipebar" type="range" min="0" max="1" step="0.001" value="1"
           aria-label="A/B wipe position" />
    <button type="button" class="minibtn" data-act="swap">Swap</button>`;

  const slotButtons = Array.from(ab.querySelectorAll<HTMLButtonElement>('.slot__opt'));
  const wipeEl = ab.querySelector('.wipebar') as HTMLInputElement;
  const swapEl = ab.querySelector('[data-act="swap"]') as HTMLButtonElement;

  slotButtons.forEach((b) => {
    b.addEventListener('click', () => spec.onSlot(b.dataset.slot as SettingsSlot));
  });

  const slotNote =
    'Which of the two settings slots the dials and preset buttons write to. Drag the divider on the picture to compare them.';
  ab.addEventListener('pointerenter', () => spec.explainer?.show('A/B compare', slotNote));
  ab.addEventListener('focusin', () => spec.explainer?.show('A/B compare', slotNote));
  ab.addEventListener('pointerleave', () => spec.explainer?.clear());
  ab.addEventListener('focusout', () => spec.explainer?.clear());

  wipeEl.addEventListener('input', () => spec.onWipe(Number(wipeEl.value)));
  swapEl.addEventListener('click', () => spec.onSwap());

  el.append(set, ab);

  function paintActive(): void {
    PRESETS.forEach((p, i) => {
      buttons[i].setAttribute('aria-pressed', active[slot] === p.id ? 'true' : 'false');
    });
  }

  return {
    el,
    setSlot(next) {
      slot = next;
      slotButtons.forEach((b) => {
        const on = b.dataset.slot === next;
        b.setAttribute('aria-checked', on ? 'true' : 'false');
        b.tabIndex = on ? 0 : -1;
      });
      paintActive();
    },
    setWipe(t) {
      wipeEl.value = String(t);
    },
    setActive(which, id) {
      active[which] = id;
      paintActive();
    },
  };
}
