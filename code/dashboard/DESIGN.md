# WSI LatentMAS Run Dashboard Design System

## 0. Research Log

- Embedded references: shortlisted Linear, Vercel, and Carbon; selected Linear's operational dark-console grammar for dense GPU status visibility.
- Lazyweb: skipped because the dashboard is an internal, live-status surface and current run state is the primary content reference.
- Imagen drafts: skipped because a status dashboard benefits from live DOM metrics rather than a fixed image-led composition.

## 1. Atmosphere & Identity

Quiet command center for long-running medical VLM experiments. The signature is a restrained dark field with thin luminance layers: live GPU activity should read first, while queued work remains deliberately quieter.

## 2. Color

| Role | Token | Value | Usage |
|---|---|---:|---|
| Canvas | `--canvas` | `#08090a` | Page background |
| Panel | `--panel` | `#0f1011` | Cards and tables |
| Raised | `--raised` | `#191a1b` | Header and hover surfaces |
| Primary text | `--text` | `#f7f8f8` | Key labels and values |
| Secondary text | `--muted` | `#8a8f98` | Metadata |
| Border | `--border` | `rgba(255,255,255,.08)` | Surface separation |
| Accent | `--accent` | `#7170ff` | Refresh and active controls |
| Accent soft | `--accent-soft` | `rgba(113,112,255,.14)` | Selected compact-control surface |
| Success | `--success` | `#34d399` | Completed work |
| Warning | `--warning` | `#fbbf24` | Queued work |
| Error | `--error` | `#fb7185` | Failed cases |

## 3. Typography

Primary: `Inter, ui-sans-serif, system-ui, sans-serif`; mono: `ui-monospace, SFMono-Regular, Menlo, monospace`. Display 28px/510; heading 18px/590; body 14px/400; caption 12px/510; tabular metrics use mono 13px.

## 4. Spacing & Layout

Base unit is 4px. Primary spacing tokens: 8, 12, 16, 20, 24, 32px. The shell is a single scroll owner with a 1200px maximum width; cards collapse to one column below 700px.

Compact controls use `--control-radius`, `--control-pad-y`, `--control-pad-x`, and `--caption-size`; `--space-1` represents the 4px base unit.

## 5. Components

### Status chip

- Structure: colored dot plus readable state label.
- Variants: running, queued, completed, failure.
- States: static semantic status; no decorative animation.
- Accessibility: text label never relies on color alone.

### GPU card

- Structure: device label, utilization/memory metrics, active workload label, progress bar.
- States: active or idle.
- Accessibility: progress has a text equivalent.

### Run row

- Structure: task name, state chip, queue order, progress fraction, success/failure counts, average seconds per question, completion ETA, queue-start wait, progress bar.
- States: active, queued, completed, degraded.
- Accessibility: readable table semantics on wide screens and card-style reflow on mobile.

### Quality-table filters

- Structure: a full-width Dataset group appears first, followed by compact Backbone, latent-step, and model-size groups.
- States: selected step uses the existing accent border and text; inactive options remain quiet.
- Accessibility: native buttons expose the selected state with `aria-pressed`; the current filter is stated in text.
- Scope: Dataset selection filters both the run-status table and quality table; GPU cards remain global hardware state.

### Quality comparison row

- Structure: task, parameter size, explicit latent-step value, speed, sample count, and all quality metrics.
- States: non-latent modes display an em dash for latent step; latent modes always expose their numeric step independently of the active filter.
- Accessibility: the desktop header and mobile card label both name `Latent step`, so filtering never hides the experiment configuration.
- Live method rows: newly launched variants appear immediately with current sample count, prefix quality, and comparable latency measured only from successful attempts on GPU 7 and 8.

## 6. Motion & Interaction

Refresh uses a 1-second polling cadence and a text timestamp. Buttons use 120ms opacity/transform feedback; reduced-motion users get no transition.

## 7. Depth & Surface

Tonal-shift with whisper-thin borders. Panels use `--panel`; hover/read emphasis uses `--raised`; no large drop shadows.

## 8. Accessibility Constraints & Accepted Debt

WCAG 2.2 AA contrast target, visible keyboard focus, semantic button and table markup, and reduced-motion support. Accepted debt: the local server is intentionally unauthenticated because it exposes only local run progress and should be bound to a trusted network.
