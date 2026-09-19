# 0912 Experiment Dashboard Design

## Purpose

An operational, read-only monitor for the current Nav2 experiments. The page must make
dataset, method, progress, provisional score, and role-call latency comparable at a glance.

## Tokens

- Canvas `#090b10`, panel `#11141b`, raised `#181c25`
- Text `#f3f5f7`, muted `#9299a6`, border `#272d38`
- Accent `#8b87ff`, success `#49d49d`, running `#62a8ff`, warning `#f0b95b`
- Spacing uses a 4px base; radii are 8px and 12px.
- UI uses the system sans stack; numeric results use the system monospace stack.

## Layout

The document owns vertical scrolling. A bounded header is followed by summary cards and
one horizontally scrollable comparison table. At narrow widths rows become labeled cards.

## Components and states

- Summary card: running jobs, completed cases, last refresh.
- Method row: dataset, variant, state, progress, score, correct count, role-call time.
- Status chip: running, completed, or partial; text always accompanies color.
- Refresh control: manual refresh alongside automatic two-second polling.

## Accessibility and motion

Semantic landmarks and tables, visible focus, AA contrast, and explicit progress text are
required. Only control-color transitions are used and are disabled for reduced motion.

