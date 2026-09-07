# Mamba 130M Visualizer Design System

This is an engineering explorer, not a landing page. Visual values are tokens, layout responsibilities have names, and every state keeps the same data contract accessible.

## Current selection inspector contract (supersedes scrolling/full-default rules below)

- **Bounded list-detail shell:** `block-size:100dvb`, topbar `auto`, workspace `minmax(0,1fr)`. Desktop has a fluid LEFT explorer and a RIGHT `--inspector-width-wide:360px` utility pane. Explorer and inspector each own local vertical scroll. The document never grows with tensor inventory. Pattern: [list-detail](https://raw.githubusercontent.com/changeroa/StyleGallery/main/patterns/split-sidebar/list-detail.md).
- **Compact selection summary:** selected kind, exact ID, scenario and a short Korean role explanation, followed by separate Inputs, Outputs, Weights and relevant State groups. Each boundary keeps its actual total count; preview at most two main activation tensors (index inputs only when there is no activation), with state/index counts explicitly distinguished. Preview values are original name, native `ne[4]`, dtype and compact logical bytes. Weights show actual count and unique GGUF payload bytes, never intermediate/peak-memory claims. Model summaries never borrow a block selection; global weights and state counts describe the model, not block 0.
- **Progressive full detail:** a closed native `고급 정보` disclosure retains the original full inspector, all 26 fields, calculator and state/provenance evidence. Its content owns a bounded `--evidence-height:320px` scroll region. It is built on first opening for each selection; selection resets it closed. The historical all-fields-visible requirement applies inside this disclosure, not to default content.
- **Compact frame:** at widths up to 1024px the RIGHT pane becomes a native modal dialog drawer, at most 360px wide with a 12px outside gutter. It overlays, never narrows, the four-card architecture. Only initial model-root landing leaves it closed with a native details button. Entity selection, selected-layer deep links/reload, and history navigation open it automatically (including reselection); desktop selection only updates its visible pane. Scenario buttons update the selected summary without forcing a closed drawer open.
- **Drawer accessibility:** native dialog semantics, visible close and reopen buttons, Escape dismissal, initial focus on close, return focus to the selected navigation action (representative card after block choice). Entity selection resets summary scrolling after the drawer is shown, so its identity and role remain the entry point even after a previously scrolled summary was closed. No custom focus loop or scroll trap. Resize releases modality and preserves the same inspector DOM. Source for the side-panel/escape mechanism: [beui drawer](https://beui.dev/r/drawer/raw); native dialog supplies modality here, with no animation/dependency.
- **Preserved flow:** four horizontal model cards, representative disclosure, 24 real block choices, then one selected block's eight horizontal stages and actual operators. Reel and bounded evidence/picker/list regions keep their existing local overflow jobs. At 1440×900 use 20px gutters; 768×1024 and 390×844 use 12px. At compact widths preserve existing model-card proportions without an adjacent squeezing column.
- **States and verification:** summary empty/populated/many, advanced closed/open, desktop pane, mobile drawer closed/open/reopened/Escape, scenario/reload, current/ancestor/hover/independent focus. Persistent selected rails extend to root/block/search selections; accessible names contain visible native labels. Actual-data summary tests and a browser summary oracle complement the unchanged 2,760-entity full-detail oracle, which intentionally opens advanced evidence before checking fields.
- **Review ownership:** no accepted accessibility debt. Parent performs independent review against fresh 1440×900, 768×1024 and 390×844 Chrome captures and geometry/row-count evidence. Older geometry and full-default requirements below are historical, not competing instructions.

## Current structural contract — model architecture overview

The user's horizontal-stage request supersedes the older three-column hierarchy/vertical-flow layouts described in the historical task references below. Preserve the existing dark blueprint palette, typography, tensor fields, capture identities, and exact routes; this is a structural change only.

- **Search first:** operator / tensor search is the first region in the explorer, ahead of breadcrumbs and the model diagram. The global scenario header stays above the workspace.
- **Four-node model overview:** both no-hash landing and `#scenario=prefill&entity=mamba-130m` select MODEL ROOT. Below search and breadcrumbs, a prominent ordered architecture diagram shows `Embedding → Mamba block × 24 → Final Norm → Output`. The second card is one native representative disclosure button, not a small header above preselected stages. Architecture arrows communicate semantic model order, not captured GGML nodes or tensor edges. The three other cards select actual `model/embedding`, `model/final-normalization`, and `model/final-projection` entities.
- **Selection before detail:** root initially shows zero block stage cards, zero chosen blocks, and the actual model inspector. Opening the representative exposes all 24 actual `block.0` through `block.23` choices in a bounded region without changing the model selection or rendering block stages. Choosing a block selects that exact capture entity, closes choices, and returns focus to the representative card. No implicit first-block fallback, shared weights/state, or 24 cloned stage graphs.
- **Persistent navigation:** all four architecture cards remain above selected block/stage/operator detail. A model return button clears block stages and operators. Model-wide stage cards show only their own actual operators and inspector; neither they nor root inherit prior block detail. The obsolete model-context disclosure is removed. Exact model/block/stage/operator hashes, scenario preservation, nearest-ancestor fallback, and browser history retain their original meanings.
- **Eight-stage reel:** the active block's actual stage children appear once, in source order, left to right. Number, readable stage name, exact stage ID, and operator count distinguish each card. Selection of a stage or its operator retains this common outline and all eight stages; operators expand below, not in place of the reel. Deep links and history reveal the selected stage inside the reel without moving document scroll or stealing focus. The ordinal arrows show stage order, not invented tensor edges.
- **Progressive evidence:** the original SVG and equivalent full semantic node/edge list live in a closed-by-default native disclosure. Block nodes display their exact ID (`block.1`), without a redundant kind prefix. Opening the disclosure exposes every original edge role, tensor identity, and endpoint without aggregation changes. Its viewport and list each have a bounded evidence-scroll job, so opening it cannot place the inspector thousands of pixels below the flow.
- **Scroll ownership:** the document owns vertical workspace scrolling at all widths. There is no left sidebar or inspector column squeezing the stage band. The stage reel alone owns horizontal stage scrolling, the expanded block picker owns its bounded choices scroll, and expanded graph evidence owns its bounded graph/list scroll. Every fluid parent has `min-inline-size:0`; document horizontal overflow is forbidden. Inspector follows the compact explorer and has a direct jump link.
- **Reel source:** [StyleGallery reel](https://raw.githubusercontent.com/changeroa/StyleGallery/main/patterns/in-line-grouping/reel.md). Adopt `grid-auto-flow:column` with local overflow, source/focus order matching the visible flow, and readable minimum card width. Native Tab access plus Left/Right/Home/End moves focus between stage cards and scrolls only the reel. No automatic scaling of eight cards into tiny labels; no auto-play or animated scroll.
- **Responsive geometry:** at `1440×900`, use 20px workspace gutters; at `768×1024` and `390×844`, use 12px gutters. The four model cards remain left-to-right and fully visible in the first viewport, with `--model-card-height:112px`, flexible tracks, wrapping labels, and `--space-4` arrow gaps (`--space-3` on compact screens). Compact model cards use `1.3 / 1.4 / 0.85 / 0.85` flexible track proportions to fit Embedding without splitting its word, body-compact type, and `--space-1` insets; desktop uses section type and `--space-3` insets. Accessible names follow the visible native button content; exact entity kinds and IDs remain in `aria-description`, routes, and selected detail rather than overriding visible labels. Block stage cards use `--stage-card-width:160px` as a readable minimum and share extra width when all fit. `--evidence-height:320px` bounds disclosed graph/list and choice/result regions; `--operator-list-height:192px` bounds the operator list. These dimensions supplement the existing 4px spacing tokens.
- **Primitive states:** model architecture card default/current/ancestor/hover/focused, representative card collapsed/expanded/focused, block choice current/hover/focused, stage card current/ancestor/hover/focused, operator list selected/empty, search empty/results, evidence closed/open, scenario and loading/error remain native labeled states. Current model, stage, and operator cards use a persistent solid `--space-1` accent top rail, distinct from the thin hover border and separate keyboard focus outline. A collapsed picker and evidence disclosure are intentional progressive access, never deleted data. Keyboard users and engineers comparing block-specific state can complete the same path as pointer users.
- **Verification:** production-browser coverage includes initial overview, expanded 24 choices, actual block selection, stage/operator context retention, bounded mobile scrolling, search, history/scenario/reload, semantic SVG/list equality, and the unchanged 2,760-entity inspector oracle. New prose is reviewed, not pinned by tests. Parent owns independent visual review; no accessibility debt is accepted silently.

The sections below retain the original design research and evidence contracts. Historical three-column shell geometry is superseded by the current structural contract above.

## 0. Research Log

- Existing UI audit: the worktree has `index.html`, `src/schema.ts`, one data-contract test, and build configuration only; there are no reusable UI components, CSS tokens, or existing visual patterns to extract. This document establishes the first system before product screens.
- Embedded references: shortlisted `together.ai.md` (technical blueprint / mono labels), `vercel.md` (precision / monochrome), and `ollama.md` (terminal-native restraint); picked the technical-blueprint discipline from `together.ai.md` with the operational restraint of `taste-skill.md`, because this is a dense model-inspection tool rather than marketing.
- Layout references: read local `layout-skill.md` plus StyleGallery `CATALOG.md`, `GUIDE.md`, `layout/index.md`, then fetched `fixed-sidenav-shell`, `list-detail`, `main-with-rail`, `panel-layout`, `dense-grid`, and `breadcrumb` from `https://raw.githubusercontent.com/changeroa/StyleGallery/main/`. Adopted named shell, list-detail, panel, dense-grid, and breadcrumb responsibilities; the source is linked, not copied.
- Lazyweb: 2 read-only searches (desktop model/graph explorer; mobile developer/data inspector), 6 result screens downloaded and 4 directly viewed. Findings: persistent context rails, explicit search, compact detail sections, a single-column mobile reading order, and stateful controls are more useful than dashboard decoration. No screenshots are shipped or copied.
- UI/UX DB: 1 targeted HTML/Tailwind search for technical data explorer guidance; it returned the accessible icon-button rule. The implementation remains vanilla TypeScript and native HTML/CSS.
- Designpowers: read routing, orchestration, direction, and review guidance. Personas and accessibility decisions below are expressed as design constraints; no separate design-state file is created because task scope permits only `DESIGN.md` and task-3 evidence.
- Visual QA: read the `/visual-qa` skill. Because this task writes no product surface, evidence uses deterministic static SVG references and geometry/state assertions rather than claiming a browser-rendered product pass. Task 11 owns fresh browser screenshots and dual-oracle review.
- Imagen drafts: skipped; no image-generation tool is available, and a technical explorer has no need for decorative imagery.
- React Dev Tooling Gate: N/A by explicit plan contract. The planned site is strict vanilla TypeScript with no React dependency; do not install React, react-grab, react-scan, or react-doctor.

## 1. Atmosphere & Identity

The surface is a quiet night-time instrument panel: a midnight-blue canvas, fine blueprint lines, sharp containment, and high-information typography. The signature is **boundary tracing**: input, output, weight, and recurrent-state edges use stable line styles and labeled shapes, while the selected entity is marked by one acid-lime focus rail. It should feel inspectable and honest, never like a product pitch.

Design read: an operational CPU graph explorer for Korean-speaking engineers and researchers, with English API/operator/dtype/tensor identifiers preserved exactly. Visual variance is low, density is medium-high, and motion is restrained. There is no hero, logo wall, pricing block, decorative gradient, fake metric, or marketing CTA.

Content register:

- Korean explains purpose, state, and consequences: `선택한 연산의 경계 텐서`.
- English remains exact for API, operator, dtype, tensor name, shape formula, and storage identifiers: `SSM_SCAN`, `F32`, `ggmlNbytes`.
- Never translate or invent a tensor value. Missing, unknown, or unobserved values use an explicit status, not zero.

## 2. Color

One dark theme is intentional. No section flips to a light theme. The palette is derived from the blueprint reference but narrows the accent to an interactive lime; flow colors are semantic marks and are never the only encoding.

### Palette tokens

| Role | Token | Value | Use |
|---|---|---:|---|
| Canvas | `--color-canvas` | `#08131B` | App background and graph field |
| Surface | `--color-surface` | `#0E1D27` | Rail, flow, inspector panels |
| Surface raised | `--color-surface-raised` | `#142832` | Selected row, table header, inline status |
| Border | `--color-border` | `#3D545F` | Panel and table boundaries |
| Text primary | `--color-text-primary` | `#F4F7F8` | Headings, tensor names, values |
| Text secondary | `--color-text-secondary` | `#B7C4CA` | Korean explanation and field labels |
| Text tertiary | `--color-text-tertiary` | `#8EA2AB` | Hints and non-selected metadata; never required alone |
| Interactive accent | `--color-accent` | `#B8F23D` | Links, selected rail, focus ring, primary action |
| Input flow | `--color-flow-input` | `#5FD7FF` | Input edge/marker plus the word `input` |
| Output flow | `--color-flow-output` | `#FFB565` | Output edge/marker plus the word `output` |
| State flow | `--color-flow-state` | `#D6A4FF` | Recurrent state edge/marker plus `state` |
| Weight flow | `--color-flow-weight` | `#82E4B1` | Weight edge/marker plus `weight` |
| Warning | `--color-status-warning` | `#FFD166` | Recoverable load or storage warning |
| Error | `--color-status-error` | `#FF7D87` | Invalid data / blocked rendering |

Contrast checks are against the canvas and use the WCAG relative-luminance calculation, not visual estimation: primary `17.43:1`, secondary `10.51:1`, tertiary `7.06:1`, accent dark text `14.12:1`, warning dark text `13.01:1`, error dark text `7.63:1`, input dark text `11.30:1`, output dark text `10.77:1`, state dark text `9.48:1`, and weight dark text `12.22:1`. The evidence receipt records the exact command and result.

Rules:

- Accent is reserved for interactive and selection feedback. Flow colors may mark edges, but each edge also has a text label, line style, and node shape.
- Do not add raw colors in product CSS. Extend this table first.
- Do not use opacity to make required text pass; use a token with measured contrast.
- Errors use icon/label/message, not color alone. Warnings use the same rule.

## 3. Typography

The type pair is `IBM Plex Sans, "Noto Sans KR", system-ui, sans-serif` for Korean explanation and `IBM Plex Mono, "SFMono-Regular", Consolas, monospace` for identifiers and measurements. Fonts must be self-hosted or use the declared fallback; the app must not fetch a remote font at runtime. Two families only: sans for reading and mono for technical identity.

| Level | Token | Size / line-height | Weight | Tracking | Use |
|---|---|---:|---:|---:|---|
| Page title | `--type-title` | `clamp(20px, 2vw, 28px) / 1.15` | 500 | `-0.02em` | Selected entity title; 28px at wide desktop, 20px at compact widths |
| Section title | `--type-section` | `18px / 1.3` | 500 | `-0.01em` | Tensor section and state panel |
| Body | `--type-body` | `15px / 1.5` | 400 | `0` | Korean explanation |
| Body compact | `--type-body-compact` | `14px / 1.45` | 400 | `0` | Rail rows and helper copy |
| Metadata | `--type-meta` | `14px / 1.4` | 450 | `0.01em` | Required field labels and values |
| Identifier | `--type-code` | `14px / 1.4` | 450 | `0` | IDs, dtype, shape, formula |
| Caption | `--type-caption` | `12px / 1.35` | 500 | `0.06em` | Uppercase structural labels |
| Touch label | `--type-touch` | `14px / 1.35` | 500 | `0` | Button and segmented control text |

Body, required metadata, and identifiers are always at least `14px`; only the non-essential uppercase caption token may be `12px`. The page title uses the explicit responsive `clamp()` token above, so the compact references use 20px rather than an undocumented one-off. Long Korean copy wraps naturally; identifiers use `overflow-wrap:anywhere` and never force viewport overflow.

## 4. Spacing & Layout

### Tokens

All authored spacing uses a 4px base. Component dimensions are named tokens so implementation does not invent one-off values.

| Token | Value | Use |
|---|---:|---|
| `--space-1` | `4px` | Icon-to-label, edge marker gap |
| `--space-2` | `8px` | Compact row and badge gap |
| `--space-3` | `12px` | Input inset, mobile gutter |
| `--space-4` | `16px` | Panel gap, standard card inset |
| `--space-5` | `20px` | Desktop outer gutter |
| `--space-6` | `24px` | Section inset and rail grouping |
| `--space-8` | `32px` | Major inspector section gap |
| `--space-10` | `40px` | Wide empty-state breathing room |
| `--control-height` | `44px` | Search and scenario action region |
| `--action-target` | `44px` | Visible hit area for every button/link row |
| `--touch-target` | `44px` | Minimum keyboard/touch action target |
| `--shell-topbar` | `56px` | Fixed global header |
| `--rail-width-wide` | `232px` | Desktop hierarchy rail |
| `--rail-width-medium` | `208px` | Tablet hierarchy rail |
| `--inspector-width-wide` | `360px` | Desktop inspector |
| `--skeleton-flow-node-width` | `280px` | Maximum loading flow-node width |

### Named shell and scroll ownership

Use a bounded `fixed-sidenav-shell`-style app shell with a topbar and three named regions. The shell uses `min-block-size: 100dvb`, a grid body with `min-block-size: 0`, and never `100vh`.

1. **Hierarchy rail (`nav`)**: desktop column one, width `--rail-width-wide`; owns `hierarchy-scroll` for the model/block/stage/operator list and search results. It stays fixed while the body is bounded.
2. **Central flow (`main`)**: desktop column two, `minmax(0, 1fr)`; owns `flow-scroll` for breadcrumb, scenario control, graph/list, and explanation. The SVG and its list share the same scroll region.
3. **Inspector (`aside`)**: desktop column three, width `--inspector-width-wide`; owns `inspector-scroll` for the selected entity and its three tensor sections. Its heading and selected-entity summary remain sticky within this region only.

The desktop body geometry at `1440x900` is: `20px` outer gutters, `232px` rail, `16px` inter-panel gap, fluid flow, `16px` inter-panel gap, `360px` inspector. The exact arithmetic is `1440 - 40 - 232 - 32 - 360 = 776px` for the central flow track. The topbar consumes `56px`; the remaining `844px` is the bounded shell body. This leaves a minimum `768px` central flow track at the target width before inner padding.

At `768x1024`, use `12px` gutters and `--rail-width-medium`; the inspector becomes a full-width region below the flow in the same document order. The single tablet vertical scroll owner is `workspace-scroll` for the entire body; rail, flow, and inspector are not independent scroll containers at this width. Do not label any tablet subpanel as a scroll owner.

At `390x844`, use one column and `12px` gutters in this order: topbar, breadcrumb/scenario, hierarchy drill-down, central flow, inspector. Only `workspace-scroll` scrolls. Do not hide inspector fields behind a desktop-only column. The graph becomes a vertical flow; the equivalent semantic list is always reachable. The scenario control retains both `PREFILL` and `DECODE` actions. The selected ancestor path is the same in every region: `mamba-130m` ancestor -> `BLOCK 00` ancestor -> `selective scan` stage -> `SSM_SCAN` operator selected.

### Responsive viewport contract

| Viewport | Shell | Rail / flow / inspector | QA intent |
|---|---|---|---|
| `1440x900` | Three-column bounded shell | `232px / fluid / 360px` | Full desktop rail, flow, inspector visible |
| `768x1024` | Rail + stacked content | `208px / fluid`, inspector below | No clipped Korean, one document scroll owner |
| `390x844` | Single-column drill-down | `nav -> main -> aside` | Same actions/data, no viewport overflow |

Use `minmax(min(16rem, 100%), 1fr)` for repeating grids and `min-inline-size: 0` on every fluid child. Every visible search, scenario, hierarchy, graph-list, retry, and clear action has a visible `--action-target` x `--action-target` region; a label may sit inside it, but the hit area is not a 32px or 40px strip. DOM order is the reading and focus order. No visual reorder is allowed to hide the hierarchy path.

## 5. Components

The primitive showcase is a required implementation gate: task 8/9 must first render these primitives in a state harness at desktop, tablet, and mobile sizes. Product screens may only compose primitives after the harness covers default, hover, active, focus, disabled, loading, empty, and error states.

### App shell chrome

- **Structure**: `header` -> `nav` + `main` -> `nav`, `main`, `aside`.
- **Variants**: desktop three-column, tablet rail-plus-stack, mobile single-column.
- **Spacing**: `--shell-topbar`, `--space-3`, `--space-4`, and named gutters.
- **States**: loading blocks graph and shows skeleton regions; invalid-data blocks product regions; normal shell preserves scenario and selection.
- **Accessibility**: landmarks are named; skip link targets `main`; all regions keep DOM order.
- **Motion**: none for shell reflow; opacity-only state announcement at `--motion-standard` when motion is allowed.

### Hierarchy row and drill-down

- **Structure**: `nav` containing an ordered list of buttons/links; each row has kind marker, exact ID/name, and selected indicator.
- **Variants**: model, block, stage, operator; selected, ancestor, unavailable-in-scenario.
- **Spacing**: `--space-2` row gap, `--touch-target` minimum action height.
- **States**: default, hover, active/pressed, visible focus, disabled/unavailable, loading skeleton, empty search, invalid selection.
- **Accessibility**: accessible name includes kind and exact identifier; `aria-current`/`aria-selected` is used only on the appropriate widget; Enter selects, Escape clears search.
- **Motion**: selection indicator may translate/opacity transition; reduced motion is static.

### Breadcrumb and scenario switch

- **Structure**: wrapping `nav[aria-label="Breadcrumb"]` followed by a labeled `fieldset`/segmented control for `prefill` and `decode`; each segment exposes a 44px action region.
- **Variants**: model, block, stage, operator depth; prefill selected, decode selected.
- **States**: keyboard focus, pressed selection, unavailable scenario, loading scenario, invalid scenario.
- **Geometry**: the outer control is `--control-height`; both scenario buttons independently meet `--action-target`, including on mobile.
- **Accessibility**: labels are visible; scenario is not conveyed only by color; switching preserves the nearest valid entity and announces that change.

### Flow node and edge legend

- **Structure**: `figure` with an SVG flow and a text legend; each node maps to an actual entity/operator ID.
- **Variants**: input, output, weight, state, selected, metadata-only/view alias.
- **States**: default, hover, focus, selected, unavailable, loading skeleton, invalid/missing edge.
- **Accessibility**: SVG has a title/description; the semantic keyboard list is adjacent and equivalent. Each edge has label + line pattern + marker shape; color is supporting information only.
- **Motion**: no auto-play or decorative movement; selection may use a 120ms opacity/transform transition.

### Tensor section and tensor row

- **Structure**: `section` with heading and an accessible table on wide screens; each mobile row becomes a labeled definition group without dropping fields.
- **Variants**: `inputs`, `outputs`, `weights`; populated, empty, mixed dtype, alias/view, state mutation.
- **States**: default, selected/focused tensor, loading skeleton, empty weights, invalid tensor, unknown storage.
- **Accessibility**: table headers remain programmatically associated; long names wrap; Korean explanation stays adjacent to exact identifiers; `caption` names the section.
- **Motion**: no row animation; section reveal is instant under reduced motion.

### Status banner and search field

- **Structure**: inline `role="status"` for loading/selection confirmation and `role="alert"` for invalid data; labeled search input with clear action.
- **Variants**: loading, error, warning, empty result, normal.
- **States**: focus, active, disabled while loading, clearable, error.
- **Accessibility**: status text is not color-only, errors explain next action, search result count is announced without stealing focus.
- **Motion**: no spinner; skeleton uses a static block when reduced motion is enabled.

### Primitive state matrix

| Primitive | Default | Hover | Active | Focus | Disabled | Loading | Empty | Error |
|---|---|---|---|---|---|---|---|---|
| Shell chrome | Regions present | N/A | Scenario action | Skip-link ring | N/A | Skeleton regions | N/A | Blocking banner |
| Hierarchy row | Readable row | Raised surface | Accent rail | 2px ring | Muted + reason | Row skeleton | Search-empty panel | Invalid-ID banner |
| Scenario switch | Current scenario | Raised segment | Pressed segment | Ring | Unavailable segment | Disabled + status | N/A | Scenario error |
| Flow node/list | Labeled node | Raised node | Selected node | Node ring | Unavailable node | Placeholder node | No graph data | Invalid graph |
| Tensor section | 3 labeled sections | N/A | Open/selected row | Row ring | N/A | Row skeletons | `0 tensors` explanation | Invalid tensor |
| Search/status | Input + count | Clear affordance | Submit/clear | Input ring | Disabled reason | Loading status | No matches | Load failure |

## 6. Motion & Interaction

Motion communicates selection or load progress only. There is no looping animation and no graph auto-pan.

| Token | Value | Use |
|---|---:|---|
| `--motion-micro` | `120ms` | Focus/pressed indicator |
| `--motion-standard` | `200ms` | Scenario/selection transition |
| `--motion-ease` | `ease-out` | Opacity and transform only |

Rules:

- Animate only `opacity` and `transform`; never animate layout, height, or scroll position.
- `prefers-reduced-motion: reduce` disables all non-essential transitions and shimmer. State changes remain immediate and fully labeled.
- Loading is a stable skeleton matching final geometry, not a timing-dependent spinner.
- Browser back/forward changes selection and scenario without a delayed visual race. Subscribe to the navigation state before triggering changes.
- Focus is never removed after search, scenario switch, error, or selection. A status message may announce the result.

## 7. Depth & Surface

The surface strategy is **tonal shift plus restrained borders**, not glass. Canvas, surface, and raised surface separate hierarchy; a 1px `--color-border` boundary makes dense tables scannable. No generic black shadow and no blur are needed.

- Level 0: canvas, no border.
- Level 1: surface panel, 1px border, no shadow.
- Level 2: raised selected row or popover, 1px border plus a dark-blue-tinted shadow only when it floats above scroll content.
- Selected: raised surface plus accent rail/focus ring; never an accent-colored full card.
- Radius scale: `--radius-control: 4px`, `--radius-panel: 8px`; no pills except the two-segment scenario control if needed for a clear grouping. Do not mix arbitrary radii.

## 8. Accessibility Constraints & Accepted Debt

### Constraints

- Target WCAG 2.2 AA: body and required metadata text meet `4.5:1`; large text meets `3:1`; focus indicators are visible against adjacent surfaces.
- Full keyboard reachability: skip link, search, scenario switch, all hierarchy rows, graph/list nodes, tensor rows, back/forward-compatible links, and error actions are tabbable in DOM order.
- The SVG is never the sole representation. The same entity/edge ordering is available as semantic HTML list/table content.
- Color is never the only state signal. Every flow edge combines color with label, line pattern, and marker shape; selected rows combine accent with text/rail.
- Korean text must wrap without clipping or orphaning a particle/connective. Exact English identifiers may break at punctuation or anywhere, but remain copyable.
- Long tensor names, unbroken hashes, shape formulas, and size strings use safe wrapping; no primary viewport horizontal scroll.
- Screen-reader labels name hierarchy kind, exact ID, scenario, tensor role, dtype, logical/native shape, numel, bytes, storage, alias, producer, and consumers.
- Reduced motion is respected; no essential information depends on animation, hover, or pointer precision.
- Touch targets are at least `--touch-target`; pointer hover is additive only.

### Data/inspector contract

At **model, block, stage, and operator** levels, render three separate sections in this order: `Inputs`, `Outputs`, `Weights`. Never merge them into one tensor list. A weight-free entity visibly renders `Weights — 0 tensors` with the explanation `이 항목은 직접 보유한 weight가 없습니다.`; it must not show fabricated zero bytes.

Each tensor row must expose all of these fields in desktop table columns and mobile labeled rows:

| Field | Required display rule |
|---|---|
| Exact identifier | Original `name`/`id`; copyable, wrapped safely |
| Dtype | Actual GGML dtype; no model-wide F32 default |
| Native shape / strides | `ne[4]` and `nb[4]` exactly |
| Logical shape | Axis labels and formula/value from the document |
| Element count | `numel`, separately labeled |
| Logical bytes | Compact payload bytes; IEC rendering next to exact bytes |
| Storage | `ggmlNbytes`, `requiredAllocBytes`, `bufferBytes`, view source/offset, or `unknown` with reason |
| Graph relations | Producer, consumers, alias/view relation, provenance, observed/derived status |

The section header also shows count and unique-range summary where the data contract provides it. Never sum intermediate tensors to claim peak memory. A view keeps its own dtype/shape and points to backing storage. Formula values are symbolic estimates unless the document marks them observed. When the inspector exceeds its bounded height, `inspector-scroll` continues below the fold; fields are not hidden or replaced with a claim that they exist.

### Full labeled inspector fixture

The static references use a **structure fixture**, not model data. Each example row is explicitly capture-bound so no synthetic tensor value is presented as observed. The implementation must render these labels and the value supplied by validated JSON:

| Example row | Required visible label and meaning |
|---|---|
| Input tensor | Exact original `name` and `id`; actual `dtype`; `role: activation/state/index` |
| Native layout | `ne[4]` = native element extents; `nb[4]` = native byte strides, each as an exact four-item value |
| Logical layout | `logical axes` = named axis order; `logical shape` = schema shape/formula; native and logical shapes remain separate |
| Counts and bytes | `numel` = exact element count; `logicalBytes` = compact payload bytes; `IEC` = derived binary display (`KiB`, `MiB`, etc.), never a replacement for exact bytes |
| GGUF source | `GGUF payload range` = unique source payload/range accounting, separate from runtime storage |
| Runtime span | `ggmlNbytes` = addressed tensor span, including stride gaps where applicable |
| Backend requirement | `requiredAllocBytes` = backend-required allocation size for a non-view; never described as a per-node heap slot |
| Backing capacity | `bufferBytes` = unique runtime backing-buffer capacity; not automatically a node allocation or peak |
| Allocator slot | `allocator slot capacity` = `unknown` with the reason that exact allocator slot capacity is not exposed |
| Relations | `viewSourceId` and `viewOffsetBytes` when present; producer; consumers; shared-range identity |
| Provenance | `captureId`; source/runtime identity; `observed` versus `derived` classification |

The fixture repeats the complete row anatomy under visible `Inputs`, `Outputs`, and `Weights` headings. The `Weights` heading remains visible even when its list is empty. A view example retains its own `dtype`, native/logical shape, `numel`, and byte fields while separately naming its backing source and offset. Desktop may continue this fixture below the inspector fold under `inspector-scroll`; mobile uses labeled row groups in the same order, not hidden columns.

### Accessibility and visual debt

| Item | Location | Why accepted | Owner / exit |
|---|---|---|---|
| None at design-contract time | Product surface | No product component is implemented in task 3, so no implementation debt is silently accepted | Task 8/9 must run the primitive showcase and task 11 must run browser/accessibility QA |

## 9. State and Content Contract

The following are product states, not optional polish. Each state keeps the shell landmarks and gives the user a next action.

| State | Layout | Required copy/action |
|---|---|---|
| Loading | Keep rail/flow/inspector geometry; skeleton rows preserve table height | `검증된 캡처 데이터를 불러오는 중입니다.`; no fake tensor values |
| Invalid data | Keep topbar and error banner; do not render graph or fallback fixture | `데이터 검증에 실패했습니다.` plus path/field and retry/reload action |
| Search empty | Keep search and breadcrumbs; replace list with bounded empty panel | `일치하는 operator 또는 tensor가 없습니다.` plus clear-search action |
| Selection | Accent rail on selected hierarchy row and selected node; inspector updates as one transaction | `선택됨` announcement with exact ID and scenario |
| Keyboard focus | 2px accent ring with offset on every control; no focus-only hidden content | Focus order follows DOM order |
| Weight-free | Render all three headings, explicit empty weight list | `0 tensors`; never hide the section |
| Mixed dtype | Per-row dtype chips/values; no global badge | Dtype text remains exact (`F32`, `F16`, `I32`, etc.) |
| Alias/view | Show view source and byte offset beside row | Explain view does not claim a new arithmetic allocation |
| State edge | Show state marker and before/after epoch labels when present | `state` label + line pattern + list item |
| Long content | Names wrap; values remain adjacent; tables become labeled row groups on mobile | No ellipsis on required identifiers; optional prose may wrap |
| Reduced motion | Same final layout, no shimmer/transition | `prefers-reduced-motion` is a CSS/media-state contract |

## 10. Flow and Information Architecture

Primary route is `model -> block -> stage -> operator`. Selection state is `(scenario, entityId)` and is hash-addressable. Search matches original operator/tensor names without changing the underlying hierarchy. A scenario switch keeps the selected ID when available; otherwise it selects the nearest valid ancestor and announces that change. Browser back/forward restores both values.

Central flow anatomy:

1. Breadcrumb and scenario switch.
2. One-line Korean explanation of the selected boundary.
3. SVG flow with labeled input/output/weight/state edges.
4. Equivalent ordered keyboard list containing the same nodes and edge roles.
5. Optional kernel-internal equation note, explicitly labeled `kernel-internal`, never a fake operator node.

Inspector anatomy:

1. Exact entity kind, ID, parent, and scenario.
2. `Inputs`, `Outputs`, `Weights` sections with the full tensor contract.
3. State/cache panel for observed before/after snapshots when present.
4. Provenance/validation link or status; no invented success state.

## 11. Implementation Guardrails

- Vanilla strict TypeScript, semantic HTML, native CSS, and SVG only. No React, graph framework, paid tooling, remote runtime service, or bundled model weights.
- Product CSS must define the tokens in this document and use them everywhere. No raw one-off hex, font size, spacing, shadow, or radius.
- The primitive showcase precedes product screens. Evidence-only SVGs in `.omo/evidence/mamba1-130m-visualizer/task-3/` are references, not product source.
- No browser model inference. The browser consumes only validated static metadata from the plan’s public JSON documents.
- No invented graph nodes for scalar internals: `SSM_SCAN` and `SSM_CONV` remain single operators, and equations are explanatory only.
- No external network request for fonts, images, analytics, or runtime data.

## 12. Verification Contract

Task 3 acceptance is satisfied when this document and the evidence directory make the following observable: named rail/flow/inspector geometry at `1440x900`, `768x1024`, and `390x844`; mobile one-column order; all tensor fields and three sections at every hierarchy level; loading/error/search-empty/selection/focus/reduced-motion layouts; semantic SVG/list equivalence; measured contrast tokens; and primitive showcase states.

Pure prose is intentionally not pinned by a text-equality test. Task 8/9 own implementation behavior. Task 11 owns real browser screenshots and visual-QA dual-oracle review at the three target viewports. This task’s static references are limited to geometry/state proof and carry no product code.
