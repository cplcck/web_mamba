# mamba1-130m-visualizer - Work Plan

## TL;DR (For humans)
**What you'll get:** Mamba-1 130M의 실제 CPU 계산을 전체 모델부터 개별 연산까지 탐색하는 웹사이트입니다. 모든 계층에서 input, output, weight 각각의 dtype과 크기를 확인하고, prefill과 decode를 비교합니다.

**Why this approach:** 검증한 변환기와 CPU 코드를 유지하고 모델 revision을 기록합니다. 실제 실행 데이터를 원본 모델과 교차검증하므로 수식 그림을 실행 결과로 오해하지 않습니다.

**What it will NOT do:** 브라우저에서 모델을 실행하거나 모델 파일을 공개하지 않습니다. 자동 공개 배포나 커널 내부 scalar 연산을 가짜 graph node로 만드는 일도 하지 않습니다.

**Effort:** Large
**Risk:** Medium - 실제 graph와 재사용되는 상태·메모리를 정확히 연결하고 수치 차이를 설명해야 합니다.
**Decisions to sanity-check:** 기존에 선택한 CPU 소스 버전을 유지합니다. 모델은 시작 시점의 최신 revision을 고정하고, 사이트는 검증된 정적 결과만 제공합니다.

계획 작성과 실행은 별도입니다. 전용 계획 검토 결과를 확인한 후 구현을 시작합니다.

---

> TL;DR (machine): Large effort, medium risk; 12 implementation tasks in 4 waves, one authorized Git bootstrap task, plus 4 final verification domains; static vanilla TypeScript explorer backed by pinned CPU capture and independent HF validation.

### Execution authorization amendment
- After repository connection, the user explicitly approved the initial local baseline commit and verified local implementation commits with "전부 허용할게". This supersedes the original no-commit restriction and the historical `Commit: N` annotations below for local verified increments only.
- No push, PR, remote settings change, or deployment is authorized. The prior question explicitly excluded them.
- Planning is complete; `/ulw-execute` now authorizes implementation by delegated workers in task-owned worktrees. The root orchestrates, records evidence, and integrates verified commits.
- Task 13 is execution infrastructure discovered during bootstrap, not additional product scope. Existing user files stay untouched. Use per-command agent Git identity `OmO <omo@localhost>` if no author identity is configured; never change global Git configuration or impersonate the user.

## Scope
### Must have
- Build a static, Korean-language Mamba-1 130M CPU graph explorer in `/home/cplcck/web_mamba`. Keep English API, operator, dtype, and tensor identifiers exact.
- Navigate whole model -> individually selectable block -> stage -> actual GGML operator. At EVERY level display separate input, output, and weight lists. Each tensor has dtype, native and logical shape, element count, logical bytes, storage information, and IEC sizes. A weight-free entity explicitly shows an empty weight list rather than hiding the section.
- Use `/home/cplcck/llama.cpp-ssm` baseline `8144f3192e5a3131cd043f284525e6ceebf82d0f` for BOTH conversion code and CPU runtime. Do not update the fork or upstream. The verified capture route requires no product-source patch: use the existing eval callback and private recurrent-memory headers from the standalone tool.
- Resolve `state-spaces/mamba-130m-hf` main ONCE at execution start. Record the full resolved SHA; download the complete immutable snapshot locally and convert that directory with `--outtype f32`, WITHOUT `--remote`. Do not replace the newly resolved revision with the historical checkpoint merely to make tests pass.
- Record model/config/tokenizer payload hashes, GGUF SHA-256, converter/runtime commit and patch identity, binary hash, compiler/CMake flags, CPU, thread settings, Python/PyTorch/Transformers versions and lockfiles.
- Capture fresh prefill (`P=1,T=16,O=1`) and retained-cache decode (`P=1,T=1,O=1`) with a single fixed 17-ID fixture, sequence ID 0, positions 0..16, and no tokenization, sampling, padding, or implicit BOS/EOS. `P=n_seqs`, `T=n_seq_tokens`, `Q=P*T`, `O=n_outputs`.
- Capture per-layer Conv history (R) and SSM state (S) before/after both calls, plus active cell mapping. Compare fresh-17 and split-16+1 in llama.cpp and HF, and compare runtimes at aligned prefill/final output rows.
- Ship validated static metadata and validation summaries; raw weights, logits, activations, and full cache arrays remain local.
- Vanilla strict TypeScript, Vite, semantic HTML, CSS, and SVG; npm lockfile; Python offline tools; standalone C++ capture executable. Ponytail means minimal code/dependencies, not a product package. Caveman means concise copy, not loss of technical information.
- Prepare a GitHub Pages workflow and verify project-site base paths locally. No actual external write or deployment.

### Approved file layout and ownership
All paths below are relative to `/home/cplcck/web_mamba`, except the explicitly named llama.cpp source tree. Existing workspace currently contains only `.omo/`; recheck at implementation start.

```text
.omo/drafts/mamba1-130m-visualizer.md      existing planning decision/review ledger
.omo/plans/mamba1-130m-visualizer.md       this plan
.omo/evidence/mamba1-130m-visualizer/      task and final QA receipts
README.md                               commands, model provenance, scope
DESIGN.md                               UI tokens, hierarchy, states, responsive rules
.gitignore                              excludes local artifacts and generated output
package.json, package-lock.json          exact frontend/test dependency resolution
index.html, tsconfig.json, vite.config.ts
playwright.config.ts
src/main.ts                             bootstrap and data-load/error state
src/styles.css                          shared UI styles and responsive rules
src/schema.ts                           runtime data checks and TypeScript types
src/explorer.ts                         hierarchy/search/selection/scenario state
src/graph.ts                            SVG graph and equivalent keyboard list
src/inspector.ts                        inputs/outputs/weights and size display
public/data/provenance.json              immutable capture/build identity
public/data/model.json                   hierarchy and GGUF weight directory
public/data/prefill.json                 observed prefill graph/tensor metadata
public/data/decode.json                  observed decode graph/tensor metadata
public/data/validation.json              validation summary, not full vectors
fixtures/token-ids.json                  canonical fixed 17-ID fixture
fixtures/calibration-ids.json            predeclared additional fixed-ID fixtures
fixtures/numerical-policy.json           reviewed configuration-bound envelope
tools/requirements.txt                  exact Python dependency versions
tools/requirements.lock                 transitive versions and distribution hashes
tools/prepare_model.py                   revision resolution, download, conversion
tools/audit_gguf.py                      independent tensor mapping/value audit
tools/reference_hf.py                    CPU F32 fresh/cached HF runs
tools/validate.py                        structural/numeric comparisons
tools/export_web.py                      raw capture -> validated public JSON
tools/capture/CMakeLists.txt
tools/capture/main.cpp                   real llama.cpp batch execution and capture
tests/data.test.ts
tests/hierarchy.test.ts
tests/test_prepare_model.py
tests/test_audit_gguf.py
tests/test_validation.py
tests/test_export_web.py
tests/explorer.spec.ts
.github/workflows/pages.yml              manual deployment trigger only
.artifacts/hf/<resolved-sha>/            ignored full HF snapshot
.artifacts/gguf/                         ignored F32 GGUF
.artifacts/manifests/                    ignored file/binary/environment receipts
.artifacts/captures/                     ignored binary arrays + raw graph metadata
.artifacts/validation/                   ignored complete comparison reports
build/                                  ignored CMake outputs
test-results/                           ignored browser screenshots/reports
dist/                                   ignored deployable HTML/CSS/JS/data only
```

`DESIGN.md`, calibration/policy fixtures, and explicit Python tests/lock are refinements of the approved file sketch, required to make UI and numerical validation reproducible. They do not add product features.

### Data contract
- Root JSON documents carry `schemaVersion: 1`, `captureId`, and `ggufSha256`. A capture ID binds source revision, binary/patch identity, token fixture, settings, and artifact hashes. Mixing IDs or scenario metadata is an export/load error.
- An entity has stable `id`, `kind` (`model|block|stage|operator`), `parentId`, ordered `children`, `inputTensorIds`, `outputTensorIds`, and `weightTensorIds`. Parent inputs/outputs are boundary edges crossing its descendants; include residual and state edges, not just the first and last visible operation. Parent weights are the union of descendant weight references.
- A tensor records `id`, original `name`, `role` (`activation|weight|state|index`), native GGML `ne[4]` and byte strides `nb[4]`, logical axis labels and shape, actual `dtype`, `numel`, type/block size, `logicalBytes`, `ggmlNbytes`, `viewSourceId`, `viewOffsetBytes`, producer/consumer IDs, provenance, and observed/derived classification.
- Separate GGUF payload storage from runtime backing storage. Runtime records use buffer identity plus byte range and observation epoch; pointer addresses never become stable public IDs. UI storage fields explicitly distinguish addressed span (`ggmlNbytes`), backend required allocation size (`requiredAllocBytes` for non-views), and unique backing-buffer capacity (`bufferBytes`). Exact allocator slot capacity is not exposed and stays `null` with a reason; never present span or required bytes as actual per-node heap allocation or peak memory.
- Shared weights are deduplicated by actual payload/backing ranges, not name equality. Report logical references, unique GGUF payload bytes, and observed unique runtime ranges separately. Runtime duplication of tied weights is not automatically zero-cost.
- `logicalBytes` is compact element payload; `ggmlNbytes` may include stride gaps. Never sum intermediate tensors to claim peak memory. Views retain their own dtype/shape but point to backing storage. Mutation snapshots use distinct epochs to avoid presenting overwritten state as an earlier value.
- Formula display uses a closed representation: constants, dimension identifiers, addition, multiplication, and subtraction where nonnegative. No `eval` or arbitrary expression execution. Canonical scenario dimensions must reproduce exported shapes/bytes exactly. Any dimension calculator is labelled a symbolic estimate, not an observed run; storage allocation remains scenario-bound.
- Render all 24 actual blocks using shared UI code, not one invented block copied into all records. Preserve last-block output-row selection (`O=1`), which can change boundary shapes. Each operator gets one semantic stage owner; cross-stage edges remain visible.
- Stage vocabulary: normalization, input projection/split, convolution/history update, dt/B/C projection, selective scan/state update, skip/gating, output projection, residual. Embedding and final norm/projection appear at model level. Raw metadata operators (views, reshape, transpose, copies) remain inspectable.
- `SSM_SCAN` and `SSM_CONV` remain single GGML operators. Explanatory equations are `kernel-internal`, with no fake operator ID, tensor allocation, or captured scalar values. `A_log -> -exp(A_log)` is offline conversion, not an invented runtime A-construction EXP node.

### Verified capture contract
- Use public `llama_decode` calls, not a separately reconstructed graph or a reimplementation of the private batch loop. Register `llama_context_params.cb_eval` before context creation.
- During an active canonical call, every `ask=true` callback serializes node metadata immediately and recursively registers all `src[0..GGML_MAX_SRC-1]` and `view_src` dependencies. Return true for each node whose value is retained; its `ask=false` callback copies logical elements using dtype/strides before reuse. Continue evaluation by returning true from the completion callback. For metadata-only views, record alias metadata without claiming a separate arithmetic kernel or materialized value.
- GGML leaf tensors do not receive node callbacks; recover them through dependency traversal. `VIEW`, `RESHAPE`, transpose, and `CPY` ARE included in the CPU scheduler's node iteration. CPU optimization does not strip them at this commit. Source: `ggml/src/ggml-backend.cpp:1290-1310,1463-1468,1742-1777`; `ggml/src/ggml-cpu/ggml-cpu.cpp:193-210`; `ggml/src/ggml.c:7145-7182`.
- Use callback occurrence order and dependency identity for stable per-scenario IDs; do not identify tensors by shape alone or merge names that happen to match. Prove callback completeness with a small independently constructed GGML graph containing a leaf, view, reshape, compute node, and disconnected state `CPY`, comparing the callback registry with that graph's actual node/leaf arrays. Canonical capture additionally requires 24 scans, 24 convolutions, and per-layer R/S writeback dependencies.
- Include `llama-memory-recurrent.h` from the pinned private headers and type-check that the model uses `llama_memory_recurrent` before accessing `r_l`, `s_l`, `cells`, `head`, and `rs_idx`. Capture full zeroed buffers before first input and canonicalize the active sequence row after each call using the same `cell.src`/row selection as `state_write` (`src/llama-memory-recurrent.cpp:733-810`). With rollback disabled, R is `[1,1536,3]` and S is `[1,1536,16]` after logical layout normalization.
- Source `ggml/src/ggml.c:3520-3547` shows `CPY` result is a view of its destination. Preserve this mutation/alias relationship. Overlapping arena offsets without a view relation indicate reuse, not necessarily semantic aliasing.
- Link via `add_subdirectory` of the pinned source into `build/capture`; include its `src`, `include`, and GGML headers. CMake disables examples/tests/server and optional GPU/BLAS paths; the capture target enables its own CTest tests. No source-tree output or detached worktree is needed for this no-patch route.
- Set `n_gpu_layers=0`, `use_extra_bufts=false`, `offload_kqv=false`, `op_offload=false`, and one compute/batch thread. Disable graph reuse with `LLAMA_GRAPH_REUSE_DISABLE=1` for explicit per-call graph capture, and record it alongside fusion settings. Do not use instrumented timing as a performance result.

### Must NOT have (guardrails, anti-slop, scope boundaries)
- Product implementation is authorized only in the execution phase and task-owned worktrees; planning-only sessions remain read-only for product files.
- Local verified commits are authorized by the execution amendment. No push, PR, remote repository creation, Pages settings mutation, or deployment.
- No edits to unrelated source changes. Planning observed `D .pi/gg/SYSTEM.md` and untracked `.omo/` in llama.cpp; preserve them. Recheck actual status rather than resetting the tree.
- No browser model inference, backend, database, authentication, telemetry, paid services, or bundled model weights.
- No HF/PyTorch graph substituted for the actual llama.cpp graph; no GPU, training, Mamba-2/3, or generalized model platform.
- No fabricated values, performance claims, bitwise cross-runtime requirement, or hidden tolerance widening.
- No blanket F32 label inherited from model name; show tensor types actually loaded/emitted.
- No React, graph framework, Ponytail, or Caveman application dependency. Use browser primitives and only requirement-backed test/build dependencies.

## Verification strategy
> Zero human intervention - all verification is agent-executed.
- Behavioral changes use TDD: failing test at the seam, verify the intended failure, smallest fix, then related test plus real entry-point run. Use Vitest for TypeScript logic, Python `unittest`, CTest for the capture contract, and Playwright for browser behavior.
- No tests pin documentation prose. No fixed sleeps or time-luck polling; subscribe to expected browser/event state with a bounded timeout before triggering an action. Start long builds/tests with the session monitor and retain exit codes.
- Evidence root `E=.omo/evidence/mamba1-130m-visualizer`. Each task stores command, exit code, input identities, and assertions in `E/task-N/`. Machine comparison reports live in `.artifacts/validation/`, screenshots in `test-results/`; evidence receipts point to their exact hashes.
- Product commands to implement: `npm run typecheck`, `npm test -- --run`, `npm run build`, `npm run test:e2e`. Python commands use `.venv/bin/python`; isolated `.venv/` is also ignored. Test evidence must be from actual runs, not commands copied into a report.

### Numeric and cache acceptance
1. Artifact audit is an exact prerequisite. Independent mapping enumerates every source tensor and expected GGUF tensor. Unchanged F32 values must match bitwise after layout normalization; conv squeeze changes shape only; compare `ssm_a` with F32 `-torch.exp(A_log)` under the pinned converter environment. Tied output omission is allowed only when equality is established. Verify metadata and vocabulary index alignment, including padded vocabulary.
2. HF uses the same local snapshot, CPU F32, `eval()`, inference mode, eager path without optional Mamba/CUDA kernels or compilation. Pin packages before generating goldens. Record branch/cache API actually used by that version.
3. Run four schedules with final logits after the same 17th input token: HF fresh-17 (`H_F`), HF split-16+1 (`H_C`), llama fresh-17 (`L_F`), llama split-16+1 (`L_C`). Keep prefill last-row logits after input 16 as an additional cross-runtime comparison.
4. Required comparisons: `H_F/L_F`, `H_F/H_C`, `L_F/L_C`, `H_C/L_C`, and HF/llama prefill outputs. Cache after prefill must equal immediately-before-decode bytes within each runtime. Compare llama split-final versus fresh-final R/S numerically per layer. Do not byte-compare HF and llama caches: layout/history length differ.
5. Hard failures: wrong shape/row/vocabulary, any NaN/Inf, absent graph/state evidence, missing active-state mapping, artifact hash mismatch, or failed conversion audit. Record max absolute error, RMSE, NMSE (`sum(error^2)/sum(reference^2)`), p50/p95/p99 absolute error, and componentwise tolerance-violation count. Zero reference energy uses exact-zero error -> NMSE 0; otherwise -> infinity/fail.
6. Top-1 with top-1/top-2 margin, top-k overlap, and cosine are diagnostics only. Never declare success solely from generated text or token ranking.
7. Numerical policy lifecycle: predeclare canonical and calibration fixtures before examining outputs. Characterize the pinned single-thread CPU configuration; run identical schedules twice to check repeatability. Investigate the first divergent semantic boundary rather than automatically widening a threshold. Exact-repeatable outputs/states become hash-bound goldens.
8. Before a public PASS, an independent implementation-time reviewer must approve the artifact audit and the explained residual differences (reduction order, exp/softplus, GEMM/GEMV, etc.). Freeze observed accepted per-comparison/per-array error envelopes and componentwise absolute-error bounds in `fixtures/numerical-policy.json`, bound to environment and reference hashes, without a hidden multiplier. No default guessed `atol/rtol`; missing policy yields `unvalidated` and nonzero release-validation exit. This is a measured implementation gate, not a future owner decision.
9. Test the frozen policy with injected nonfinite values, vocabulary/row swaps, stale cache mappings, and finite perturbations outside its envelope; all must fail. A later regression cannot approve or regenerate its own policy. No broad model-conformance claim from the small corpus.

### UI acceptance
- For both scenarios, enumerate all entity IDs and verify three tensor sections and every referenced tensor's dtype/shape/numel/size. Verify mixed dtypes, empty weights, aliases, last-block shape changes, and state edges.
- Desktop: hierarchy rail, central flow, inspector, breadcrumb, scenario switch, search. Mobile: same data/actions via single-column drill-down and inspector. Explicit loading, invalid-data, empty-search, selection, and keyboard-focus states.
- Browser QA at 1440x900, 768x1024, and 390x844. No viewport overflow or clipped Korean copy; keyboard reaches every action with visible focus; search and browser back/forward preserve consistent selection/scenario.
- Accessible HTML tensor tables and a keyboard list accompany SVG; color is not the only encoding. Respect reduced motion and WCAG 2.2 AA contrast.
- Serve `dist` beneath `/web_mamba/` locally; direct hash-link reload, CSS/JS/JSON requests, and navigation all succeed. Browser requests must not fetch model files or external runtime services.

## Execution strategy
### Parallel execution waves
- Wave 1 (foundation): tasks 1-3 independently establish model preparation, data contract, and design. Three real domains; do not split solely to increase agent count.
- Wave 2 (evidence): tasks 4-6 build CPU capture, HF reference, and exact artifact audit after their prerequisites.
- Wave 3 (integration): tasks 7-9 validate/export data and build explorer/inspector. Tasks 8-9 may use small explicit test fixtures while 7 runs, but cannot pass real-data acceptance until 7 succeeds.
- Wave 4 (release preparation): tasks 10-12 finish local Pages behavior, exhaustive browser QA, and documentation.
- Final: manual QA plus one independent gate reviewer audits F1/F2/F4. The four final rows are acceptance domains, not a four-reviewer panel.
- Keep file ownership disjoint. Shared schema changes are integrated by the lead before dependents proceed. No worker rewrites another worker's files.
### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 13 | none | 1,2,3 | none |
| 1 | 13 | 4,5,6 | 2,3 |
| 2 | 13 | 4,7,8,9 | 1,3 |
| 3 | 13 | 8,9 | 1,2 |
| 4 | 1,2 | 7 | 5,6 |
| 5 | 1 | 7 | 4,6 |
| 6 | 1 | 7 | 4,5 |
| 7 | 2,4,5,6 | 10,11,12 | 8,9 development only |
| 8 | 2,3 | 10,11,12 | 7,9 |
| 9 | 2,3 | 10,11,12 | 7,8 |
| 10 | 7,8,9 | 11,12 | none |
| 11 | 7,8,9,10 | final | 12 |
| 12 | 7,8,9,10 | final | 11 |

## Todos
> Implementation + Test = ONE todo. Never separate.

- [x] 13. Establish authorized local Git baseline and phase worktree
  - Recommended task executor category: quick - execution infrastructure only.
  - Scope: preserve existing files; stage only the selected plan for the first baseline commit. Use local branch main and existing origin. Create the Foundation branch/worktree at `/home/cplcck/web_mamba-wt/mamba1-130m-visualizer-foundation`. Record baseline SHA and worktree path in Boulder.
  - Dependencies: none; blocks 1,2,3. Authorization: user's explicit local commit approval.
  - Acceptance: `git rev-parse HEAD` returns a commit and `git worktree list --porcelain` identifies the task-owned branch/path. No remote write occurs.
  - Happy QA: verify origin, baseline tracked-file inventory, and worktree ownership. Evidence: `.omo/ulw-execute/ledger.jsonl`.
  - Failure QA: confirm existing unrelated `.omo` artifacts are not staged, committed, or deleted. No blanket `git add .` or global identity edits.
  - Commit: Y - local baseline only; subsequent verified commits follow existing history.

- [x] 1. Prepare immutable model inputs and provenance
  - Recommended task executor category: deep - Python artifact preparation spans revision, conversion, and reproducibility contracts.
  - Files: `tools/prepare_model.py`, `tools/requirements.txt`, `tools/requirements.lock`, `fixtures/token-ids.json`, `fixtures/calibration-ids.json`, `tests/test_prepare_model.py`, `.gitignore`.
  - Work: use an isolated `.venv`, pin compatible Python dependencies and their hashes, verify baseline commit, capture pre-existing source status, resolve HF main once, then call `snapshot_download(revision=resolved_sha)` for the entire tree into the SHA-specific directory. Hash payload excluding HF cache metadata. Invoke the pinned converter on the local directory and persist the output hash. Existing same-SHA snapshots are verified, not blindly trusted. A later ordinary run uses the stored SHA; explicit `--refresh-revision` is the only refresh path.
  - Fix canonical IDs to `[1,42,314,2718,7,99,1024,17,2048,13,512,9,4096,23,128,31,64]`. Calibration fixtures are reversed canonical IDs, 17 repetitions of ID 42, and IDs 100 through 116. Validate all IDs against the resolved vocabulary and keep the fixtures hash-bound. These are synthetic inputs, not tokenized text.
  - Check config still identifies Mamba-1 130M (24 layers, hidden 768, inner 1536, state 16, convolution 4, dt rank 48). Unexpected architecture is a failure, not permission to visualize another model.
  - CLI: `.venv/bin/python tools/prepare_model.py --source /home/cplcck/llama.cpp-ssm --artifact-dir .artifacts` produces `.artifacts/manifests/model.json`; this manifest resolves all subsequent snapshot/GGUF paths.
  - Dependencies: none; blocks 4,5,6.
  - References: `conversion/mamba.py:16-100`, `convert_hf_to_gguf.py:195` under the pinned source; https://huggingface.co/docs/huggingface_hub/guides/cli#download-a-specific-revision ; approved version decision in the draft.
  - Acceptance: `python -m unittest discover -s tests -p test_prepare_model.py` passes under `.venv`; real CLI exits 0, records full immutable SHA, validates hashes, and never passes `--remote`.
  - Happy QA: invoke the real CLI and a second ordinary run; identities match and both point to the same recorded revision. Evidence: `E/task-1/provenance.json`.
  - Failure QA: dependency-injected Hub responses simulate moving main between resolution/download; assert download still uses the first SHA. Tamper an isolated payload copy and assert verification fails without overwriting the canonical snapshot. Evidence: `E/task-1/failures.txt`.
  - Commit: N - no commit authorization.

- [ ] 2. Establish frontend scaffold and executable data contracts
  - Recommended task executor category: unspecified-low - bounded TypeScript scaffold and schema work.
  - Files: `package.json`, `package-lock.json`, `index.html`, `tsconfig.json`, `vite.config.ts`, `src/schema.ts`, `tests/data.test.ts`.
  - Work: minimal Vite/vanilla TypeScript with strict checking and Vitest. Implement the Data contract above, safe symbolic dimension arithmetic, range-aware storage accounting, and ID/reference validation. Validate JSON at the browser boundary; no silent coercion or blanket dtype defaults. Tiny synthetic fixtures are test-only and clearly labelled.
  - Dependencies: none; blocks 4,7,8,9.
  - References: this plan's Data contract; pinned `ggml/include/ggml.h` tensor layout; `src/models/mamba.cpp:107-114` last-block row selection; https://vite.dev/guide/static-deploy#github-pages .
  - Acceptance: `npm run typecheck` and `npm test -- --run tests/data.test.ts` pass. Tests cover model/block/stage/operator sections, empty weight sets, mixed dtypes, strided views, shared ranges, and safe integer overflow.
  - Happy QA: validate two scenario fixtures and recompute `numel`, bytes, and boundary references. Evidence: `E/task-2/schema.txt`.
  - Failure QA: reject missing dtype, dangling producer, wrong capture ID, negative/overflowing dimensions, and unsupported formula operation. Distinguish unknown storage from zero. Evidence: `E/task-2/rejections.txt`.
  - Commit: N.

- [ ] 3. Define the engineering explorer design contract
  - Recommended task executor category: visual-engineering - visual hierarchy and accessibility design.
  - Files: `DESIGN.md`, `.omo/evidence/mamba1-130m-visualizer/task-3/` design reference artifacts.
  - Work: document an engineering-blueprint interface, compact mono metadata, Korean explanatory copy, and stable semantic flow colors. Define exact desktop/mobile layout, scroll ownership, spacing/type/color tokens, focus/hover/selected states, reduced motion, loading/error/search-empty states, SVG/list equivalence, and the three tensor sections at every level. No marketing landing page or gratuitous animation.
  - Use frontend and visual-QA skills at execution; read their routed design/layout/perfection references. Document references and tool-unavailable lanes honestly. Do not introduce paid services, React tooling, logos, or app dependencies for agent guidance. Establish reusable primitive examples before product screens.
  - Skill applicability for tasks 3,8,9,11: the frontend React Dev Tooling Gate is NOT APPLICABLE because this site is vanilla TypeScript and has no React dependency. `frontend/references/design/README.md:61-91` explicitly limits that gate to React projects and exempts vanilla. Keep its Design System and Primitive Showcase gates; use the Playwright state harness, browser accessibility checks, and Chromium performance inspection in task 11 instead of react-grab/react-scan/react-doctor. Do not install React to satisfy an inapplicable gate.
  - Dependencies: none; blocks 8,9.
  - References: draft Visual direction, Navigation, Communication discipline; frontend skill at `/home/cplcck/.nvm/versions/node/v24.14.1/lib/node_modules/omo-ai/plugin/skills/frontend/SKILL.md`; this plan UI acceptance.
  - Acceptance: design contract covers every named state and all three target viewports with no unexplained product fork. No source component is written by this task.
  - Happy QA: inspect the reference layouts and token contrast calculations; demonstrate hierarchy/inspector content fits the viewport geometry. Evidence: `E/task-3/design-review.md`.
  - Failure QA: review worst-case long tensor names, Korean wrapping, mixed dtype lists, empty weights, and reduced-motion state against the design; record corrections before implementation. Evidence: `E/task-3/edge-states.md`.
  - Commit: N.

- [ ] 4. Build real CPU graph and recurrent-state capture
  - Recommended task executor category: deep - internal graph/state lifetime and instrumentation require source-grounded C++ work.
  - Files: `tools/capture/CMakeLists.txt`, `tools/capture/main.cpp`, `tools/capture/capture.test.cpp`, and ignored `build/` outputs. Follow the Verified capture contract above; no llama.cpp product-source modifications.
  - Work: link the pinned source in a separate CMake build; CPU only (`n_gpu_layers=0`, no optional GPU/BLAS backend), one compute and batch thread, `n_seq_max=1`, `n_rs_seq=0`, context capacity at least 32, batch/microbatch capacity at least 32, only final input marked for logits. Export configured and effective values.
  - Set `GGML_CPU_DISABLE_FUSION=1` before backend initialization for BOTH canonical capture and corresponding llama numeric runs. Capture metadata for graph nodes and leaves, all `src` edges, view roots/offsets, operation parameters, state mutations, and original names. Values are copied before storage reuse or mutation, never read retrospectively from recycled pointers.
  - Run `split` and `fresh` schedules with the fixture. Split calls retain the same context/sequence: prefill positions 0..15, then position 16. Snapshot R/S and sequence cell mapping before prefill, after prefill, immediately before decode, and after decode; fresh also snapshots after 17. Synchronize before external readback. Discard constructor/reservation/warmup graphs, not actual executed nodes.
  - CLI: `build/capture/mamba-capture --manifest .artifacts/manifests/model.json --tokens fixtures/token-ids.json --schedule split --out .artifacts/captures/llama-split` and corresponding `--schedule fresh --out .artifacts/captures/llama-fresh`.
  - Dependencies: 1,2; blocks 7.
  - References: pinned `src/models/mamba-base.cpp:30-148`, `src/models/mamba.cpp:89-143`, `src/llama-memory-recurrent.h:64-114`, `src/llama-context.h:58-74,137-144`, `ggml/src/ggml-cpu/ggml-cpu.c:3029-3058,3891-3894`.
  - Acceptance: `cmake -S tools/capture -B build/capture -DLLAMA_SOURCE_DIR=/home/cplcck/llama.cpp-ssm -DCMAKE_BUILD_TYPE=Release` and `cmake --build build/capture --parallel 6` exit 0; `ctest --test-dir build/capture --output-on-failure` passes; both real CLI schedules exit 0. Every layer has `SSM_CONV`, `SSM_SCAN`, state writeback, and truthful dtype/size metadata. Metadata-only nodes have scheduler metadata observations but no claimed separate arithmetic execution.
  - Happy QA: assert prefill/decode graph identities and exact same-runtime after-prefill/before-decode state bytes; compare positions -1 -> 15 -> 16 and decode read source mapping. Evidence: `E/task-4/capture.json`.
  - Failure QA: capture tests exercise empty-cache single-token run (must not label it cached decode), invalid token ID, wrong position, and missing tensor dependencies; exporter rejects incomplete capture. Compare observer-enabled versus disabled runs to prove observation does not alter logits/state. Evidence: `E/task-4/capture-failures.txt`.
  - Commit: N.

- [ ] 5. Generate the pinned HF F32 reference schedules
  - Recommended task executor category: deep - runtime/cache oracle integration must preserve inference semantics.
  - Files: `tools/reference_hf.py`, HF harness cases in `tests/test_validation.py`; task 5 owns these cases before task 7 adds comparator cases.
  - Work: load the local manifest snapshot only, force CPU F32/eval/inference mode, no optional fused Mamba implementation. Run fresh and split schedules on the same fixtures as task 4, retaining a separate cache per split run. Export final and prefill logits with explicit input index, token ID, vocabulary axis, and HF runtime identity. Deep-copy mutable cache snapshots before the next call.
  - CLI: `.venv/bin/python tools/reference_hf.py --manifest .artifacts/manifests/model.json --tokens fixtures/token-ids.json --out .artifacts/captures/hf`.
  - Dependencies: 1; blocks 7.
  - References: inspected HF `modeling_mamba.py` `MambaModel.forward` cache contract and `MambaForCausalLM.forward` row selection; use the actual locked package's source/hash rather than assume historical line numbers. This plan Numeric and cache acceptance.
  - Acceptance: `python -m unittest discover -s tests -p test_validation.py` passes available harness cases; real CLI outputs finite `[vocab]` vectors for prefill/final positions with the expected identities. HF fresh/cached agreement is measured, not presumed.
  - Happy QA: run the CLI on canonical fixture and confirm split-after-prefill snapshot is unchanged by storing it separately from the live cache. Evidence: `E/task-5/hf-reference.json`.
  - Failure QA: missing snapshot, wrong vocabulary index, and accidental non-F32/optional-kernel execution are detected before comparison. Evidence: `E/task-5/hf-failures.txt`.
  - Commit: N.

- [ ] 6. Audit the source-to-GGUF conversion exactly
  - Recommended task executor category: deep - exhaustive independent tensor correspondence and transformed values.
  - Files: `tools/audit_gguf.py`, `tests/test_audit_gguf.py`.
  - Work: use pinned `gguf-py` reader and safetensors to enumerate every tensor and independently implement the expected mapping. Check all metadata and tensor names, counts, dtypes, shapes, values, vocabulary mapping, and tied weight handling. Do not use the converter's transformation function as the auditor's implementation.
  - CLI: `.venv/bin/python tools/audit_gguf.py --manifest .artifacts/manifests/model.json --out .artifacts/validation/artifact-audit.json`.
  - Dependencies: 1; blocks 7.
  - References: `conversion/mamba.py:28-100`, `src/models/mamba.cpp:35-77`, `gguf-py/gguf/tensor_mapping.py` under pinned source; Data contract storage distinctions.
  - Acceptance: unit tests pass and real audit exits 0 only if every source/target tensor is accounted for, allowing only explicitly verified tied omission and documented transformation/padding.
  - Happy QA: full canonical audit with counts and mapped tensor hashes. Evidence: `E/task-6/artifact-audit.json`.
  - Failure QA: tiny synthetic pairs contain missing/extra tensor, wrong squeeze axis, raw rather than transformed A, and unequal omitted output weight. Each fails at a named tensor/metadata field. Evidence: `E/task-6/corruption-tests.txt`.
  - Commit: N.

- [ ] 7. Validate all schedules and export canonical website data
  - Recommended task executor category: deep - cross-runtime numeric diagnosis and provenance-bound export.
  - Files: `tools/validate.py`, `tools/export_web.py`, `tests/test_validation.py`, `tests/test_export_web.py`, `fixtures/numerical-policy.json`, `public/data/*.json`.
  - Work: implement the full Numeric and cache acceptance contract, characterize canonical/calibration fixtures, obtain the independent numeric review, freeze policy, and then run release validation. Comparator configuration is separate from candidate input so a run cannot loosen its own policy.
  - Export only validated metadata and bounded statistics, not full tensor arrays. Generate model/block/stage/operator membership, parent boundary edges, per-layer weights, alias/range accounting, and scenario-specific tensors from actual captures. Verify all raw graph operators have a hierarchy owner; fail on unmapped operators rather than silently omit them.
  - CLIs: `.venv/bin/python tools/validate.py --manifest .artifacts/manifests/model.json --captures .artifacts/captures --policy fixtures/numerical-policy.json --out .artifacts/validation/report.json`; `.venv/bin/python tools/export_web.py --manifest .artifacts/manifests/model.json --captures .artifacts/captures --validation .artifacts/validation/report.json --out public/data`.
  - Dependencies: 2,4,5,6; blocks 10,11,12.
  - References: Data contract and Numeric and cache acceptance; `src/models/mamba.cpp:107-114`; `mamba-base.cpp:59-77,114-148`; historical `tests/test-backend-ops.cpp` NMSE formula is a diagnostic reference, NOT a full-model tolerance.
  - Acceptance: Python tests pass, release validation exits 0, export exits 0, and `npm test -- --run tests/data.test.ts` validates real public JSON. All public documents bind to the same artifact/capture identity.
  - Happy QA: four-schedule matrix plus prefill and all 24 layers' split/fresh R/S checks, policy review receipt, real-data schema/coverage report. Evidence: `E/task-7/validation.json`.
  - Failure QA: reject stale report/hash, shifted output row, dropped graph node, broken state mapping, nonfinite values, and finite error exceeding policy. A missing/unreviewed policy cannot produce a public PASS. Evidence: `E/task-7/negative-cases.txt`.
  - Commit: N.

- [ ] 8. Implement hierarchy navigation and actual operator graph
  - Recommended task executor category: visual-engineering - accessible hierarchy/graph interaction.
  - Files: `src/explorer.ts`, `src/graph.ts`, `tests/hierarchy.test.ts`.
  - Work: implement state as scenario plus entity ID; stable hash URL, back/forward, search by original tensor/operator name, breadcrumb, all 24 block choices, stage drill-down, and SVG flow with keyboard-accessible list equivalent. Preserve semantic entity selection on scenario switch; if absent, select nearest available ancestor and communicate it.
  - Consume schema types and `DESIGN.md`; export selection/state events for task 9 instead of editing its files. Show input/output/state edge distinctions. Collapsing repeated blocks is presentation only, not data substitution. Fit/reset and native scrolling provide graph navigation without a general graph framework.
  - Dependencies: 2,3; real-data gate additionally requires 7; blocks 10,11,12.
  - References: `DESIGN.md` from 3, `src/schema.ts` from 2, approved hierarchy scope and UI acceptance in this plan.
  - Acceptance: `npm test -- --run tests/hierarchy.test.ts` and `npm run typecheck` pass; browser can reach every entity in both real scenarios.
  - Happy QA: select model, block 0, stage, operator, then block 23; switch scenario and traverse browser history. Evidence: `E/task-8/navigation.json`.
  - Failure QA: unknown URL ID, search miss, and scenario-missing ID produce defined non-crashing states without misleading tensor selection. Evidence: `E/task-8/navigation-errors.json`.
  - Commit: N.

- [ ] 9. Implement every-level tensor inspector and responsive shell
  - Recommended task executor category: visual-engineering - dense tensor metadata across screen sizes.
  - Files: `src/main.ts`, `src/inspector.ts`, `src/styles.css`.
  - Work: load/validate static documents and wire navigation. For every entity show separate inputs, outputs, weights with tensor-specific dtype, native/logical shape, strides, numel, bytes/IEC, aliases, producer/consumers, and evidence. Show parent weight union and unique storage totals, not one parent dtype. R/S cache panels explain before/after state; kernel equations remain explanatory. Include provenance/validation panel and safe symbolic dimension calculator.
  - Render untrusted names as text, never raw HTML. Browser data-load failure blocks graph rendering and explains invalid/missing data instead of displaying synthetic fallback values.
  - Dependencies: 2,3; integrates 8 and real data 7 for acceptance; blocks 10,11,12.
  - References: `DESIGN.md`, schema contract, UI acceptance, user requirement that EVERY hierarchy level includes input/output/weight dtype and size.
  - Acceptance: typecheck/build pass and browser assertions verify all required fields on all entity IDs in both scenarios; no column hides required information on mobile.
  - Happy QA: inspect mixed I32/F32 tensors, view aliases, state mutations, tied weight references, weight-free operators, and last-block shapes. Evidence: `E/task-9/inspector-coverage.json` and screenshots.
  - Failure QA: malformed/missing JSON, very long tensor names, empty weights, and unsupported symbolic dimensions show defined states without overflow or fake zero allocation. Evidence: `E/task-9/inspector-errors.json`.
  - Commit: N.

- [ ] 10. Prepare a locally verified GitHub Pages build
  - Recommended task executor category: unspecified-low - bounded static routing and workflow integration.
  - Files: `vite.config.ts`, `.github/workflows/pages.yml`, build-related package scripts; integrates 8/9 entrypoint.
  - Work: use `import.meta.env.BASE_URL` for data/assets and hash-based entity routing. Default local base `/`; build test base `/web_mamba/`; Pages workflow derives repository base from GitHub context (account-root repository uses `/`). Workflow is `workflow_dispatch` only, with build artifact upload and Pages deploy permissions; no activation or dispatch now.
  - Only `dist` is published. Add a build-content check rejecting models, raw captures, local absolute paths, secrets, and stale validation identities from the deployable tree.
  - Dependencies: 7,8,9; blocks 11,12.
  - References: https://vite.dev/guide/static-deploy#github-pages ; https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages ; approved no-external-write scope.
  - Acceptance: `npm run build -- --base=/web_mamba/` exits 0; local preview serves the nested base; all assets/data return 200 and correct content type. Workflow is syntactically valid and packages `dist` only.
  - Happy QA: load nested base, direct entity hash URL, refresh, switch scenario, and verify same tensor metadata. Evidence: `E/task-10/pages-local.json`.
  - Failure QA: failed data request yields the designed error; build content check rejects a synthetic forbidden artifact in a temporary output directory, never modifies canonical output to hide a problem. Evidence: `E/task-10/publish-boundary.txt`.
  - Commit: N.

- [ ] 11. Exercise the complete website in a real browser
  - Recommended task executor category: visual-engineering - exhaustive interactive and visual verification.
  - Files: `playwright.config.ts`, `tests/explorer.spec.ts`, `test-results/`; targeted in-scope UI corrections with their owner.
  - Work: configure Playwright webServer readiness and bounded event-driven assertions. Enumerate all entity/scenario combinations for metadata checks; enumerate distinct visual state families across three viewports for screenshots. Inspect every captured screenshot; do not infer mobile quality from desktop or block-23 correctness from block 0.
  - Test keyboard-only navigation, focus, accessible names/roles, contrast, reduced motion, loading/error/search-empty state, scenario switch, history, weight dedup, dimension calculator, public provenance, direct nested-base reload, and no model/external-service requests. Run visual-QA and frontend performance checks on the production build; record measured results and failures honestly.
  - Dependencies: 7,8,9,10; blocks final.
  - References: this plan UI acceptance; `DESIGN.md`; visual-QA skill; generated dataset used as enumeration source.
  - Acceptance: `npm run test:e2e` exits 0 in one reliable run after fixes; desktop/tablet/mobile screenshots have no clipping, overflow, lost data, or inaccessible required action. Runtime console has no uncaught errors.
  - Happy QA: complete traversal/metadata coverage and screenshot matrix. Evidence: `E/task-11/browser-report.json`, linked screenshots and accessibility report.
  - Failure QA: intercept a JSON response before navigation to return invalid schema; verify fail-closed UI. Test zero-result search and out-of-range estimate. No sleeps or prose-equality tests. Evidence: `E/task-11/error-states.json`.
  - Commit: N.

- [ ] 12. Document reproducible execution and the publication boundary
  - Recommended task executor category: writing - concise operational documentation from verified receipts.
  - Files: `README.md`, final provenance summaries; `.gitignore` refinements if generated directories require exclusion.
  - Work: describe source ownership, HF revision selection vs llama commit, complete tool sequence, immutable re-run procedure, fixture/row semantics, numeric policy status, artifact locations, dtype/size terms, kernel-local limitations, and local preview. Include exact commands that have actually passed and explain Pages is prepared but not deployed. Note Ponytail/Caveman are agent guidance only.
  - Document environment/bootstrap from lockfiles and separate numeric calibration from regression verification. No command silently refreshes main or overwrites goldens. Include capture instrumentation identity if used.
  - Dependencies: 7,8,9,10; blocks final.
  - References: task 1/4/7/10 receipts, approved file layout, this plan's exclusions.
  - Acceptance: every README command has a matching successful task receipt or is explicitly a future external-deployment instruction requiring approval; all documented paths exist or are identified generated outputs. No new prose-pinning tests.
  - Happy QA: replay documented local metadata validation/build/preview commands against canonical artifacts, reusing already captured model run evidence where identities match. Evidence: `E/task-12/readme-check.md`.
  - Failure QA: verify documentation distinguishes absent model snapshot, unreviewed numerical policy, and unconfigured remote deployment; no success claims for any of these states. Evidence: `E/task-12/scope-check.md`.
  - Commit: N.

## Final verification wave
- [ ] F1. Plan compliance audit
  - One independent gate reviewer maps every scope requirement and task acceptance to current evidence. Verify every hierarchy level's input/output/weight fields and full raw-operator coverage, not a representative screenshot.
  - Happy: all mappings have fresh receipts. Failure: missing stage, weight dtype, cache continuity, or numeric gate blocks approval.
  - Evidence: `E/final/compliance.md`. Depends on 1-12. Commit: N.
- [ ] F2. Code quality review
  - The same gate reviewer inspects focused diffs, typecheck/test/build results, minimal dependencies, source preservation, and capture observer noninterference. Reconcile diagnostics before build; never suppress errors.
  - Happy: source and generated artifacts agree. Failure: hidden tolerance widening, swallowed data error, timing-luck test, or fabricated metadata blocks approval.
  - Evidence: `E/final/code-review.md`. Depends on 1-12. Commit: N.
- [ ] F3. Real manual QA
  - Lead drives the production nested-base site in Chromium, inspects desktop/tablet/mobile, switches both scenarios, follows model -> block -> stage -> operator, and reads input/output/weight dtype/size at each level. Verify state and provenance panels against machine reports.
  - Happy: complete visual-state inventory inspected. Failure: missing field, broken deep link, clipped copy, or stale tensor after scenario switch is fixed and affected checks rerun.
  - Evidence: `E/final/manual-qa.md` plus fresh screenshots. Depends on 1-12. Commit: N.
- [ ] F4. Scope fidelity
  - Same gate reviewer verifies no model payloads in `dist`, no external deployment/write, no unrelated source changes, no invented scalar GGML nodes, and no peak-memory claim derived by adding tensor payloads.
  - Happy: artifact inventory and source status match scope. Failure: any boundary violation blocks delivery.
  - Evidence: `E/final/scope.md`. Depends on 1-12. Commit: N.

## Commit strategy
- Local baseline and verified implementation commits are authorized by the execution amendment. Commit one verified increment at a time; never rewrite history or include unrelated files. Root performs local integration after independent verification.
- Preserve the pinned llama.cpp baseline and all unrelated modifications. Record capture-only patches separately if required; do not claim a patched binary is the unmodified commit.
- Do not create a GitHub repository, push, dispatch workflows, alter Pages settings, or deploy. Those are later owner actions.

## Success criteria
- Actual pinned CPU execution and immutable F32 GGUF provenance support every public graph record.
- Both canonical scenarios use fixed IDs and proven cache continuity; artifact audit, the full HF/llama schedule matrix, and reviewed numeric policy pass.
- Every model/block/stage/operator entity exposes each input/output/weight dtype and size without alias double-counting or fabricated allocations.
- All 24 blocks, metadata operators, recurrent mutations, and last-block output-row selection remain accurately inspectable.
- Website is responsive, keyboard accessible, browser-tested, and works under a GitHub project-site base path with only static published data.
- Automated checks and F1-F4 evidence are clean or explicitly explain a pre-existing unrelated issue; in-scope defects are fixed. No false green based on unrun tests.
- Execution handoff reports exact artifact paths, source/config identities, verification commands/results, and that public deployment did not occur.
