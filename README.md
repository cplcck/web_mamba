# Mamba-1 130M CPU 계산 탐색기

실제 llama.cpp CPU capture를 읽는 한국어 정적 웹사이트입니다. 전체 모델 → 24개 block → stage → GGML operator를 탐색하고, 각 계층의 input/output/weight dtype과 크기를 확인합니다. 브라우저에서 모델을 실행하지 않습니다.

## 웹사이트 실행

일반 clone에서는 Node.js와 npm만 필요합니다. Python, llama.cpp, HF snapshot, GGUF 없이 저장소의 `public/data/*.json`으로 빌드합니다. 검증 환경은 Node.js 24.14.1, npm 11.11.0이며 의존성은 `package-lock.json`으로 설치합니다.

저장소 루트에서:

```sh
npm ci
npm run typecheck
npm run build -- --base=/web_mamba/
npm run preview -- --host 127.0.0.1 --port 4173 --strictPort --base=/web_mamba/
```

`http://127.0.0.1:4173/web_mamba/`를 엽니다. preview는 계속 실행되는 서버이며 종료는 Ctrl-C입니다. `/` 기준 빌드는 다음 명령으로 별도 확인했습니다.

```sh
npm run build
```

빌드는 publication boundary 검사도 실행합니다. 모델 파일, raw capture, 비공개 경로, secret 패턴, 승인되지 않았거나 현재 public JSON과 다른 데이터를 배포 결과에 허용하지 않습니다.

검증한 publication 테스트:

```sh
npm test -- --run tests/publication.test.ts
```

브라우저 검증은 `npm run test:e2e`로 실행합니다. 정확히 고정된 Playwright 1.63.0과 별도로 설치된 Google Chrome이 필요합니다. 현재 설정은 `channel: 'chrome'`을 사용하며 browser download 없이 검증했습니다. task11의 실제 실행 결과는 6개 통과, skip 0, retry 0입니다. prefill/decode의 총 2760개 탐색 상태를 전수 확인했고, keyboard/focus, semantic label, reduced motion, text contrast의 targeted accessibility 검사도 수행했습니다. 전체 WCAG 인증을 의미하지 않습니다.

## 관측 조건

`fixtures/token-ids.json`의 고정 입력은 다음 17개 ID입니다. 문자열 tokenization, sampling, padding, 암묵적 BOS/EOS 삽입은 없습니다.

```text
[1,42,314,2718,7,99,1024,17,2048,13,512,9,4096,23,128,31,64]
```

- sequence ID는 0, position은 0..16입니다.
- prefill은 처음 16개 입력, `P=1,T=16,Q=16,O=1`입니다.
- cached decode는 같은 context의 상태를 유지한 마지막 1개 입력, `P=1,T=1,Q=1,O=1`입니다. 빈 상태에서 실행한 single-token run이 아닙니다.
- `P=n_seqs`, `T=n_seq_tokens`, `Q=P*T`, `O=n_outputs`입니다. 각 호출은 마지막 입력 뒤의 logits 한 행만 보존합니다. prefill 비교는 position 15, 최종 비교는 position 16입니다.
- 모델은 hidden 768, inner 1536, state 16, convolution 4, dt rank 48, 24 layers입니다. logits vocabulary는 50280입니다.
- CPU compute/batch thread는 각각 1입니다. GPU offload는 없고 `GGML_CPU_DISABLE_FUSION=1`, `LLAMA_GRAPH_REUSE_DISABLE=1`을 사용했습니다. 요청 context 32와 실제 context 256을 구분해 기록합니다.

상태는 attention KV cache가 아닙니다. R은 Conv history, S는 SSM state입니다. 활성 sequence의 논리 shape는 layer마다 R `[1,1536,3]`, S `[1,1536,16]`입니다. 최초 전체 zero buffer와 활성 row snapshot도 구분합니다. after-prefill과 before-decode 사이의 48개 R/S 배열 및 cell mapping은 같은 runtime 안에서 정확히 이어집니다. HF와 llama.cpp는 cache layout/history 길이가 달라 raw cache를 서로 byte 비교하지 않습니다.

## tensor와 연산 읽기

- `dtype`는 실제 tensor별 값입니다. F32 GGUF라고 index까지 F32로 표시하지 않습니다. 공개 capture에는 F32와 I32가 있습니다.
- 공개 graph `Tensor`의 `nativeShape`와 `strides`는 각각 GGML `ne[4]`와 byte stride `nb[4]`입니다. UI에서는 `native ne[4]`, `native nb[4]`로 표시합니다. `logicalShape`는 의미 축 기준이며, `numel`은 원소 수입니다. 실제 zero-element tensor도 보존합니다.
- 별도 R/S `StateArraySummary`는 논리 shape를 `shape`, native byte stride를 `nativeStrides`에 기록합니다. graph `Tensor.strides`와 필드 이름을 구분합니다.
- `logicalBytes`는 compact payload, `ggmlNbytes`는 stride 간격을 포함할 수 있는 addressed span입니다. byte 값과 IEC 단위 KiB/MiB를 함께 읽습니다.
- view는 source와 offset을 참조합니다. 별도 allocation으로 세지 않습니다. `requiredAllocBytes`는 non-view의 backend 요구량, `bufferBytes`는 backing buffer 전체 용량입니다.
- 실제 allocator slot 크기는 알 수 없어 `null`과 이유를 표시합니다. unknown은 0이 아닙니다. 중간 tensor 크기의 합은 peak memory가 아닙니다.
- GGUF payload, runtime backing range, 논리 weight reference는 별도 집계입니다. tied weight도 runtime 중복 저장이 자동으로 0 byte가 되지는 않습니다.
- weight가 없는 entity는 빈 weight 목록을 표시합니다. 모든 block은 실제 기록이며 마지막 block의 `O=1` row selection도 보존합니다.
- `VIEW`, `RESHAPE`, transpose, `CPY`는 inspect 가능한 metadata/mutation 연산입니다. scheduler 관측을 별도 arithmetic kernel 실행으로 과장하지 않습니다.
- `SSM_SCAN`과 `SSM_CONV`는 각각 단일 GGML operator입니다. 내부 수식은 `kernel-internal` 설명이며 scalar graph node나 allocation을 만들어 내지 않습니다.
- `A_log -> -exp(A_log)`는 offline conversion입니다. runtime에 가상의 A 생성 `EXP` node를 추가하지 않습니다.
- dimension calculator는 symbolic estimate입니다. 다른 입력 크기의 실제 실행이나 allocation을 측정했다는 뜻이 아닙니다.

## 고정된 provenance

모델 revision과 converter/runtime commit은 서로 다른 고정 대상입니다.

| 대상 | 고정값 |
| --- | --- |
| HF repository | `state-spaces/mamba-130m-hf` |
| 실행 시작 시 main을 한 번 해석한 revision | `1e76775f628fbf1350fbe4dbb3d971ba64af25a1` |
| converter와 CPU runtime | `8144f3192e5a3131cd043f284525e6ceebf82d0f` |
| F32 GGUF SHA-256 | `af897eef0adfda25444be8fd2e6cbe3e858a107f7926625f7fb71d37f51bcaae` |
| preparation manifest SHA-256 | `d2e457d3cbf5ef4da63d243ad1a3d9c63de369e8816c7b5a429246298b1c77b7` |
| capture binary SHA-256 | `db94bba2fef1e53ede267cb0f3deb622cc0b4f9ce4b8e5a4a167ffbe3ecde7b8` |
| frozen policy SHA-256 | `9fba342f1ec68a2917286a589ecf4057d3f811c43571c0da8fe8637979fa65c7` |

공개 root capture ID는 `web-fc50f53501f3760325c1385978d2b5e580271e36ab6f5471b9b466ea7d904a21`입니다. scenario capture ID와 파일 hash는 `public/data/manifest.json`에 있습니다.

전체 immutable HF snapshot을 로컬 디렉터리에 받고 고정 converter의 `--outtype f32`로 변환했습니다. `--remote`는 사용하지 않았습니다. 일반 prepare 재실행은 저장된 revision과 payload hash를 검사하며 main을 다시 선택하지 않습니다. 이 문서는 `--refresh-revision` 실행을 지시하지 않습니다.

llama.cpp 원본은 `/home/cplcck/llama.cpp-ssm`에 보존합니다. 제품 source patch 없이 standalone `tools/capture`에서 public `llama_decode`, eval callback, 고정 private recurrent-memory headers를 사용했습니다. binary/compiler/CMake flags와 capture 도구 파일 hash는 `public/data/provenance.json`에 있습니다. 계측 시간은 성능 결과가 아닙니다.

공개 provenance에서는 source의 raw status 목록과 로컬 경로를 제거했습니다. commit, converter hash, product patch hash, worktree diff hash는 유지합니다. 원래 status와 payload별 hash는 비공개 Foundation manifest에 그대로 남아 있습니다.

## offline 생성과 immutable replay

이 절은 일반 clone용 bootstrap이 아닙니다. 검증 당시의 Foundation/Evidence/Integration 디렉터리와 비공개 산출물을 보존한 환경을 설명합니다. 아래 축약 경로를 사용합니다.

```sh
F=/home/cplcck/web_mamba-wt/mamba1-130m-visualizer-foundation
V=/home/cplcck/web_mamba-wt/mamba1-130m-visualizer-evidence
I=/home/cplcck/web_mamba-wt/mamba1-130m-visualizer-integration
```

Foundation의 `.venv/bin/python`은 Python 3.10.12입니다. `tools/requirements.txt`는 직접 의존성, `tools/requirements.lock`은 정확한 transitive distribution과 hash를 기록합니다. CPU torch 2.11.0+cpu, Transformers 4.57.6을 사용하며 GGUF Python 코드는 고정 llama.cpp에서 가져옵니다. 별도 임시 venv에 hash-locked clean install도 확인했습니다.

### 최초 생성 기록: 보존된 출력에 다시 실행하지 않음

다음은 성공 receipt가 있는 순서와 명령입니다. 이미 존재하는 snapshot, reference, capture, golden을 새로 만들거나 덮어쓰는 재현 절차가 아닙니다.

Foundation 환경 설치와 모델 준비:

```sh
cd "$F"
uv pip install --python .venv/bin/python --require-hashes --index-strategy unsafe-best-match -r tools/requirements.lock
.venv/bin/python tools/prepare_model.py --source /home/cplcck/llama.cpp-ssm --artifact-dir .artifacts
```

Evidence에서 exact conversion audit와 HF reference:

```sh
cd "$V"
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 "$F/.venv/bin/python" -B tools/audit_gguf.py --manifest "$F/.artifacts/manifests/model.json" --out .artifacts/validation/artifact-audit.json
"$F/.venv/bin/python" -B tools/reference_hf.py --manifest "$F/.artifacts/manifests/model.json" --tokens "$F/fixtures/token-ids.json" --out .artifacts/captures/hf
```

audit는 242개 source/target tensor, vocabulary와 padding을 전수 확인했습니다. identity/squeeze는 값이 bitwise 일치하고, 24개 A 변환은 고정 환경의 F32 `-torch.exp(A_log)`와 정확히 일치합니다. HF는 로컬 snapshot, CPU F32, eval/inference mode의 eager 경로를 사용하며 optional Mamba/CUDA kernel과 compilation을 사용하지 않습니다.

같은 Evidence에서 native build와 두 schedule:

```sh
cmake -S tools/capture -B build/capture -DLLAMA_SOURCE_DIR=/home/cplcck/llama.cpp-ssm -DCMAKE_BUILD_TYPE=Release
cmake --build build/capture --parallel 6
ctest --test-dir build/capture --output-on-failure
build/capture/mamba-capture --manifest "$F/.artifacts/manifests/model.json" --tokens "$F/fixtures/token-ids.json" --schedule split --out .artifacts/captures/llama-split
build/capture/mamba-capture --manifest "$F/.artifacts/manifests/model.json" --tokens "$F/fixtures/token-ids.json" --schedule fresh --out .artifacts/captures/llama-fresh
```

reference/capture는 기존 출력 디렉터리를 거부합니다. 불완전한 출력을 성공으로 간주하거나 자동 resume하지 않습니다. 위 canonical 생성만으로 수치 승인 corpus가 완성되지는 않습니다. Integration의 `.artifacts/task-7/captures`에는 canonical과 사전 선언한 reversed, repeated-42, range-100-116의 반복 실행 및 hash-bound golden inventory가 보존되어 있습니다.

### 승인된 corpus 재검증과 export

수치 replay는 **보존된 Integration cwd에서만** 지원합니다. frozen policy의 `approval.receipt`에는 다음 절대 경로가 들어 있습니다.

```text
/home/cplcck/web_mamba-wt/mamba1-130m-visualizer-integration/fixtures/numerical-review.json
```

Foundation manifest/venv/snapshot/GGUF, Evidence audit/binary, Integration corpus 및 trusted authorization을 원래 위치와 bytes로 유지해야 합니다. policy를 새 clone 경로로 바꾸거나 refreeze해서 경로 검사를 통과시키지 않습니다. 입력이 없거나 환경/hash가 달라지면 기존 PASS의 replay가 아닙니다.

다음은 이 cwd에서 성공한 검증/export 명령입니다. 다시 실행하면 **파생 report와 public metadata를 다시 씁니다**. golden, capture, snapshot, 승인 파일과 frozen policy를 생성하거나 갱신하는 명령은 아닙니다. 단순 웹사이트 실행에는 필요 없습니다.

```sh
cd "$I"
"$F/.venv/bin/python" -B tools/validate.py --manifest "$F/.artifacts/manifests/model.json" --captures .artifacts/task-7/captures --policy fixtures/numerical-policy.json --out .artifacts/task-7/validation/approved-release-report.json
"$F/.venv/bin/python" -B tools/export_web.py --manifest "$F/.artifacts/manifests/model.json" --captures .artifacts/task-7/captures --policy fixtures/numerical-policy.json --validation .artifacts/task-7/validation/approved-release-report.json --out public/data
```

export는 주어진 PASS report만 믿지 않고 현재 입력을 재검증합니다. Integration exporter에도 공개 source status 제거가 반영되어 있습니다. 이 README 검증에서는 위 대형 모델 작업을 다시 실행하지 않고 기존 receipt와 일치하는 hash를 재사용했습니다.

## 수치 PASS의 범위

HF fresh-17/split-16+1과 llama.cpp fresh-17/split-16+1을 비교했습니다. 다섯 logits 비교는 `H_F/L_F`, `H_F/H_C`, `L_F/L_C`, `H_C/L_C`, prefill HF/llama입니다. 각 runtime의 24 layers R/S fresh/split 비교까지 네 fixture에서 총 404개 비교가 승인된 범위를 만족했습니다.

`fixtures/numerical-policy.json`은 측정 corpus/configuration에 한정된 정책입니다. 각 배열에서 관측한 max absolute error를 componentwise bound로 적용하고, RMSE, NMSE, p50/p95/p99도 관측값으로 제한합니다. 추측한 공통 atol/rtol이나 숨은 multiplier가 없습니다. NMSE는 `sum(error^2)/sum(reference^2)`입니다. top-1/margin, top-k overlap, cosine은 진단일 뿐 합격 기준을 대체하지 않습니다.

잔차는 normalization reduction, shape-dependent SGEMM, convolution 누적, exp/softplus와 recurrence 연산 결합 순서의 저장 입력 control로 조사했습니다. 독립 reviewer가 exact audit와 설명된 잔차를 승인했고, 별도 root authorization 뒤 정책을 freeze했습니다. 후보가 자기 receipt나 tolerance를 수정해 승인할 수 없습니다.

NaN/Inf, 잘못된 row/vocabulary/state mapping, hash 불일치, 누락된 graph evidence, envelope 밖의 finite perturbation은 실패입니다. policy가 없거나 미승인 상태면 `unvalidated`이며 release validation은 nonzero로 종료합니다. 네 개의 합성 fixture에 대한 측정 gate이지, 다른 CPU/configuration이나 광범위한 모델 적합성 보장이 아닙니다. 생성 문장이나 token 순위만으로 정확성을 주장하지 않습니다.

## 파일과 공개 경계

- `src/`, `DESIGN.md`: vanilla strict TypeScript/SVG/HTML/CSS 탐색기와 UI 계약.
- `public/data/`: model, manifest, provenance, prefill, decode, validation의 정적 metadata와 제한된 통계만 포함합니다.
- Foundation `.artifacts/hf`, `.artifacts/gguf`, `.artifacts/manifests`: 비공개 immutable 입력과 원본 hash manifest.
- Evidence `.artifacts/captures`, `.artifacts/validation`, `build/capture`: 비공개 reference/capture/audit와 native binary.
- Integration `.artifacts/task-7`: 비공개 승인 corpus, golden과 전체 비교 report.
- `.venv`, `.artifacts`, `build`, `dist`, `node_modules`, `test-results`, `playwright-report`는 Git에서 제외합니다. retained task evidence도 일반 clone에 포함된다고 가정하지 않습니다.

`.github/workflows/pages.yml`은 `workflow_dispatch`만 준비했습니다. repository 이름에서 project base를 계산하고 account-root repository는 `/`를 사용합니다. upload 대상은 `dist`뿐입니다. raw weights, logits, activations, 전체 R/S vector는 공개하지 않습니다.

**실제 배포는 하지 않았습니다.** push, PR, remote 설정 변경, Pages 활성화, workflow dispatch 권한이 이 문서나 workflow 준비로 부여되지 않습니다. 원격 배포 설정이 없어도 로컬 정적 빌드는 독립적으로 동작합니다.

Ponytail과 Caveman은 agent의 구현/문체 지침일 뿐 package나 제품 dependency가 아닙니다.

## 검증 기록

각 보존 worktree에서 `E=.omo/evidence/mamba1-130m-visualizer`입니다.

| 범위 | 보존 receipt |
| --- | --- |
| 환경/prepare | Foundation `E/task-1/lock-install.json`, `clean-lock-replay.json`, `real-final-first-command.json`, `DoneClaim.json` |
| native build/capture | Evidence `E/task-4/DoneClaim.json`, `capture.json` |
| HF reference | Evidence `E/task-5/canonical.command`, `DoneClaim.json` |
| exact audit | Evidence `E/task-6/artifact-audit.json` |
| 독립 수치 승인/검증 | Integration `E/task-7/independent-numeric-delta/review.md`, `root-authorization-install.json`, `release-report.json` |
| status 비공개화/Pages | Release `E/task-10/DoneClaim.json`, commit `2a139b6d2f8594bbe728e1ffb368bd003306a915` |
| 이 문서의 재사용/실행 확인 | Release `E/task-12/readme-check.md` |
