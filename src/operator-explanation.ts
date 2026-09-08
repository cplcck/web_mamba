import type { CaptureDocument, Entity, Tensor } from './schema';

export type OperatorOperand = { readonly tensor: Tensor; readonly label: string };
export type OperatorFact = { readonly key: string; readonly label: string; readonly value: string | number | readonly number[] };
export type OperatorExplanation = {
  readonly op: string; readonly summary: string; readonly operation: string;
  readonly inputs: readonly OperatorOperand[]; readonly outputs: readonly OperatorOperand[];
  readonly facts: readonly OperatorFact[]; readonly notes: readonly string[];
};

// Pinned 8144f319: ggml.h enums; ggml.c constructors; ggml-cpu/{ops.cpp,vec.cpp,ggml-cpu.c}.
const unaryNames = ['ABS', 'SGN', 'NEG', 'STEP', 'TANH', 'ELU', 'RELU', 'SIGMOID', 'GELU', 'GELU_QUICK', 'SILU',
  'HARDSWISH', 'HARDSIGMOID', 'EXP', 'EXPM1', 'SOFTPLUS', 'GELU_ERF', 'XIELU', 'FLOOR', 'CEIL', 'ROUND', 'TRUNC'] as const;
const gluNames = ['REGLU', 'GEGLU', 'SWIGLU', 'SWIGLU_OAI', 'GEGLU_ERF', 'GEGLU_QUICK'] as const;
const unaryActions = ['절댓값으로 바꿉니다', '부호를 나타내는 값으로 바꿉니다', '부호를 뒤집습니다', '양수인지에 따라 0 또는 1로 바꿉니다',
  '쌍곡탄젠트로 값을 눌러 줍니다', '음수 영역에 지수함수를 적용합니다', '음수를 0으로 바꿉니다', '시그모이드로 0과 1 사이로 바꿉니다',
  'GELU로 값의 크기에 따라 통과량을 조절합니다', '빠른 GELU 근사로 통과량을 조절합니다', '자기 값으로 0과 1 사이의 비율을 구하고 그 비율을 원래 값에 곱합니다(SiLU). 이 비율을 구하는 함수가 시그모이드이며, 곱한 결과 자체가 0과 1 사이로 제한되는 것은 아닙니다',
  '구간별 직선 게이트를 곱합니다', '구간별 직선으로 0과 1 사이로 바꿉니다', '자연상수 e의 거듭제곱으로 바꿉니다', '지수함수 값에서 1을 뺍니다',
  'softplus로 부드러운 양수 값으로 바꿉니다', '오차함수 기반 GELU로 통과량을 조절합니다', 'XIELU 비선형 함수를 적용합니다',
  '아래 정수로 내립니다', '위 정수로 올립니다', '가까운 정수로 반올림합니다', '소수 부분을 버립니다'] as const;
type Effect = 'metadata-only' | 'copy' | 'compute' | 'unavailable';
const descriptions: Readonly<Record<string, readonly [string, string, Effect]>> = {
  GET_ROWS: ['행 번호 목록에 따라 필요한 행을 골라 가져옵니다.', '행 번호가 담긴 목록(인덱스)을 읽고, 데이터에서 해당 행의 숫자들을 골라 결과에 복사합니다. 어떤 행을 골랐는지는 실제 행 번호 값이 있어야 알 수 있습니다.', 'copy'],
  RESHAPE: ['같은 데이터를 다른 모양으로 해석합니다.', '원소의 저장 순서와 개수를 유지하면서 축 크기를 다시 지정하고, 원래 저장 공간을 공유합니다.', 'metadata-only'],
  VIEW: ['저장된 데이터의 전체나 일부를 복사 없이 바라봅니다.', '데이터를 새로 옮기지 않고, 기존 저장 공간에서 어디부터 얼마나 읽을지 지정합니다. 시작 위치까지의 거리를 오프셋, 같은 축에서 다음 숫자로 이동할 간격을 stride라고 하며 둘 다 바이트 단위입니다.', 'metadata-only'],
  SCALE: ['각 숫자에 정해진 배율을 곱하고 보정값을 더합니다.', '입력의 각 숫자에 배율(scale)을 곱한 뒤 보정값(bias, 편향)을 더합니다. 이 두 수는 데이터의 원소 값이 아니라 캡처에 기록된 연산 설정입니다.', 'compute'],
  CPY: ['입력 데이터를 지정된 목적지에 복사합니다.', '첫 입력의 원소를 목적지 텐서의 저장 공간에 기록합니다. 자료형이 다르면 목적지 자료형으로 변환합니다.', 'copy'],
  RMS_NORM: ['각 행의 숫자들을 공통 기준으로 나눠 크기를 조절합니다.', '한 행의 숫자들을 각각 제곱하고 그 평균을 구합니다. 여기에 0으로 나누는 일을 막기 위한 설정값(epsilon)을 더한 뒤 제곱근을 구하고, 원래 숫자들을 모두 그 값으로 나눕니다. 이렇게 묶음의 크기 기준을 맞추는 작업을 정규화라고 합니다.', 'compute'],
  MUL: ['대응하는 원소끼리 곱합니다.', '두 입력의 대응하는 원소를 곱합니다. 작은 입력은 GGML의 반복 규칙에 따라 큰 입력의 축에 맞춰 반복해서 사용합니다.', 'compute'],
  ADD: ['대응하는 원소끼리 더합니다.', '두 입력의 대응하는 원소를 더합니다. 편향처럼 작은 입력은 GGML의 반복 규칙에 따라 큰 입력의 축에 맞춰 반복해서 사용합니다.', 'compute'],
  MUL_MAT: ['여러 입력 숫자에 계수를 곱해 더하여 각 출력 숫자를 만듭니다.', '출력 숫자 하나마다 여러 입력 숫자에 각각 계수(얼마나 반영할지 정하는 수)를 곱한 뒤 모두 더합니다. 출력마다 다른 계수 묶음을 사용하며, 이를 여러 입력 묶음에 반복합니다. 이런 계산을 숫자 표 전체에 수행하는 것이 행렬 곱입니다.', 'compute'],
  TRANSPOSE: ['첫 두 축의 해석을 서로 바꿉니다.', 'ne0과 ne1의 크기와 바이트 간격을 서로 바꿉니다. 저장된 원소를 옮기지 않고 읽는 축의 순서만 바꿉니다.', 'metadata-only'],
  CONCAT: ['지정한 축을 따라 두 데이터를 이어 붙입니다.', '지정된 GGML ne 축에서 첫 입력 뒤에 두 번째 입력을 이어 붙여 결과에 복사합니다. 다른 축의 크기는 유지합니다.', 'copy'],
  SSM_CONV: ['짧은 입력 구간의 숫자들에 계수를 곱해 더합니다.', '짧은 입력 구간의 숫자들에 정해진 계수(필터)를 각각 곱해 더합니다. 이 구간을 시간 순서대로 한 칸씩 옮겨 반복합니다. 같은 종류의 숫자가 이어지는 줄을 채널이라고 하며, 각 채널을 따로 계산하고 채널끼리는 섞지 않습니다.', 'compute'],
  SSM_SCAN: ['이전 기억과 현재 입력을 합쳐 기억과 출력을 차례로 갱신합니다.', '문장을 나눈 입력 조각(토큰)을 하나씩 처리합니다. 앞서 읽은 내용을 숫자로 요약해 둔 기억(상태)을 얼마나 남길지 조절하고, 현재 입력의 정보를 더해 새 기억을 만듭니다. 새 기억의 숫자들에 계수를 곱해 더하면 이번 단계의 출력이 됩니다. 이 기억을 다음 토큰 처리에 이어 쓰며, 결과에는 각 단계의 출력과 처리 후 기억을 함께 담습니다.', 'compute'],
  CONT: ['원소를 연속된 저장 배치로 복사합니다.', '입력의 바이트 간격을 따라 원소를 읽고, 출력에는 빈 간격 없는 연속 배치로 복사합니다. 출력 모양은 캡처된 축 크기를 따릅니다.', 'copy'],
} as const;

function subtype(output: Pick<Tensor, 'op' | 'opParamsI32'> | undefined): string | undefined {
  const id = output?.opParamsI32?.[0];
  if (id === undefined) return undefined;
  switch (output?.op) {
    case 'UNARY': return unaryNames[id];
    case 'GLU': return gluNames[id];
    default: return undefined;
  }
}
export function operatorSummary(output: Pick<Tensor, 'op' | 'opParamsI32'> | undefined): string {
  switch (output?.op) {
    case 'UNARY': return subtype(output) === 'SILU' ? '각 숫자에 자기 값으로 정한 0과 1 사이의 비율을 곱합니다.' : subtype(output) ? `각 숫자를 ${subtype(output)} 규칙에 따라 바꿉니다.` : '각 숫자에 적용할 함수의 세부 종류를 확인할 수 없습니다.';
    case 'GLU': return subtype(output) ? `${subtype(output)}로 조절용 숫자를 변환해 데이터에 곱합니다.` : '데이터에 곱할 조절값을 만드는 함수의 종류를 확인할 수 없습니다.';
    default: return descriptions[output?.op ?? '']?.[0] ?? '캡처 근거만으로 연산의 의미를 확인할 수 없습니다.';
  }
}
function floatParameter(output: Tensor, index: number): number | undefined {
  const bits = output.opParamsI32?.[index];
  if (bits === undefined) return undefined;
  const bytes = new DataView(new ArrayBuffer(4));
  bytes.setInt32(0, bits, true);
  return bytes.getFloat32(0, true);
}

/** Facts use GGML ne0..ne3 ordering and byte strides, not display/reversed logicalShape.
 * input* / shapeRelation compare the first displayed input with the opcode-bearing output.
 * 'unavailable' means absent metadata, never zero. No fact contains captured tensor element values.
 */
export function buildOperatorExplanation(document: CaptureDocument, entity: Entity): OperatorExplanation | null {
  if (entity.kind !== 'operator') return null;
  const tensors = new Map(document.tensors.map(t => [t.id, t]));
  const tensor = (id: string): Tensor => {
    const found = tensors.get(id);
    if (!found) throw new RangeError(`Unknown tensor reference: ${id}`);
    return found;
  };
  const outputs = entity.outputTensorIds.map(tensor);
  const output = outputs.find(t => t.op);
  if (!output?.op) return null;
  const op = output.op;
  const allInputs = [...new Set([...entity.inputTensorIds, ...entity.weightTensorIds])].map(tensor);
  const weights = entity.weightTensorIds.map(tensor);
  const data = entity.inputTensorIds.filter(id => !entity.weightTensorIds.includes(id)).map(tensor);
  // web_graph.py preserves order inside each partition, but moves weights out of GGML src[].
  // These constructors place weights at src[0] (matrix/rows), src[3] (scan), or after data.
  let sources: readonly (Tensor | undefined)[] = allInputs;
  switch (op) {
    case 'GET_ROWS': case 'MUL_MAT': sources = weights.length ? [...weights, ...data] : allInputs; break;
    case 'SSM_SCAN': sources = weights.length ? [...data.slice(0, 3), ...weights, ...data.slice(3)] : allInputs; break;
  }
  const [a, b] = sources;
  const description = descriptions[op];
  const effect = description?.[2] ?? (op === 'UNARY' || op === 'GLU' ? 'compute' : 'unavailable');
  let operation = description?.[1] ?? operatorSummary(output);
  const labels = new Map<string, string>();
  const facts: OperatorFact[] = [];
  const notes = ['축 크기는 GGML ne0, ne1, ne2, ne3 순서이며, stride는 바이트 단위입니다. 텐서의 실제 원소 값은 이 공개 캡처에 없습니다.'];
  const fact = (key: string, label: string, value: OperatorFact['value'] | undefined | null) => {
    facts.push({ key, label, value: value ?? 'unavailable' });
  };
  const roles = (names: readonly string[]) => names.forEach((name, i) => {
    const source = sources[i];
    if (source) labels.set(source.id, name);
  });
  fact('effect', '처리 종류', effect);
  fact('elementValues', '텐서 원소 값', 'unavailable');
  fact('parameterEvidence', '연산 설정 근거', output.opParamsI32?.length ? 'captured' : 'unavailable');
  fact('outputShape', '출력 축 크기 (ne 순서)', output.nativeShape);
  fact('outputNumel', '출력 원소 수', output.numel);
  fact('outputStridesBytes', '출력 축 간격 (바이트)', output.strides);
  const firstInput = allInputs[0];
  if (firstInput) {
    fact('inputShape', '첫 입력 축 크기 (ne 순서)', firstInput.nativeShape);
    fact('inputNumel', '첫 입력 원소 수', firstInput.numel);
    fact('inputStridesBytes', '첫 입력 축 간격 (바이트)', firstInput.strides);
    fact('shapeRelation', '첫 입력과 출력의 축 크기 비교', firstInput.nativeShape.length === output.nativeShape.length && firstInput.nativeShape.every((n, i) => n === output.nativeShape[i]) ? 'equal' : 'different');
  }
  if (output.viewSourceId !== null) {
    fact('viewSourceId', '출력이 공유하는 저장 원본 ID', output.viewSourceId);
    fact('viewOffsetBytes', '저장 원본 기준 누적 오프셋 (바이트)', output.viewOffsetBytes);
  }
  switch (effect) {
    case 'metadata-only': notes.push('모양과 읽는 위치만 지정하는 메타데이터 연산입니다. 원소를 복사하거나 산술 계산하지 않습니다.'); break;
    case 'copy': notes.push('원소를 읽어 결과에 기록하는 복사 계열입니다. 모양만 바꾸는 메타데이터 연산과 다릅니다.'); break;
    case 'compute': notes.push('원소를 계산하는 연산입니다. 입력과 출력 모양이 같아도 값이 그대로라는 뜻은 아닙니다.'); break;
    case 'unavailable': notes.push('이 opcode의 세부 계약은 설명 모델에서 지원하지 않습니다. 캡처된 입출력 메타데이터만 표시합니다.'); break;
    default: { const exhaustive: never = effect; return exhaustive; }
  }
  if (output.numel === 0) notes.push('출력 원소 수가 0인 빈 텐서입니다. 연산 노드가 있다는 사실만으로 실제 원소 계산이 수행되었다고 볼 수 없습니다.');
  if (output.arithmeticExecution === false) notes.push('이 캡처는 이 노드의 산술 실행을 기록하지 않았습니다. 아래 설명은 GGML 연산 계약입니다.');
  switch (op) {
    case 'GET_ROWS':
      roles(['행 데이터', '행 인덱스']);
      fact('dataTensorId', '행 데이터 ID', a?.id); fact('indexTensorId', '인덱스 텐서 ID', b?.id);
      fact('indexCount', '인덱스 원소 수', b?.numel); fact('indexValues', '선택한 행 번호의 실제 값', 'unavailable');
      fact('rowAxis', '행 선택 축', 1); fact('rowWidth', '한 행의 원소 수', a?.nativeShape[0]); fact('availableRows', '데이터의 행 수', a?.nativeShape[1]);
      notes.push('인덱스 텐서의 ID와 모양은 알 수 있지만, 실제 토큰 번호나 선택된 행 번호는 알 수 없습니다.');
      break;
    case 'RESHAPE':
      roles(['모양을 다시 해석할 데이터']);
      fact('sourceTensorId', '입력 데이터 ID', a?.id); fact('logicalRank', '원래 호출의 논리적 차원 수', 'unavailable');
      if (a?.nativeShape.every((n, i) => n === output.nativeShape[i])) operation = '캡처된 네 축 크기는 입력과 출력이 같습니다. 같은 저장 데이터를 다시 해석하는 RESHAPE 노드이며, 이 기록만으로 보이지 않는 논리 축의 변경을 복원할 수는 없습니다.';
      notes.push('GGML은 사용하지 않는 뒤쪽 축도 크기 1로 채웁니다. ne[4]만으로 원래 reshape 호출의 차원 수를 단정하지 않습니다.');
      break;
    case 'VIEW': {
      roles(['바라볼 원본 데이터']); fact('sourceTensorId', '직접 입력 ID', a?.id);
      const low = output.opParamsI32?.[0], high = output.opParamsI32?.[1];
      const offset = low === undefined || high === undefined ? undefined : (low >>> 0) + (high >>> 0) * 2 ** 32;
      fact('sourceOffsetBytes', '직접 입력 기준 오프셋 (바이트)', offset !== undefined && Number.isSafeInteger(offset) ? offset : undefined);
      notes.push('직접 입력 기준 오프셋은 64비트 size_t 설정입니다. viewSourceId는 중첩 view를 거슬러 올라간 저장 원본이며 viewOffsetBytes는 그 원본 기준 누적 오프셋입니다.');
      break;
    }
    case 'SCALE':
      roles(['배율을 적용할 데이터']); fact('scale', '배율 (F32)', floatParameter(output, 0)); fact('bias', '편향 (F32)', floatParameter(output, 1));
      notes.push('출력이 저장 원본을 공유하면 그 공간에 결과를 기록합니다. 공유 여부만으로 값이 보존된다고 해석하면 안 됩니다.');
      break;
    case 'CPY':
      roles(['복사할 데이터', '복사 목적지']); fact('sourceTensorId', '복사 원본 ID', a?.id); fact('destinationTensorId', '목적지 텐서 ID', b?.id);
      notes.push(output.viewSourceId ? '이 출력은 viewSourceId가 가리키는 기존 저장 공간의 view입니다. CPY가 새 메모리를 할당했다는 뜻이 아닙니다.' : '출력의 저장 원본 view 정보가 없습니다. CPY라는 이름만으로 기존 공간 공유나 새 메모리 할당을 단정하지 않습니다.');
      break;
    case 'RMS_NORM':
      roles(['정규화할 데이터']); fact('epsilon', '안정화 상수 epsilon (F32)', floatParameter(output, 0));
      fact('normalizationAxis', '제곱평균을 구하는 축', 0); fact('normalizationWidth', '정규화 묶음의 원소 수', a?.nativeShape[0]);
      notes.push('평균값을 빼는 정규화가 아닙니다. 학습된 가중치를 곱하는 MUL은 별도 노드입니다.');
      break;
    case 'ADD': case 'MUL':
      roles(op === 'ADD' ? ['더할 데이터', '더할 값 / 편향'] : ['곱할 데이터', '곱할 값 / 가중치']);
      fact('broadcastAxes', '두 입력 크기가 다른 반복 축 (ne 번호)', a && b ? a.nativeShape.flatMap((n, i) => n !== b.nativeShape[i] ? [i] : []) : undefined);
      break;
    case 'MUL_MAT':
      roles(['행렬 A / 투영 가중치', '행렬 B / 입력 특징']);
      fact('matrixATensorId', 'GGML src0 행렬 A ID', a?.id); fact('matrixBTensorId', 'GGML src1 행렬 B ID', b?.id);
      fact('reductionAxis', '곱한 값을 더하는 공통 축', 0); fact('reductionSize', '공통 축의 원소 수', a?.nativeShape[0]);
      fact('outputAxis0Size', '출력 ne0 = A의 ne1', output.nativeShape[0]); fact('outputAxis1Size', '출력 ne1 = B의 ne1', output.nativeShape[1]);
      fact('outputBatchShape', '출력 배치 축 ne2, ne3', output.nativeShape.slice(2));
      notes.push('GGML ne0은 대응 숫자를 곱해 더하는 공통 축입니다. 결과의 ne0 크기는 A의 ne1, ne1 크기는 B의 ne1에서 오며, ne2와 ne3은 여러 묶음을 구분하는 B의 배치 축입니다. A의 배치 축은 B에 맞춰 반복될 수 있습니다. 이 축 순서를 화면의 가로·세로나 일반적인 row-major 행렬 표기와 혼동하지 않습니다.');
      break;
    case 'TRANSPOSE': roles(['축을 바꿔 읽을 데이터']); fact('axisPermutation', '출력 축별 입력 축 번호', [1, 0, 2, 3]); break;
    case 'CONCAT':
      roles(['앞에 놓을 데이터', '뒤에 이을 데이터']); fact('concatAxis', '이어 붙이는 GGML ne 축', output.opParamsI32?.[0]);
      break;
    case 'SSM_CONV':
      roles(['이력과 현재 입력', '시간 필터 가중치']);
      fact('kernelWidth', '필터 창 길이', b?.nativeShape[0]); fact('historyWidth', '필요한 앞쪽 이력 길이', b?.nativeShape[0] === undefined ? undefined : b.nativeShape[0] - 1);
      fact('channels', '독립 채널 수', output.nativeShape[0]); fact('tokenCount', '시퀀스당 출력 토큰 수', output.nativeShape[1]);
      fact('windowStep', '시간 창 이동 간격', 1); fact('sequenceCount', '시퀀스 수', output.nativeShape[2]);
      notes.push('필터 창 길이는 가중치의 ne0입니다. 입력 ne0에서 창 길이를 빼고 1을 더한 수만큼 출력 토큰을 만듭니다.');
      break;
    case 'UNARY': {
      roles(['함수를 적용할 데이터']); const id = output.opParamsI32?.[0];
      fact('subtypeId', 'GGML 단항 함수 번호', id); fact('subtype', 'GGML 단항 함수 종류', subtype(output) ?? (id === undefined ? 'unavailable' : 'unknown'));
      operation = id !== undefined && unaryActions[id] ? `각 원소에 대해 ${unaryActions[id]}. 출력 모양은 입력과 같습니다.` : operatorSummary(output);
      break;
    }
    case 'SSM_SCAN': {
      roles(['이전 상태 s', '토큰 입력 x', '시간 간격 dt', '상태 감쇠 가중치 A', '입력 반영 계수 B', '출력 읽기 계수 C', '상태 행 인덱스']);
      const [state, x, , , , , ids] = sources;
      const [width, dim, heads] = state?.nativeShape ?? [];
      const sequences = ids?.nativeShape[0], slots = output.opParamsI32?.[0];
      const stateNumel = width === undefined || dim === undefined || heads === undefined || sequences === undefined || slots === undefined ? undefined : width * dim * heads * sequences * slots;
      fact('stateWidth', '상태 축 길이', width); fact('headDimension', '헤드당 입력 폭', x?.nativeShape[0]); fact('headCount', '헤드 수', x?.nativeShape[1]);
      fact('tokenCount', '시퀀스당 토큰 수', x?.nativeShape[2]); fact('sequenceCount', '시퀀스 수', sequences); fact('stateSlots', '상태 저장 슬롯 수 K', slots);
      fact('stateIndexTensorId', '이전 상태 행 인덱스 ID', ids?.id); fact('indexValues', '이전 상태 행 번호의 실제 값', 'unavailable');
      fact('activationShape', '앞쪽 활성값 영역의 ne 축 크기', x?.nativeShape); fact('activationNumel', '앞쪽 활성값 원소 수', x?.numel);
      fact('stateShape', '상태 슬롯 하나의 ne 축 크기', width === undefined || dim === undefined || heads === undefined || sequences === undefined ? undefined : [width, dim, heads, sequences]);
      fact('stateNumel', '뒤쪽 상태 영역 전체 원소 수', stateNumel); fact('stateOffsetBytes', '출력 시작 기준 상태 영역 오프셋 (바이트)', x === undefined ? undefined : x.numel * 4);
      notes.push('GGML 결과는 하나의 F32 텐서입니다. 앞에는 x와 같은 개수의 활성값 y, 바로 뒤에는 상태가 저장됩니다. 이는 별개의 두 출력 텐서가 아닙니다.');
      notes.push(slots === 1 ? 'K=1이므로 뒤쪽 영역은 모든 토큰을 처리한 최종 상태입니다. 이후 VIEW와 CPY가 이 영역을 분리해 상태 캐시에 기록합니다.' : '상태 슬롯 0은 최종 상태이며, K>1이면 뒤쪽 슬롯에 끝에서 거슬러 올라간 토큰의 상태도 보관합니다. K가 없으면 상태 영역의 크기를 확정할 수 없습니다.');
      notes.push('dt는 기억을 바꾸는 시간 간격을 정하는 입력이며, softplus는 이를 부드럽게 양수로 바꾸는 함수입니다. A는 이전 기억을 남기는 정도, B는 현재 입력을 반영하는 정도, C는 기억에서 출력을 읽는 계수입니다. 이 계산과 지수함수는 SSM_SCAN 커널 내부 동작이며, 별도로 캡처된 연산 노드나 실제 중간 원소 값이 아닙니다.');
      break;
    }
    case 'CONT': roles(['연속 배치로 옮길 데이터']); notes.push('값을 계산하는 연산이 아니라 저장 배치를 정리하는 복사입니다. 실제 물리 메모리 할당이나 재사용은 이 opcode만으로 알 수 없습니다.'); break;
    case 'GLU': {
      const id = output.opParamsI32?.[0], swapped = output.opParamsI32?.[1], name = subtype(output);
      roles(b ? ['활성 함수를 적용할 게이트', '게이트를 곱할 데이터'] : ['게이트와 데이터가 함께 든 입력']);
      fact('subtypeId', 'GGML 게이트 함수 번호', id); fact('subtype', 'GGML 게이트 함수 종류', name ?? (id === undefined ? 'unavailable' : 'unknown'));
      fact('gluLayout', '게이트 입력 배치', b ? 'separate' : 'packed'); fact('swapped', 'packed 입력 반쪽 교환 설정', swapped);
      fact('gateTensorId', '활성 함수를 적용하는 텐서 ID', a?.id); fact('valueTensorId', '게이트와 곱하는 텐서 ID', (b ?? a)?.id);
      const half = a?.nativeShape[0] === undefined ? undefined : a.nativeShape[0] / 2;
      fact('gateOffsetElements', '각 ne0 행에서 게이트 시작 원소', b ? 0 : swapped === undefined ? undefined : swapped ? half : 0);
      fact('valueOffsetElements', '각 ne0 행에서 데이터 시작 원소', b ? 0 : swapped === undefined ? undefined : swapped ? 0 : half);
      operation = name ? `${name}는 조절용 숫자 묶음(게이트)을 정해진 함수로 바꾼 뒤, 그 결과를 데이터의 대응 숫자에 곱합니다. 게이트는 데이터에 무엇을 곱할지 정하는 입력이지, 그 자체가 0과 1 사이의 비율이라는 뜻은 아닙니다.` : operatorSummary(output);
      if (name === 'SWIGLU') operation = '첫 번째 입력은 데이터를 조절할 숫자 묶음(게이트)입니다. 각 게이트 숫자로 0과 1 사이의 비율을 구하는 시그모이드 함수를 적용하고, 그 비율을 원래 게이트 숫자에 곱합니다. 이 계산이 SiLU이며, 그 결과를 두 번째 입력의 대응 숫자에 곱합니다. 비율만 0과 1 사이이고, 게이트 숫자나 SiLU 결과가 그 범위로 제한되는 것은 아닙니다.';
      if (!b && name) operation = `${name}는 한 입력을 첫 번째 축(ne0)에서 반으로 나눠 조절용 숫자(게이트)와 데이터로 사용합니다. swapped는 두 반쪽의 역할을 바꾸는 설정입니다. 게이트를 정해진 함수로 바꾼 뒤 데이터에 곱하며, 게이트 자체가 0과 1 사이의 비율이라는 뜻은 아닙니다.`;
      if (name === 'SWIGLU_OAI') {
        fact('alpha', '시그모이드 배율 alpha (F32)', floatParameter(output, 2)); fact('limit', '입력 제한값 limit (F32)', floatParameter(output, 3));
        operation = '조절용 숫자(게이트)는 limit보다 커지지 않게 하고, 데이터는 -limit과 limit 사이로 제한합니다. 게이트에 배율 alpha를 적용해 시그모이드로 0과 1 사이의 비율을 구한 뒤, 원래의 제한된 게이트에 그 비율을 곱합니다. 마지막으로 제한된 데이터에 1을 더한 값과 곱합니다. 비율의 범위와 최종 결과의 범위는 서로 다릅니다.';
      }
      notes.push(b ? '두 입력을 따로 받는 경우 src0이 활성 함수 쪽이고 src1이 곱할 데이터입니다. swapped는 이 경우 적용되지 않습니다.' : '한 입력을 받는 경우에만 ne0 축의 두 반쪽을 나눕니다. 오프셋은 바이트가 아니라 각 ne0 행 안의 원소 수입니다.');
      break;
    }
  }
  return { op, summary: operatorSummary(output), operation,
    inputs: allInputs.map(tensor => ({ tensor, label: labels.get(tensor.id) ?? ({ activation: '입력 데이터', weight: '가중치', state: '상태', index: '인덱스' } as const)[tensor.role] })),
    outputs: outputs.map(tensor => ({ tensor, label: op === 'SSM_SCAN' ? '활성값과 상태를 합친 결과' : '연산 결과' })), facts, notes };
}
