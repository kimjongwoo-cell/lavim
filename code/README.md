# WSI-LatentMAS 실행 안내

이 폴더는 WSI-VQA 실험을 실행하는 통합 코드입니다. 모든 실험은 `run.py`에서
데이터셋, 모델, 방법, latent step, GPU와 주요 추론 옵션을 지정합니다.

## 기본 실행

```bash
cd /home/users/whddn12316/wsi_latentmas_0806/code
source /home/users/whddn12316/venvs/wsi-latentmas-py312/bin/activate

python run.py \
  --dataset wsi-vqa \
  --model qwen-4b \
  --method single \
  --gpus 5,6,7,8 \
  --steps 5
```

실행 전에 데이터와 모델 경로가 현재 서버의 기본 설정과 맞는지 확인합니다.
경로가 다른 경우 `run.py --help`에 표시되는 옵션과 프로젝트 설정 파일을
확인하여 수정합니다.

## 주요 방법

`--method`에는 다음 값을 사용할 수 있습니다.

- `single`: 단일 모델 추론
- `single-no-thinking`: thinking을 사용하지 않는 단일 모델 추론
- `vlmas`: 텍스트 기반 멀티에이전트 추론
- `latent-base`: latent 멀티에이전트 기준선
- `pruning-v3`: visual KV pruning 실험
- `reallocation-v2`: pathology-aware reallocation 실험
- `both`: pruning과 reallocation 결합 실험

## 모델과 데이터셋

모델은 `qwen-2b`, `qwen-4b`, `qwen-8b` 중에서 선택합니다.

데이터셋은 다음 중 하나를 선택합니다.

```text
wsi-vqa, expert-vqa, slidebench, tcga, gtex, panda
```

latent 방법의 경우 latent step을 바꾸어 반복 실행합니다.

```bash
python run.py --dataset wsi-vqa --model qwen-4b \
  --method latent-base --gpus 5,6,7,8 --steps 5
python run.py --dataset wsi-vqa --model qwen-4b \
  --method latent-base --gpus 5,6,7,8 --steps 10
```

## 자주 사용하는 옵션

- `--patch-budget 8`: 질문당 사용할 patch 수
- `--io-pipeline`: patch 입출력 pipeline 사용
- `--navigator-kv`: Navigator KV 전달
- `--canonical-open-options`: open-ended 답변의 canonical option 처리
- `--greedy-decoding`: greedy decoding 사용
- `--answerer-protocol structured_json`: 구조화된 최종 답변 형식
- `--output-root PATH`: 결과 저장 위치
- `--resume`: 기존 결과에서 이어서 실행
- `--dry-run`: 실제 추론 없이 실행 계획만 확인

예를 들어 4B latent base를 step 20으로 이어서 실행하려면:

```bash
python run.py \
  --dataset wsi-vqa \
  --model qwen-4b \
  --method latent-base \
  --gpus 5,6,7,8 \
  --steps 20 \
  --patch-budget 8 \
  --io-pipeline \
  --navigator-kv \
  --resume
```

## 결과와 대시보드

결과는 기본적으로 저장소의 `results/` 아래에 저장됩니다. 실행 상태와
성능은 대시보드에서 확인할 수 있습니다.

```bash
cd /home/users/whddn12316/wsi_latentmas_0806/code
python -m dashboard.server --port 8767
```

브라우저에서 `http://127.0.0.1:8767`을 열면 작업 진행률, GPU 할당,
초/질문, 완료 표본 수와 평가 지표를 확인할 수 있습니다.

## 도움말

전체 옵션은 다음 명령으로 확인합니다.

```bash
python run.py --help
```
