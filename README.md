# Korean Whisper Fine-tuning

회의 지원 앱 **Mote**의 한국어 음성 인식을 위해 수행한 Whisper 파인튜닝과 도메인 데이터 구축 실험입니다. 한국어 기본 모델을 학습한 뒤 개발·경제 분야의 음성–텍스트 데이터를 직접 구성해 추가 학습하고, 모델과 데이터셋을 Hugging Face에 공개했습니다.

**Gyutak Hahn · 한규탁** — 데이터 수집·전처리, 모델 학습·평가, 모델 공개 및 추론 적용을 담당했습니다.

[Models & Datasets](https://huggingface.co/imTak) · [Mote AI Backend](https://github.com/MettingApp/AI) · [Portfolio](https://hahngyutak.github.io/portfolio/?project=mote#projects)

## My Contribution

| 작업 | 직접 수행한 내용 |
|---|---|
| 도메인 데이터 구축 | 영상 음성과 한국어 자막을 수집하고 자막 시간에 맞춰 음성을 분할했습니다. 개발·경제 분야별 음성–텍스트 쌍을 구성하고 학습·평가용 데이터셋으로 공개했습니다. |
| 음성·정답 전처리 | 오디오를 최종 16 kHz로 통일하고 1초 이하 샘플을 제외했습니다. 자막의 일부 구두점·개행을 정리하고, Whisper 입력 특징과 한국어 정답 토큰으로 변환했습니다. |
| 한국어·도메인 학습 | Hugging Face의 Whisper 학습 예제를 기반으로 한국어 데이터와 직접 구성한 도메인 데이터에 맞춰 학습 파이프라인을 구성했습니다. 한국어 체크포인트에서 개발·경제 모델을 각각 추가 학습했습니다. |
| 학습 메모리 관리 | FP16 혼합정밀도와 gradient checkpointing을 적용했습니다. 음성 특징과 정답 토큰의 패딩을 분리하고 정답 패딩은 `-100`으로 처리해 loss 계산에서 제외했습니다. |
| 평가·체크포인트 관리 | 생성된 전사문과 정답으로 WER를 계산하고, 낮은 WER를 기준으로 최적 체크포인트를 불러오도록 구성했습니다. 모델과 processor를 함께 저장·공개했습니다. |
| 추론·서비스 적용 | CTranslate2 변환·양자화와 faster-whisper 기반 CPU 추론을 구성했습니다. Mote에서는 화자 분리·요약 기능을 연결하고 백엔드 담당 팀원과 AI API·앱 연동을 수행했습니다. |

Mote 발표자료에 기록된 모델 파일 용량은 **3.2 GB → 820 MB**입니다. 서비스 연동 코드는 [Mote AI 저장소](https://github.com/MettingApp/AI)에서 확인할 수 있습니다.

## Pipeline

```text
영상 음성 + 한국어 자막
        ↓ 자막 타임스탬프 기반 분할·텍스트 정리
음성–정답 쌍 → 최종 16 kHz 변환 → 1초 이하 샘플 제외
        ↓ 세그먼트 단위 train/test 분할
log-Mel 입력 특징 + 한국어 전사 토큰
        ↓ FP16 · gradient checkpointing · Seq2SeqTrainer
생성 전사문 WER 평가 → 체크포인트 선택
        ↓ 모델·processor 공개
CTranslate2 변환 → faster-whisper 추론 → Mote AI 기능 연동
```

### Data Preparation

- [`vutube_custom.py`](./vutube_custom.py): VCtube를 바탕으로 한국어 수동 자막을 가져오고, 자막 시작 시간·길이에 맞춰 WAV 구간을 만듭니다. 인접 자막의 시작 시간을 반영해 분할 길이를 조정하며 일부 구두점과 개행을 제거합니다.
- [`make_hfds_economy.py`](./make_hfds_economy.py): 음성·자막 수집과 분할을 실행한 뒤 `subtitle.csv`를 Hugging Face Dataset으로 변환합니다. 최종 오디오 열을 `Audio(sampling_rate=16000)`으로 설정하고 `len(array) > 16000`인 샘플만 남깁니다.
- [Mote의 데이터 구축 코드](https://github.com/MettingApp/AI/blob/main/make_hug_ds.py): 영상별 세그먼트를 결합하고 80:20 비율로 나누는 관련 파이프라인입니다.

학습에 사용하는 핵심 필드는 다음과 같습니다. `name`은 단순 파일명 문자열이 아니라 디코딩된 음성을 제공하는 **Audio 열**입니다.

| 필드 | 내용 | 학습 시 변환 |
|---|---|---|
| `name` | 음성 배열·sampling rate | `input_features`: log-Mel 특징 |
| `text` | 대응하는 한국어 전사문 | `labels`: 한국어 토큰 ID |

한국어 기본 학습 노트북은 Common Voice의 `audio`·`sentence` 열을 사용하며, 도메인 학습의 `name`·`text`와 구분합니다. 다운로드 중간 WAV의 설정은 21 kHz이지만 최종 Dataset의 입력은 16 kHz입니다.

### Public Datasets

아래 수량은 **2026-10-30에 확인한 Hugging Face 공개 스냅샷** 기준입니다.

| 분야 | 데이터셋 | Train | Test | 합계 |
|---|---|---:|---:|---:|
| 개발 | [korean-audio-text-develop](https://huggingface.co/datasets/imTak/korean-audio-text-develop) | 4,111 | 1,028 | 5,139 |
| 경제 | [korean-audio-text-economy](https://huggingface.co/datasets/imTak/korean-audio-text-economy) | 34,548 | 8,637 | 43,185 |

기존 코드의 `imTak/korean-speak-Develop`과 `imTak/Economy`는 현재 위 데이터셋으로 연결됩니다. 저장된 노트북 출력은 다른 실행 시점의 기록이므로 현재 데이터 수와 동일하다고 가정하지 않습니다.

## Training

### Experiments & Configuration

기본 모델은 `openai/whisper-large-v3-turbo`입니다. 한국어 모델 카드에는 Zeroth-Korean·Common Voice 17 학습이 기록돼 있으며, 현재 [`fine_tune_whisper.ipynb`](./fine_tune_whisper.ipynb)의 셀은 Common Voice 17 한국어 `train+validation` / `test`를 사용하도록 설정돼 있습니다.

도메인 학습은 한국어 체크포인트 `imTak/whisper_large_v3_ko_ft_ft`에서 시작합니다. 이 ID의 현재 공개 이름은 [`imTak/whisper_large_v3_turbo_Korean2`](https://huggingface.co/imTak/whisper_large_v3_turbo_Korean2)입니다.

| 항목 | 한국어 기본 노트북 | 개발 분야 Python | 경제 분야 Python |
|---|---|---|---|
| 파일 | [`fine_tune_whisper.ipynb`](./fine_tune_whisper.ipynb) | [`fine_tune_whisper3.py`](./fine_tune_whisper3.py) | [`fine_tune_whisper2.py`](./fine_tune_whisper2.py) |
| 시작 모델 | Whisper large-v3-turbo | 한국어 파인튜닝 모델 | 한국어 파인튜닝 모델 |
| 학습 batch / GPU | 16 | 32 | 32 |
| 평가 batch / GPU | 8 | 32 | 32 |
| 최대 step | 4,000 | 8,000 | 8,000 |
| 평가 간격 | 1,000 step | 500 step | 2,000 step |
| 저장 간격 | 2,000 step | 1,000 step | 2,000 step |

공통 설정은 learning rate `1e-5`, warmup `500` step, gradient accumulation `1`, FP16, gradient checkpointing, `predict_with_generate=True`, `generation_max_length=225`입니다. 언어는 `ko`, 작업은 `transcribe`로 지정했습니다.

도메인 `.ipynb`에는 batch 16·4,000 step의 별도 실험 설정도 남아 있습니다. 위 표는 **현재 각 파일의 설정**을 요약한 것으로, 모든 공개 모델이 같은 설정으로 만들어졌다는 의미는 아닙니다.

### Environment & Running

기존 실험은 Python 3.10과 Transformers 4.45 계열을 사용했습니다. 아래는 원본의 Dataset Audio 반환 형식을 사용하는 환경 구성 예시입니다. CUDA에 맞는 PyTorch 환경을 사용하고 FFmpeg를 설치합니다.

```bash
git clone https://github.com/HahnGyuTak/openai-whisper-finetuning.git
cd openai-whisper-finetuning

conda create -n whisper-ko python=3.10 -y
conda activate whisper-ko
conda install -c conda-forge ffmpeg -y

python -m pip install -r requirements.txt
python -m pip install "transformers==4.45.2" "datasets[audio]==2.21.0" \
  evaluate jiwer soundfile pandas tensorboard gradio jupyterlab
```

현재 파일에는 실험 서버의 GPU 번호와 로컬 체크포인트 경로가 남아 있으므로 **처음 학습할 때 아래 설정을 수정한 후 실행**합니다.

1. `.py` 파일 상단의 `CUDA_VISIBLE_DEVICES='6'` 또는 `'7'`을 사용할 GPU 번호로 바꾸거나 제거합니다. 파일 안에서 값을 다시 지정하므로 셸에서만 GPU 번호를 바꾸면 덮어써집니다.
2. `TrainingArguments`의 `resume_from_checkpoint` 경로를 제거하거나 `None`으로 바꾸고, `trainer.train(".../checkpoint-...")`도 `trainer.train()`으로 바꿉니다. 이어 학습할 때는 실제 존재하는 경로를 `trainer.train(resume_from_checkpoint=...)`에 지정합니다.
3. `model_id`, 데이터셋 ID, `output_dir`를 확인합니다. Hub에 업로드한다면 `hub_model_id="본인계정/모델명"`을 명시하고 `huggingface-cli login`으로 인증합니다.
4. 로컬 실험만 수행한다면 `push_to_hub=False`로 바꾸고 **마지막 `trainer.push_to_hub(**kwargs)` 호출도 비활성화**합니다. W&B를 쓰지 않으면 `report_to=[]`로 설정합니다.
5. GPU 메모리에 맞춰 batch size를 조절합니다. batch를 줄이고 유효 batch를 유지하려면 gradient accumulation을 함께 조정합니다.

```bash
# 위 설정을 수정한 뒤, 원하는 실험 한 가지를 실행합니다.
python fine_tune_whisper3.py  # 개발 분야
# 또는
python fine_tune_whisper2.py  # 경제 분야

# 노트북으로 실험하려면
jupyter lab
```

이 저장소는 당시의 실험 스크립트와 실행 기록을 보존합니다. 환경 예시가 전체 의존성의 완전한 lockfile이나 GPU 학습 재현 검증을 대신하지는 않습니다.

## Evaluation

`compute_metrics()`는 생성 토큰과 정답 토큰을 문자열로 복원한 뒤 `evaluate.load("wer")`로 WER를 계산합니다. 정답의 `-100`은 decoding 전에 padding token으로 복원하며, 결과는 **퍼센트 값**으로 기록합니다.

```text
WER (%) = (치환 + 삭제 + 삽입) / 정답 단어 수 × 100
```

`metric_for_best_model="wer"`, `greater_is_better=False`, `load_best_model_at_end=True`로 낮은 WER의 저장 체크포인트를 선택합니다. 별도의 평가용 텍스트 정규화를 적용하지 않아 한국어 띄어쓰기·표기 차이도 결과에 영향을 줄 수 있습니다.

### Reported Results

아래 값은 **각 공개 모델 카드의 model-index에 기록된 WER**입니다.

| 공개 모델 | 카드에 기록된 평가 데이터 | WER (%) |
|---|---|---:|
| [Korean2](https://huggingface.co/imTak/whisper_large_v3_turbo_Korean2) | `Bingsu/zeroth-korean` · test | 5.2703 |
| [Develop](https://huggingface.co/imTak/whisper_large_v3_turbo_korean_Develop) | `imTak/korean-speak-Develop` · test | 16.4370 |
| [Economy](https://huggingface.co/imTak/whisper_large_v3_turbo_korean_Economy) | `imTak/Economy` · test | 44.9921 |

평가 데이터를 서로 다르게 사용한 결과이므로 모델 간 순위나 원모델 대비 개선율로 비교하지 않습니다. 현재 노트북 설정·저장된 실행 로그·공개 모델 카드가 모두 같은 실험을 가리킨다고 가정하지 않습니다.

**현재 평가 구성의 범위:** 도메인 데이터는 세그먼트 단위 무작위 분할이며 영상·화자 단위로 분리한 평가가 아닙니다. 학습 중 `test` split을 `eval_dataset`으로 사용해 체크포인트를 선택하므로, 이 split은 검증셋 역할도 겸합니다. 독립 최종 테스트나 다른 회의 환경에서의 일반화 성능을 보장하는 수치로 해석하지 않습니다.

## Public Models & Inference

| 용도 | Transformers 모델 | faster-whisper / CTranslate2 모델 |
|---|---|---|
| 한국어 | [whisper_large_v3_turbo_Korean2](https://huggingface.co/imTak/whisper_large_v3_turbo_Korean2) | [faster-whisper_Korean_L3turbo](https://huggingface.co/imTak/faster-whisper_Korean_L3turbo) |
| 개발 | [whisper_large_v3_turbo_korean_Develop](https://huggingface.co/imTak/whisper_large_v3_turbo_korean_Develop) | [faster-whisper_large_v3_turbo_korean_Develop](https://huggingface.co/imTak/faster-whisper_large_v3_turbo_korean_Develop) |
| 경제 | [whisper_large_v3_turbo_korean_Economy](https://huggingface.co/imTak/whisper_large_v3_turbo_korean_Economy) | [faster-whisper_large_v3_turbo_korean_Economy](https://huggingface.co/imTak/faster-whisper_large_v3_turbo_korean_Economy) |

### Transformers

아래 예시는 공개된 개발 분야 모델로 `meeting.wav`를 전사합니다. GPU에서는 FP16, CPU에서는 FP32를 사용합니다.

```python
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

model_id = "imTak/whisper_large_v3_turbo_korean_Develop"
use_cuda = torch.cuda.is_available()
dtype = torch.float16 if use_cuda else torch.float32

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
processor = AutoProcessor.from_pretrained(model_id)

asr = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=dtype,
    device=0 if use_cuda else -1,
    chunk_length_s=30,
)
result = asr(
    "meeting.wav",
    generate_kwargs={"language": "korean", "task": "transcribe"},
)
print(result["text"])
```

### faster-whisper on CPU

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "imTak/faster-whisper_large_v3_turbo_korean_Develop",
    device="cpu",
    compute_type="int8",
)
segments, info = model.transcribe("meeting.wav", language="ko", beam_size=5)
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

## Repository Guide

| 파일 | 내용 |
|---|---|
| [`fine_tune_whisper.ipynb`](./fine_tune_whisper.ipynb) | 한국어 기본 학습, WER 평가, 데모·변환 실험 |
| [`fine_tune_whisper2.py`](./fine_tune_whisper2.py), [노트북](./fine_tune_whisper2.ipynb) | 경제 분야 추가 학습 |
| [`fine_tune_whisper3.py`](./fine_tune_whisper3.py), [노트북](./fine_tune_whisper3.ipynb) | 개발 분야 추가 학습 |
| [`make_hfds_economy.py`](./make_hfds_economy.py), [`vutube_custom.py`](./vutube_custom.py) | 영상·자막 수집, 분할, Dataset 구성·공개 |
| [`tube_dataset.ipynb`](./tube_dataset.ipynb), [`test.ipynb`](./test.ipynb) | 데이터 처리·화자 분리·추론 실험 |
| [`wav2md.py`](./wav2md.py) | 화자 분리·전사·회의 요약을 연결하는 실험 코드 |
| [`requirements.txt`](./requirements.txt), [`install.sh`](./install.sh) | 당시 패키지·환경 구성 기록 |

`wav2md.py`에는 로컬 파일·인증 설정과 비활성화된 처리 단계가 남아 있습니다. 앱의 전체 실행·연동 흐름은 [Mote AI 저장소](https://github.com/MettingApp/AI)를 함께 참고합니다.

## References

- [Hugging Face — Fine-Tune Whisper for Multilingual ASR](https://huggingface.co/blog/fine-tune-whisper): 학습 파이프라인·collator·WER 계산의 기반 예제
- [Transformers 4.45.2 — Whisper](https://huggingface.co/docs/transformers/v4.45.2/en/model_doc/whisper), [Trainer](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/trainer)
- [OpenAI Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [Mote 발표자료](https://github.com/user-attachments/files/19148912/Mote.pdf): 모델 파일 용량 비교와 팀 서비스 구성

