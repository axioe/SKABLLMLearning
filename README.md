# 펌프 이상탐지 + LLM 정비 어시스턴트

딥러닝 프로젝트(SKAB 펌프 이상탐지)를 LLM 서비스로 확장한 프로젝트입니다.
정상 운전만 학습한 **LSTM 오토인코더(DL)**가 이상을 판정하면, **정비 문서 검색(RAG)**과 **LLM**이 점검 안내를 만듭니다.

LLM은 고장을 직접 판단하지 않습니다. 딥러닝 판정 결과(JSON)와 검색된 정비 문서를 근거로 원인 후보, 점검 체크리스트, 긴급도별 대응을 정리합니다.

## 시스템 구조

```
센서 입력 (실험 CSV 재생 또는 슬라이더)
  → [DL]  PumpAnomalyDetector      LSTM AE 이상점수·임계값, 비교 모델 5종, 센서별 기여도
  → 판정 결과 JSON                  이상 여부, 긴급도, 기여 센서 상위 3개 (실험명·정답은 넣지 않음)
  → [RAG] MaintenanceKnowledgeBase BM25 + FAISS 하이브리드 검색 (정비 문서 4종)
  → [LLM] MaintenanceAdvisor       로컬: Ollama gemma4:e2b / 배포: Gemini API
  → 출력  Streamlit(배포)·Gradio(로컬) 화면 · 후속 질문 챗봇 · 정비 리포트 (Markdown·PDF)
```

## 실행 환경별 동작

| 환경 | LLM | 임베딩 (벡터 검색) | 화면 |
| --- | --- | --- | --- |
| 로컬 노트북 | Ollama gemma4:e2b | jhgan/ko-sroberta-multitask (sentence-transformers) | `streamlit_app.py` 또는 `app.py`(Gradio) |
| Streamlit Cloud | Gemini 2.5 Flash (Secrets의 키) | gemini-embedding-001 | `streamlit_app.py` |
| API 키 없음 | 사용 불가 (안내 문구 표시) | 사용 불가 → BM25만 | 딥러닝 판정과 문서 검색은 정상 동작 |

`LLM_BACKEND=auto`(기본)는 Ollama가 켜져 있으면 Ollama, 아니면 Gemini를 씁니다. Streamlit Cloud에서는 Ollama를 띄울 수 없고 torch 설치가 무거워, LLM과 임베딩을 모두 Gemini API로 바꿉니다.

## 파일 구성

| 파일 | 클래스 | 역할 |
| --- | --- | --- |
| `config.py` | - | 경로, 센서 이름, LLM·RAG 설정 (.env 또는 Streamlit Secrets) |
| `train_models.py` | - | 딥러닝 프로젝트 원본 코드 (모델 구조, 점수 계산) |
| `anomaly_detector.py` | `PumpAnomalyDetector`, `DetectionResult` | 모델 로드, 이상 판정, 센서별 기여도, LLM용 dict 변환 |
| `knowledge_base.py` | `MaintenanceKnowledgeBase`, `GeminiEmbeddings` | 문서 분할, BM25 + FAISS 하이브리드 검색, 임베딩 실패 시 BM25 전환 |
| `llm_advisor.py` | `LLMClient`, `MaintenanceAdvisor` | Ollama/Gemini 호출(자동 선택), 정비 안내, 후속 질문 답변 |
| `report_builder.py` | `ReportBuilder` | 정비 리포트 Markdown·PDF 생성 |
| `evaluate_llm.py` | `AdvisorEvaluator` | LLM 모델 × RAG 사용 여부 비교 실험 |
| `ui_common.py` | - | 두 화면이 같이 쓰는 그래프·표·오류 문구 |
| `streamlit_app.py` | - | Streamlit 화면 (배포용) |
| `app.py` | - | Gradio 화면 (로컬용) |
| `knowledge/` | - | RAG 지식 문서 4종 |
| `models/` | - | 딥러닝 프로젝트에서 학습한 모델 (시연 실험 4개 × 5종) |
| `dataset/SKAB/` | - | 시연 실험 CSV 4개 |
| `requirements.txt` / `packages.txt` | - | Streamlit Cloud 설치 목록 / 한글 글꼴(fonts-nanum) |
| `requirements-local.txt` | - | 로컬 추가 패키지 (Ollama, 로컬 임베딩, Gradio) |

## Streamlit Community Cloud 배포

1. 이 폴더를 GitHub 저장소(예: `axioe/SKABLLMLearning`)에 올립니다. `.env`, `.streamlit/secrets.toml`, `vector_index/`는 `.gitignore`로 제외됩니다.
2. [share.streamlit.io](https://share.streamlit.io)에서 **Create app** → 저장소, 브랜치 `main`, Main file `streamlit_app.py`를 고릅니다.
3. **Advanced settings**에서 Python 버전을 **3.12**로 고르고, Secrets에 다음을 넣습니다 (`.streamlit/secrets.toml.example` 참고).
   ```toml
   GEMINI_API_KEY = "Google AI Studio에서 발급한 키"
   ```
4. 처음 배포는 TensorFlow 설치 때문에 몇 분 걸립니다. 첫 판정 때 정비 문서 22개를 Gemini로 임베딩해 인덱스를 만듭니다.

## 로컬 실행 (Windows, Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-local.txt
ollama pull gemma4:e2b
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Gradio 화면은 `.venv\Scripts\python.exe app.py` (http://localhost:7861)로 실행합니다.

각 클래스 파일은 단독으로 실행해 확인할 수 있습니다.

```bash
.venv\Scripts\python.exe anomaly_detector.py   # 고장 4종 판정 결과 JSON
.venv\Scripts\python.exe knowledge_base.py     # 검색 방식별 결과 비교
.venv\Scripts\python.exe llm_advisor.py        # DL → RAG → LLM 전체 흐름
.venv\Scripts\python.exe evaluate_llm.py       # LLM 비교 실험 (outputs/llm_eval*.csv)
```

## 참고

- 시연 노트북: 16GB RAM, GPU 없음. gemma2(9B)는 메모리 부족으로 실패해 gemma4:e2b를 기본으로 사용합니다.
- 센서 기여도는 평소와 가장 다르게 움직인 센서이며, 고장 원인의 증명이 아닙니다.
- 지식 문서는 일반적인 원심 펌프 정비 지식과 딥러닝 프로젝트 분석 결과로 작성했습니다. 실제 설비에서는 제조사 매뉴얼을 우선합니다.
- Gemini 무료 사용량을 넘으면 정비 안내 자리에 "잠시 후 다시 시도" 안내가 나오고, 판정 화면은 그대로 유지됩니다.
- 데이터: Katser & Kozitsin, *Skoltech Anomaly Benchmark (SKAB)*, 2020. GPL-3.0. https://github.com/waico/SKAB
