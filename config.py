"""프로젝트 전체 설정: 경로, 센서 이름, LLM/임베딩 모델 선택.

설정은 환경변수(.env) 또는 Streamlit Cloud의 Secrets에서 읽는다.
    LLM_BACKEND=auto    (기본) Ollama가 켜져 있으면 Ollama, 없고 GEMINI_API_KEY가 있으면 Gemini
    LLM_BACKEND=ollama  로컬 실행 (데이터가 외부로 나가지 않음, 발표 노트북)
    LLM_BACKEND=gemini  API 실행 (Streamlit Cloud처럼 Ollama를 띄울 수 없는 서버)
    EMBEDDING_BACKEND=auto  sentence-transformers가 설치되어 있으면 로컬 임베딩, 없으면 Gemini 임베딩, 둘 다 없으면 BM25만
"""
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')


def _setting(name: str, default: str = '') -> str:
    """환경변수 → Streamlit Secrets 순서로 읽는다 (Streamlit 밖에서 실행해도 오류가 나지 않게)."""
    value = os.getenv(name)
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, default))
    except Exception:   # streamlit 미설치, secrets.toml 없음 등
        return default


def _writable(path: Path) -> Path:
    """쓰기 가능한 폴더를 돌려준다. 배포 서버에서 프로젝트 폴더에 쓸 수 없으면 임시 폴더를 쓴다."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / '.write_test'
        probe.write_text('ok')
        probe.unlink()
        return path
    except OSError:
        alt = Path(tempfile.gettempdir()) / 'pump_llm' / path.name
        alt.mkdir(parents=True, exist_ok=True)
        return alt

# ---------- 경로 ----------
MODEL_DIR = BASE_DIR / 'models'            # 딥러닝 프로젝트에서 학습한 모델 (재학습 없이 사용)
DATA_DIR = BASE_DIR / 'dataset' / 'SKAB'   # 시연용 실험 CSV 4개
KNOWLEDGE_DIR = BASE_DIR / 'knowledge'     # RAG 지식 문서 (Markdown)
INDEX_DIR = _writable(BASE_DIR / 'vector_index')   # FAISS 인덱스 캐시 (임베딩 방식별 하위 폴더)
OUTPUT_DIR = _writable(BASE_DIR / 'outputs')       # 정비 리포트, 평가 결과

# ---------- 센서 ----------
SENSOR_KO = {
    'Accelerometer1RMS': '진동 1', 'Accelerometer2RMS': '진동 2', 'Current': '전류', 'Pressure': '압력',
    'Temperature': '모터 온도', 'Thermocouple': '유체 온도', 'Voltage': '전압', 'Volume Flow RateRMS': '유량',
}
SENSOR_UNIT = {
    'Accelerometer1RMS': 'g', 'Accelerometer2RMS': 'g', 'Current': 'A', 'Pressure': 'Bar',
    'Temperature': '°C', 'Thermocouple': '°C', 'Voltage': 'V', 'Volume Flow RateRMS': 'L/min',
}

# ---------- 판정 ----------
# 점수 ÷ 임계값 비율로 긴급도를 나눈다 (딥러닝 프로젝트의 운영 임계값 ×1.25 / ×2.0 근거)
SEVERITY_LEVELS = [(2.0, '높음'), (1.0, '주의'), (0.0, '정상')]

# ---------- LLM ----------
LLM_BACKEND = _setting('LLM_BACKEND', 'auto').lower()   # auto / ollama / gemini
OLLAMA_BASE_URL = _setting('OLLAMA_BASE_URL', 'http://localhost:11434')
# 기본은 가벼운 gemma4:e2b (GPU 없는 16GB 노트북에서 gemma2 9B는 메모리 부족·느린 응답). gemma2는 비교 실험에 사용
OLLAMA_MODEL = _setting('OLLAMA_MODEL', 'gemma4:e2b')
GEMINI_MODEL = _setting('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_API_KEY = _setting('GEMINI_API_KEY', '')
LLM_TEMPERATURE = float(_setting('LLM_TEMPERATURE', '0.2'))   # 정비 안내는 일관성이 중요해 낮게 설정

# ---------- RAG ----------
EMBEDDING_BACKEND = _setting('EMBEDDING_BACKEND', 'auto').lower()   # auto / huggingface / gemini / none
EMBEDDING_MODEL = _setting('EMBEDDING_MODEL', 'jhgan/ko-sroberta-multitask')   # 로컬 (sentence-transformers)
GEMINI_EMBEDDING_MODEL = _setting('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')   # 배포 서버 (torch 설치 불필요)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
TOP_K = 4
HYBRID_WEIGHT = 0.5   # 최종 점수 = 0.5 × 벡터 유사도 + 0.5 × BM25 (둘 다 0~1로 정규화)
