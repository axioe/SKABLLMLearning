"""[LLM] 정비 어시스턴트: 이상탐지 결과 + 검색된 정비 문서로 점검 안내를 만든다.

LLM은 고장을 직접 판단하지 않는다. 딥러닝 판정(JSON)과 RAG 문서를 근거로
원인 후보, 점검 체크리스트, 긴급도별 대응을 정리하는 역할만 한다.
"""
import json
import time
from dataclasses import dataclass, field

import config
from anomaly_detector import DetectionResult
from knowledge_base import MaintenanceKnowledgeBase

SYSTEM_PROMPT = """당신은 원심 펌프 설비의 정비 보조 AI입니다. 현장 작업자에게 한국어로 점검 안내를 합니다.

반드시 지킬 규칙:
1. [이상탐지 결과]와 [참고 문서]에 있는 내용만 근거로 씁니다. 문서에 없는 수치, 부품, 절차는 지어내지 않습니다.
2. 고장 원인을 확정하지 않습니다. "가능성이 있습니다", "확인이 필요합니다"처럼 후보로 말합니다.
3. 센서 기여도는 평소와 가장 다르게 움직인 센서일 뿐, 원인의 증명이 아니라는 점을 기억합니다.
4. 근거로 쓴 문서는 문장 끝에 [1], [2]처럼 번호를 붙입니다.
5. 짧고 구체적으로 씁니다. 작업자가 바로 따라 할 수 있는 순서로 정리합니다."""

ADVICE_TEMPLATE = """[이상탐지 결과]
{result_json}

[참고 문서]
{context}

위 정보를 바탕으로 아래 형식의 Markdown으로 답하세요.

### 상태 요약
판정, 긴급도, 가장 크게 벗어난 센서를 1~2문장으로.

### 가능성 있는 원인 후보
최대 3개. 각 후보마다 근거가 된 센서 변화와 문서 번호를 함께 씁니다.

### 점검 체크리스트
우선순위 순서로 최대 5개. "- [ ] " 형식으로 씁니다.

### 대응 방법
긴급도에 맞는 대응을 2~3문장으로.

### 주의사항
이 안내의 한계를 1~2문장으로."""

NORMAL_TEMPLATE = """[이상탐지 결과]
{result_json}

[참고 문서]
{context}

판정이 정상입니다. 아래 형식의 Markdown으로 짧게 답하세요.

### 상태 요약
정상 판정과 점수 수준을 1~2문장으로.

### 관찰 포인트
정상 범위 안에서 상대적으로 크게 움직인 센서가 있다면 무엇을 지켜보면 되는지 1~2개."""

FOLLOWUP_TEMPLATE = """[이상탐지 결과]
{result_json}

[참고 문서]
{context}

[작업자 질문]
{question}

위 정보만 근거로 질문에 3~6문장으로 답하세요. 문서에 답이 없으면 "제공된 문서에서는 확인할 수 없습니다"라고 말하고, 확인할 방법을 제안하세요."""


@dataclass
class Advice:
    text: str
    sources: list = field(default_factory=list)   # RetrievedChunk
    query: str = ''
    model: str = ''
    seconds: float = 0.0


def ollama_running(timeout: float = 1.5) -> bool:
    """Ollama 서버가 켜져 있는지 확인한다 (Streamlit Cloud에서는 항상 False)."""
    try:
        import requests
        return requests.get(f'{config.OLLAMA_BASE_URL}/api/tags', timeout=timeout).ok
    except Exception:
        return False


def resolve_backend(backend: str = config.LLM_BACKEND) -> str:
    """auto면 실행 환경에 맞는 백엔드를 고른다: 로컬 Ollama → Gemini API 순서."""
    if backend in ('ollama', 'gemini'):
        return backend
    if ollama_running():
        return 'ollama'
    return 'gemini' if config.GEMINI_API_KEY else 'ollama'


class LLMClient:
    """Ollama(로컬)와 Gemini(API)를 같은 방식으로 호출한다."""

    def __init__(self, backend: str = config.LLM_BACKEND, model: str | None = None,
                 temperature: float = config.LLM_TEMPERATURE):
        self.backend = resolve_backend(backend)
        self.temperature = temperature
        self.model = model or (config.OLLAMA_MODEL if self.backend == 'ollama' else config.GEMINI_MODEL)
        self._client = None

    @property
    def name(self) -> str:
        return f'{self.backend}:{self.model}'

    def _get_client(self):
        if self._client is None:
            if self.backend == 'ollama':
                from langchain_ollama import ChatOllama
                # num_ctx: 한글 문서 4개 + 판정 JSON이 기본 4096 토큰을 넘지 않도록 늘린다
                # reasoning=False: gemma4처럼 생각(thinking) 기능이 있는 모델은 끄고 바로 답하게 해 속도를 높인다
                extra = {'reasoning': False} if self._supports_thinking() else {}
                self._client = ChatOllama(model=self.model, base_url=config.OLLAMA_BASE_URL,
                                          temperature=self.temperature, num_ctx=8192, **extra)
            elif self.backend == 'gemini':
                if not config.GEMINI_API_KEY:
                    raise RuntimeError('GEMINI_API_KEY가 없습니다. .env 파일에 키를 넣어 주세요.')
                from google import genai
                self._client = genai.Client(api_key=config.GEMINI_API_KEY)
            else:
                raise ValueError(f'지원하지 않는 LLM_BACKEND: {self.backend}')
        return self._client

    def _supports_thinking(self) -> bool:
        try:
            import requests
            info = requests.post(f'{config.OLLAMA_BASE_URL}/api/show', json={'model': self.model}, timeout=5).json()
            return 'thinking' in info.get('capabilities', [])
        except Exception:
            return False

    def chat(self, system: str, user: str) -> str:
        client = self._get_client()
        if self.backend == 'ollama':
            from langchain_core.messages import HumanMessage, SystemMessage
            return client.invoke([SystemMessage(system), HumanMessage(user)]).content
        from google.genai import types
        resp = client.models.generate_content(
            model=self.model, contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=self.temperature))
        if not resp.text:   # 안전 필터 등으로 빈 응답이 오면 화면이 깨지지 않게 안내 문구를 돌려준다
            return '(LLM이 빈 응답을 돌려주었습니다. 잠시 후 다시 시도해 주세요.)'
        return resp.text

    def status(self) -> tuple:
        """LLM을 호출하지 않고 설정만 확인한다 (화면을 열 때마다 API 비용·시간이 들지 않게). (가능 여부, 메시지)"""
        if self.backend == 'gemini':
            if not config.GEMINI_API_KEY:
                return False, 'GEMINI_API_KEY가 없습니다. Streamlit Cloud의 Secrets(또는 .env)에 키를 넣어 주세요.'
            return True, f'{self.name} 사용'
        if not ollama_running():
            return False, ('Ollama가 실행 중이 아닙니다. 로컬에서는 Ollama를 켜고, 배포 서버에서는 '
                           'GEMINI_API_KEY를 설정해 주세요.')
        return True, f'{self.name} 사용'

    def is_available(self) -> tuple:
        """연결 확인. (가능 여부, 메시지)"""
        try:
            self.chat('짧게 답하세요.', '연결 확인: OK라고만 답하세요.')
            return True, f'{self.name} 연결됨'
        except Exception as e:   # 시연 화면에서 원인을 보여주기 위해 메시지만 돌려준다
            return False, f'{self.name} 연결 실패: {e}'


class MaintenanceAdvisor:
    """이상탐지 결과 → 검색 질의 → 문서 검색 → 프롬프트 → LLM 답변."""

    def __init__(self, kb: MaintenanceKnowledgeBase, llm: LLMClient):
        self.kb = kb
        self.llm = llm

    @staticmethod
    def build_query(result: DetectionResult) -> str:
        """판정 결과를 검색 질의로 바꾼다. 예: '모터 온도 낮음 유체 온도 낮음 원인 점검'.

        '정상보다', '긴급도'처럼 여러 문서에 흔한 단어는 넣지 않는다 (BM25 점수가 센서 이름보다 커지는 문제).
        진동 1·2는 '진동' 하나로 합치고, 오차 비중 10% 이상인 센서만 최대 2개 쓴다.
        """
        parts = []
        for c in result.top_sensors:
            name = '진동' if c.name_ko.startswith('진동') else c.name_ko
            part = f"{name} {'높음' if c.z > 0 else '낮음'}"
            if (not parts or c.share >= 10) and name not in ' '.join(parts):
                parts.append(part)
        return ' '.join(parts[:2]) + (' 원인 점검' if result.is_anomaly else ' 관찰')

    def retrieve(self, result: DetectionResult, mode: str = 'hybrid') -> list:
        """센서 증상 문서 + 긴급도별 대응 문서(이상일 때 항상 포함)."""
        chunks = self.kb.search(self.build_query(result), k=config.TOP_K - 1, mode=mode)
        if result.is_anomaly:
            chunks += [c for c in self.kb.section('긴급도별 대응') if c.text not in {x.text for x in chunks}]
        return chunks

    @staticmethod
    def format_context(chunks: list) -> str:
        if not chunks:
            return '(참고 문서 없음: 일반 지식으로 답하되, 확실하지 않은 내용은 확인이 필요하다고 밝히세요.)'
        return '\n\n'.join(f'[{i}] {c.text}' for i, c in enumerate(chunks, 1))

    def advise(self, result: DetectionResult, use_rag: bool = True, mode: str = 'hybrid') -> Advice:
        query = self.build_query(result)
        chunks = self.retrieve(result, mode) if use_rag else []
        template = ADVICE_TEMPLATE if result.is_anomaly else NORMAL_TEMPLATE
        prompt = template.format(result_json=json.dumps(result.to_llm_dict(), ensure_ascii=False, indent=1),
                                 context=self.format_context(chunks))
        t0 = time.perf_counter()
        text = self.llm.chat(SYSTEM_PROMPT, prompt)
        return Advice(text=text, sources=chunks, query=query, model=self.llm.name,
                      seconds=time.perf_counter() - t0)

    def answer(self, question: str, result: DetectionResult | None) -> Advice:
        """후속 질문: 질문 + 현재 판정의 기여 센서로 다시 검색해 답한다."""
        sensors = ' '.join(c.name_ko for c in result.top_sensors[:2]) if result else ''
        query = f'{question} {sensors}'.strip()
        chunks = self.kb.search(query)
        result_json = json.dumps(result.to_llm_dict(), ensure_ascii=False, indent=1) if result else '(아직 판정 없음)'
        prompt = FOLLOWUP_TEMPLATE.format(result_json=result_json, context=self.format_context(chunks),
                                          question=question)
        t0 = time.perf_counter()
        text = self.llm.chat(SYSTEM_PROMPT, prompt)
        return Advice(text=text, sources=chunks, query=query, model=self.llm.name,
                      seconds=time.perf_counter() - t0)


if __name__ == '__main__':
    from anomaly_detector import PumpAnomalyDetector

    detector = PumpAnomalyDetector()
    kb = MaintenanceKnowledgeBase()
    kb.warm_up()    # 임베딩 모델을 LLM보다 먼저 올린다 (메모리 여유가 있을 때)
    advisor = MaintenanceAdvisor(kb, LLMClient())
    print(advisor.llm.is_available()[1])
    tl = detector.score_timeline('other/7')
    r = detector.detect_at('other/7', int(tl.loc[tl['anomaly'] == 1, 'row'].median()))
    adv = advisor.advise(r)
    print('질의:', adv.query)
    print('출처:', [c.cite() for c in adv.sources])
    print(f'({adv.seconds:.1f}초)\n', adv.text)
