"""[RAG] 정비 지식 검색: knowledge/ 폴더의 Markdown 문서를 나눠 하이브리드 검색한다.

검색 방식 (강의 6.9 RAG 종류별 비교의 Hybrid RAG)
    - 벡터 검색: HuggingFace 임베딩 + FAISS (의미가 비슷한 문장)
    - 키워드 검색: BM25 (센서 이름, 고장 이름처럼 정확한 단어)
    - 최종 점수 = HYBRID_WEIGHT × 벡터 점수 + (1 − HYBRID_WEIGHT) × BM25 점수 (각각 0~1로 정규화)

비교 실험을 위해 mode='vector' / 'bm25' / 'hybrid'를 고를 수 있다.

임베딩 방식 (config.EMBEDDING_BACKEND)
    - huggingface: 로컬 sentence-transformers (노트북 실행, 인터넷 불필요)
    - gemini: Gemini 임베딩 API (Streamlit Cloud: torch를 설치하지 않아 배포가 가볍다)
    - none 또는 임베딩 실패: BM25만으로 검색한다 (앱이 멈추지 않게)
"""
import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

import config


@dataclass
class RetrievedChunk:
    text: str
    source: str     # 파일 이름
    section: str    # 문서 제목 > 섹션 제목
    score: float

    def cite(self) -> str:
        return f'{self.source} · {self.section}'


def tokenize(text: str) -> list:
    """BM25용 토큰: 단어 + 한글 단어의 2글자 조각. 조사가 붙어도('유량이', '유량을') 같은 조각('유량')이 남는다."""
    words = re.findall(r'[A-Za-z0-9]+|[가-힣]+', text.lower())
    tokens = list(words)
    for w in words:
        if re.match(r'[가-힣]', w) and len(w) > 2:
            tokens += [w[i:i + 2] for i in range(len(w) - 1)]
    return tokens


class GeminiEmbeddings:
    """LangChain Embeddings 인터페이스를 따르는 Gemini 임베딩 (google-genai 사용)."""

    def __init__(self, model: str = config.GEMINI_EMBEDDING_MODEL, api_key: str = config.GEMINI_API_KEY):
        from google import genai
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def _embed(self, texts: list, task: str) -> list:
        from google.genai import types
        out = []
        for i in range(0, len(texts), 50):    # 한 번에 보낼 수 있는 개수 제한을 피해 나누어 보낸다
            resp = self.client.models.embed_content(model=self.model, contents=texts[i:i + 50],
                                                    config=types.EmbedContentConfig(task_type=task))
            for e in resp.embeddings:
                v = np.asarray(e.values, dtype='float32')
                out.append((v / (np.linalg.norm(v) + 1e-12)).tolist())   # 코사인 변환식(1 − d/2)을 쓰려면 정규화 필요
        return out

    def embed_documents(self, texts: list) -> list:
        return self._embed(list(texts), 'RETRIEVAL_DOCUMENT')

    def embed_query(self, text: str) -> list:
        return self._embed([text], 'RETRIEVAL_QUERY')[0]

    def __call__(self, text: str) -> list:   # 구버전 FAISS 호환
        return self.embed_query(text)


def pick_embedding_backend(backend: str = config.EMBEDDING_BACKEND) -> str:
    """auto면 설치된 라이브러리와 API 키를 보고 고른다."""
    if backend != 'auto':
        return backend
    try:
        import langchain_huggingface  # noqa: F401
        import sentence_transformers  # noqa: F401
        return 'huggingface'
    except ImportError:
        return 'gemini' if config.GEMINI_API_KEY else 'none'


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


class MaintenanceKnowledgeBase:
    """정비 문서를 청크로 나누고, 벡터 + BM25 하이브리드 검색을 제공한다."""

    def __init__(self, knowledge_dir=config.KNOWLEDGE_DIR, index_dir=config.INDEX_DIR):
        self.knowledge_dir = knowledge_dir
        self.index_dir = index_dir
        self.chunks = self._load_chunks()
        self.embedding_backend = pick_embedding_backend()
        self.vector_error = ''     # 벡터 검색을 쓸 수 없을 때의 이유 (화면에 표시)
        self._embeddings = None
        self._vectorstore = None
        self._bm25 = None

    # ---------- 1. 문서 로드와 분할 ----------
    def _load_chunks(self) -> list:
        """Markdown을 '##' 섹션 단위로 먼저 나누고, 긴 섹션은 CHUNK_SIZE로 다시 나눈다."""
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
                                                  separators=['\n\n', '\n', '. ', ' '])
        chunks = []
        for path in sorted(self.knowledge_dir.glob('*.md')):
            text = path.read_text(encoding='utf-8')
            title = re.search(r'^# (.+)$', text, re.M).group(1).strip()
            for part in re.split(r'^## ', text, flags=re.M)[1:]:
                heading, _, body = part.partition('\n')
                section = f'{title} > {heading.strip()}'
                for piece in splitter.split_text(body.strip()):
                    # 청크 앞에 제목을 붙여 두면 검색할 때 문맥을 잃지 않는다
                    chunks.append({'text': f'[{section}]\n{piece}', 'source': path.name, 'section': section})
        return chunks

    @property
    def embedding_name(self) -> str:
        return {'huggingface': config.EMBEDDING_MODEL, 'gemini': config.GEMINI_EMBEDDING_MODEL}.get(
            self.embedding_backend, 'BM25만 사용')

    def _fingerprint(self) -> str:
        raw = json.dumps([c['text'] for c in self.chunks], ensure_ascii=False) + self.embedding_name
        return hashlib.md5(raw.encode('utf-8')).hexdigest()

    # ---------- 2. 인덱스 (처음 한 번 만들고 저장) ----------
    @property
    def embeddings(self):
        if self._embeddings is None:
            if self.embedding_backend == 'huggingface':
                from langchain_huggingface import HuggingFaceEmbeddings
                self._embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL,
                                                         encode_kwargs={'normalize_embeddings': True})
            elif self.embedding_backend == 'gemini':
                self._embeddings = GeminiEmbeddings()
            else:
                raise RuntimeError('임베딩을 사용하지 않는 설정입니다.')
        return self._embeddings

    @property
    def vector_ok(self) -> bool:
        """벡터 검색 사용 가능 여부. 처음 호출할 때 인덱스를 만들고, 실패하면 BM25만 쓰도록 바꾼다."""
        if self.embedding_backend == 'none':
            self.vector_error = self.vector_error or '임베딩(로컬 모델 또는 GEMINI_API_KEY)이 없어 BM25만 사용합니다.'
            return False
        try:
            _ = self.vectorstore
            return True
        except Exception as e:   # API 키 오류, 할당량 초과, 네트워크 등
            self.vector_error = f'벡터 검색을 쓸 수 없어 BM25만 사용합니다: {str(e)[:150]}'
            self.embedding_backend = 'none'
            return False

    @property
    def vectorstore(self):
        """FAISS 인덱스. 문서가 바뀌지 않았으면 저장된 인덱스를 불러온다."""
        if self._vectorstore is None:
            from langchain_community.vectorstores import FAISS

            index_dir = self.index_dir / self.embedding_backend   # 임베딩 방식마다 벡터가 달라 따로 저장
            stamp = index_dir / 'fingerprint.txt'
            fp = self._fingerprint()
            if stamp.exists() and stamp.read_text() == fp:
                # 직접 만든 인덱스 파일만 불러오므로 역직렬화를 허용한다
                self._vectorstore = FAISS.load_local(str(index_dir), self.embeddings,
                                                     allow_dangerous_deserialization=True)
            else:
                self._vectorstore = FAISS.from_texts(
                    [c['text'] for c in self.chunks], self.embeddings,
                    metadatas=[{'i': i} for i in range(len(self.chunks))])
                try:
                    index_dir.mkdir(parents=True, exist_ok=True)
                    self._vectorstore.save_local(str(index_dir))
                    stamp.write_text(fp)
                except OSError:
                    pass   # 저장하지 못해도 메모리의 인덱스로 검색한다
        return self._vectorstore

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(c['text']) for c in self.chunks])
        return self._bm25

    # ---------- 3. 검색 ----------
    def _vector_scores(self, query: str) -> np.ndarray:
        """모든 청크의 코사인 유사도. 정규화된 벡터의 FAISS L2² 거리 d에서 cos = 1 − d/2."""
        scores = np.zeros(len(self.chunks))
        for doc, dist in self.vectorstore.similarity_search_with_score(query, k=len(self.chunks)):
            scores[doc.metadata['i']] = 1 - dist / 2
        return scores

    def search(self, query: str, k: int = config.TOP_K, mode: str = 'hybrid') -> list:
        if mode in ('vector', 'hybrid') and not self.vector_ok:
            mode = 'bm25'
        if mode == 'vector':
            scores = _minmax(self._vector_scores(query))
        elif mode == 'bm25':
            scores = _minmax(np.asarray(self.bm25.get_scores(tokenize(query))))
        else:
            w = config.HYBRID_WEIGHT
            scores = (w * _minmax(self._vector_scores(query))
                      + (1 - w) * _minmax(np.asarray(self.bm25.get_scores(tokenize(query)))))
        top = np.argsort(scores)[::-1][:k]
        return [RetrievedChunk(self.chunks[i]['text'], self.chunks[i]['source'],
                               self.chunks[i]['section'], float(scores[i])) for i in top]

    def section(self, heading: str) -> list:
        """섹션 제목으로 청크를 그대로 가져온다 (검색 없이 항상 넣을 문서용)."""
        return [RetrievedChunk(c['text'], c['source'], c['section'], 1.0)
                for c in self.chunks if c['section'].endswith(heading)]

    def warm_up(self):
        """앱 시작 시 임베딩 모델과 인덱스를 미리 준비한다. 벡터 검색이 실패해도 BM25는 준비된다."""
        _ = self.vector_ok, self.bm25
        return len(self.chunks)


if __name__ == '__main__':
    kb = MaintenanceKnowledgeBase()
    print(f'청크 {kb.warm_up()}개 · 임베딩 {kb.embedding_backend} ({kb.embedding_name}) {kb.vector_error}')
    for q in ['진동이 정상보다 높음 로터 불균형 점검', '유체 온도 상승 원인', '유량 감소 밸브']:
        print(f'\n질의: {q}')
        for mode in ['vector', 'bm25', 'hybrid']:
            print(f'  {mode:6s}:', [r.section.split(' > ')[-1] for r in kb.search(q, k=3, mode=mode)])
