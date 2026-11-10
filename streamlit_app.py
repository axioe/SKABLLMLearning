"""펌프 이상탐지 + LLM 정비 어시스턴트 (Streamlit 배포 화면)

로컬 실행:  .venv\\Scripts\\streamlit run streamlit_app.py
배포: Streamlit Community Cloud에서 이 파일을 Main file로 지정한다.
    - Ollama를 띄울 수 없으므로 Secrets에 GEMINI_API_KEY를 넣으면 Gemini로 자동 전환된다.
    - 키가 없어도 딥러닝 판정과 BM25 문서 검색은 동작하고, 정비 안내 자리에 설정 방법을 보여준다.
"""
import math
import os
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')

import streamlit as st

st.set_page_config(page_title='펌프 이상탐지 + LLM 정비 어시스턴트', page_icon='🛠️', layout='wide')

import matplotlib.pyplot as plt

import config
import train_models as tm
from anomaly_detector import PumpAnomalyDetector
from knowledge_base import MaintenanceKnowledgeBase
from llm_advisor import LLMClient, MaintenanceAdvisor
from report_builder import ReportBuilder
from ui_common import (OVERALL, contribution_plot, friendly_error, representative_row, sources_md, status_md,
                       timeline_plot, votes_df)


# ---------- 객체 준비: 모든 사용자가 한 번만 만든 것을 같이 쓴다 ----------
@st.cache_resource(show_spinner='딥러닝 모델을 불러오는 중...')
def load_detector():
    return PumpAnomalyDetector()


@st.cache_resource(show_spinner='정비 문서 검색 인덱스를 준비하는 중...')
def load_kb():
    kb = MaintenanceKnowledgeBase()
    kb.warm_up()   # 벡터 검색이 실패해도 BM25로 동작한다
    return kb


@st.cache_resource
def load_advisor():
    return MaintenanceAdvisor(load_kb(), LLMClient())


detector, kb, advisor, reporter = load_detector(), load_kb(), load_advisor(), ReportBuilder()
LLM_OK, LLM_MSG = advisor.llm.status()
EXPERIMENTS = detector.experiments                  # {'valve1/0': '입구 밸브 닫힘', ...}
EXP_LABELS = {f'{fault} ({name})': name for name, fault in EXPERIMENTS.items()}
CASE_NAMES = list(detector.cases)

ss = st.session_state
for key, default in [('result', None), ('advice', None), ('advice_error', ''), ('messages', [])]:
    ss.setdefault(key, default)


# ---------- 입력 (사이드바) ----------
def set_row(kind):
    ss.row = representative_row(detector, EXP_LABELS[ss.exp], kind)


def on_exp_change():
    set_row('fault')


with st.sidebar:
    st.header('입력')
    mode = st.radio('입력 방식', ['실험 데이터 재생', '센서값 직접 조절'], horizontal=True)
    if mode == '실험 데이터 재생':
        first = list(EXP_LABELS)[0]
        ss.setdefault('exp', first)
        ss.setdefault('row', representative_row(detector, EXP_LABELS[first], 'fault'))
        st.selectbox('실험 (SKAB 펌프 테스트베드)', list(EXP_LABELS), key='exp', on_change=on_exp_change)
        n = len(detector.frame(EXP_LABELS[ss.exp]))
        ss.row = min(max(ss.row, tm.WINDOW - 1), n - 1)
        st.slider('시점 (최근 10초로 판정)', tm.WINDOW - 1, n - 1, key='row')
        c1, c2 = st.columns(2)
        c1.button('정상 구간 예시', on_click=set_row, args=('normal',), width='stretch')
        c2.button('고장 구간 예시', on_click=set_row, args=('fault',), width='stretch')
    else:
        case = st.selectbox('조건 (대표 구간의 10초 변동 유지)', CASE_NAMES)
        low, high = detector.slider_range(case)
        init = detector.preset_values(case, 'normal' if case == tm.NORMAL else 'fault')
        values = []
        for i, s in enumerate(tm.SENSORS):
            lo, hi = float(low[i]), float(high[i])
            # 기본 간격(0.01)으로는 진동처럼 변동이 아주 작은 센서가 반올림되어 이상으로 바뀐다 → 범위의 1/1000 간격
            step = 10 ** math.floor(math.log10((hi - lo) / 1000)) if hi > lo else 0.001
            digits = max(0, -int(math.log10(step)))
            values.append(st.slider(f'{config.SENSOR_KO[s]} ({config.SENSOR_UNIT[s]})', lo, hi,
                                    round(float(min(max(init[i], lo), hi)), digits), step=step,
                                    format=f'%.{digits}f', key=f'v_{case}_{i}'))
    st.divider()
    use_rag = st.checkbox('RAG 사용 (정비 문서 검색)', value=True)
    rag_mode = st.selectbox('검색 방식', ['hybrid', 'vector', 'bm25'],
                            help='벡터 검색을 쓸 수 없는 환경에서는 자동으로 bm25로 검색합니다.')
    run = st.button('판정 + 정비 안내', type='primary', width='stretch')
    st.caption(f"LLM: {LLM_MSG}  \n검색: 문서 청크 {len(kb.chunks)}개 · 임베딩 {kb.embedding_name}")
    if kb.vector_error:
        st.caption(kb.vector_error)

# ---------- 화면 ----------
st.title('펌프 이상탐지 + LLM 정비 어시스턴트')
st.markdown('판정 **LSTM 오토인코더 (DL)** → 근거 **정비 문서 검색 (RAG)** → 안내 **LLM**  \n'
            '정상 운전만 학습한 딥러닝 모델이 이상을 판정하고, LLM은 고장을 직접 판단하지 않고 근거 문서로 점검 안내만 만듭니다.')
if not LLM_OK:
    st.warning(f'정비 안내(LLM)를 사용할 수 없습니다. {LLM_MSG} 딥러닝 판정과 문서 검색은 그대로 동작합니다.')

tab1, tab2, tab3 = st.tabs(['이상 판정 + 정비 안내', '후속 질문 챗봇', '모델 성능 (DL)'])

with tab1:
    if run:
        if mode == '실험 데이터 재생':
            ss.result = detector.detect_at(EXP_LABELS[ss.exp], int(ss.row))
        else:
            ss.result = detector.detect_from_values(case, values)
        ss.advice, ss.advice_error = None, ''

    result = ss.result
    left, right = st.columns([5, 5])
    with left:
        if mode == '실험 데이터 재생':
            fig = timeline_plot(detector, EXP_LABELS[ss.exp], ss.row)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            st.caption('파란선: 이상점수(로그 축) · 점선: 임계값 · 빨간 영역: 실제 이상 구간 · 빨간 세로선: 판정 시점')
        if result is not None:
            st.markdown(status_md(result, heading='###'))
    with right:
        if result is not None:
            fig = contribution_plot(result)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
            st.dataframe(votes_df(result), hide_index=True, width='stretch')

    if result is None:
        st.info('왼쪽에서 실험과 시점을 고른 뒤 **판정 + 정비 안내**를 누르세요.')
    else:
        if run and LLM_OK:
            with st.spinner('정비 문서를 검색하고 안내를 작성하는 중입니다... (20~60초)'):
                try:
                    ss.advice = advisor.advise(result, use_rag=use_rag, mode=rag_mode)
                except Exception as e:   # API 한도, 키 오류, 연결 오류: 판정 화면은 유지
                    ss.advice_error = friendly_error(e)
        st.divider()
        a_col, s_col = st.columns([6, 4])
        with a_col:
            st.subheader('AI 정비 안내')
            if ss.advice is not None:
                st.markdown(ss.advice.text)
                st.caption(f'{ss.advice.model} · {ss.advice.seconds:.1f}초')
            elif ss.advice_error:
                st.error(ss.advice_error)
            elif not LLM_OK:
                st.write('LLM 설정 후 사용할 수 있습니다.')
        with s_col:
            st.subheader('근거 문서')
            if ss.advice is not None:
                st.markdown(sources_md(ss.advice))
                md, pdf = reporter.build(result, ss.advice)
                st.download_button('정비 리포트 (Markdown)', md.encode('utf-8'), 'maintenance_report.md',
                                   'text/markdown', width='stretch')
                if pdf:
                    st.download_button('정비 리포트 (PDF)', pdf, 'maintenance_report.pdf', 'application/pdf',
                                       width='stretch')

with tab2:
    st.write('현재 판정 결과를 바탕으로 질문하세요. 답변은 정비 문서 검색 결과를 근거로 합니다.')
    if ss.result is None:
        st.caption('판정을 먼저 실행하면 그 결과를 함께 참고합니다.')
    examples = ['진동이 높으면 무엇부터 점검해야 해?', '고장 조치 후에도 온도 때문에 경보가 계속 나면?',
                '점검 전에 지켜야 할 안전수칙은?']
    cols = st.columns(len(examples))
    clicked = next((q for c, q in zip(cols, examples) if c.button(q, width='stretch')), None)
    for m in ss.messages:
        with st.chat_message(m['role']):
            st.markdown(m['content'])
    question = st.chat_input('질문을 입력하세요', disabled=not LLM_OK) or clicked
    if question:
        ss.messages.append({'role': 'user', 'content': question})
        with st.chat_message('user'):
            st.markdown(question)
        with st.chat_message('assistant'):
            if not LLM_OK:
                reply = f'LLM을 사용할 수 없습니다: {LLM_MSG}'
            else:
                with st.spinner('답변을 작성하는 중...'):
                    try:
                        adv = advisor.answer(question, ss.result)
                        cites = ', '.join(f'[{i}] {c.section.split(" > ")[-1]}' for i, c in enumerate(adv.sources, 1))
                        reply = f'{adv.text}\n\n<sub>참고: {cites}</sub>'
                    except Exception as e:
                        reply = friendly_error(e)
            st.markdown(reply, unsafe_allow_html=True)
        ss.messages.append({'role': 'assistant', 'content': reply})

with tab3:
    st.subheader('33개 실험 공식 성능 (시드 42)')
    st.write('평가 구간의 54.2%가 이상이라, 항상 경보를 울려도 F1 0.703입니다. F1은 이 기준선과 함께 읽어야 합니다.')
    st.dataframe(OVERALL, hide_index=True, width='stretch')
