"""펌프 이상탐지 + LLM 정비 어시스턴트 (Gradio 시연 화면)

실행:  .venv\\Scripts\\python.exe app.py
외부 공유: share=True로 실행하면 gradio.live 공유 링크가 생긴다 (발표 PC에서 Ollama가 켜져 있어야 함).
"""
import os
import sys
import warnings

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
warnings.filterwarnings('ignore')
for stream in (sys.stdout, sys.stderr):    # Windows 콘솔(cp949)에서 한글 출력 오류 방지
    if hasattr(stream, 'reconfigure'):
        stream.reconfigure(encoding='utf-8', errors='replace')

import gradio as gr

import config
import train_models as tm
from anomaly_detector import PumpAnomalyDetector
from knowledge_base import MaintenanceKnowledgeBase
from llm_advisor import LLMClient, MaintenanceAdvisor
from report_builder import ReportBuilder
from ui_common import (OVERALL, contribution_plot, friendly_error, representative_row, sources_md, status_md,
                       timeline_plot as _timeline, votes_df)

# ---------- 객체 준비 (앱 시작 시 한 번) ----------
detector = PumpAnomalyDetector()
kb = MaintenanceKnowledgeBase()
kb.warm_up()
advisor = MaintenanceAdvisor(kb, LLMClient())
reporter = ReportBuilder()
LLM_OK, LLM_MSG = advisor.llm.status()

EXP_LABELS = {f'{fault} ({name})': name for name, fault in detector.experiments.items()}
CASE_NAMES = list(detector.cases)


def timeline_plot(name, row):
    return _timeline(detector, name, row)


# ---------- 이벤트 ----------
def run(mode, exp_label, row, case, use_rag, rag_mode, *values):
    """판정은 바로 보여주고, LLM 안내는 생성되는 대로 이어서 보여준다."""
    if mode == '실험 데이터 재생':
        name = EXP_LABELS[exp_label]
        result = detector.detect_at(name, int(row))
        tl_fig = timeline_plot(name, result.row)
    else:
        result = detector.detect_from_values(case, values)
        tl_fig = None
    base = [result, status_md(result), contribution_plot(result), votes_df(result), tl_fig]
    yield base + ['_정비 안내를 생성하는 중입니다..._', '', None]

    if not LLM_OK:
        yield base + [f'LLM을 사용할 수 없습니다: {LLM_MSG}', '', None]
        return
    try:
        advice = advisor.advise(result, use_rag=use_rag, mode=rag_mode)
    except Exception as e:   # 메모리 부족, 연결 오류 등: 판정 결과는 그대로 두고 안내만 실패 표시
        yield base + [friendly_error(e), '', None]
        return
    footer = f'\n\n<sub>{advice.model} · {advice.seconds:.1f}초</sub>'
    yield base + [advice.text + footer, sources_md(advice), advice]


def save_report(result, advice):
    if result is None or advice is None:
        gr.Warning('먼저 판정과 정비 안내를 실행해 주세요.')
        return None
    paths = reporter.save(result, advice)
    return [str(p) for p in paths.values()]


def jump(exp_label, kind):
    """정상/고장 구간의 대표 시점으로 슬라이더를 옮긴다."""
    return representative_row(detector, EXP_LABELS[exp_label], kind)


def on_exp_change(exp_label):
    name = EXP_LABELS[exp_label]
    n = len(detector.frame(name))
    row = jump(exp_label, 'fault')
    return gr.update(maximum=n - 1, value=row), timeline_plot(name, row)


def on_case_change(case):
    low, high = detector.slider_range(case)
    vals = detector.preset_values(case, 'normal' if case == tm.NORMAL else 'fault')
    return [gr.update(minimum=float(low[i]), maximum=float(high[i]), value=round(float(vals[i]), 3))
            for i in range(len(tm.SENSORS))]


def chat(message, history, result):
    if not LLM_OK:
        return f'LLM을 사용할 수 없습니다: {LLM_MSG}'
    try:
        advice = advisor.answer(message, result)
    except Exception as e:
        return friendly_error(e)
    cites = ', '.join(f'[{i}] {c.section.split(" > ")[-1]}' for i, c in enumerate(advice.sources, 1))
    return f'{advice.text}\n\n<sub>참고: {cites}</sub>'


# ---------- 화면 ----------
with gr.Blocks(title='펌프 이상탐지 + LLM 정비 어시스턴트') as demo:
    result_state = gr.State(None)
    advice_state = gr.State(None)

    gr.Markdown('# 펌프 이상탐지 + LLM 정비 어시스턴트\n'
                '정상 운전만 학습한 <b>LSTM 오토인코더(DL)</b>가 이상을 판정하고, '
                '<b>정비 문서 검색(RAG)</b>과 <b>LLM</b>이 점검 안내를 만듭니다. '
                f'\n\n<sub>LLM 상태: {LLM_MSG} · 지식 문서 청크 {len(kb.chunks)}개</sub>')

    with gr.Tab('이상 판정 + 정비 안내'):
        with gr.Row():
            with gr.Column(scale=4):
                mode = gr.Radio(['실험 데이터 재생', '센서값 직접 조절'], value='실험 데이터 재생', label='입력 방식')
                with gr.Group(visible=True) as replay_box:
                    first = list(EXP_LABELS)[0]
                    exp_dd = gr.Dropdown(list(EXP_LABELS), value=first, label='실험 (SKAB 펌프 테스트베드)')
                    n0 = len(detector.frame(EXP_LABELS[first]))
                    row0 = jump(first, 'fault')
                    row_sl = gr.Slider(tm.WINDOW - 1, n0 - 1, value=row0, step=1, label='시점 (최근 10초로 판정)')
                    with gr.Row():
                        normal_btn = gr.Button('정상 구간 예시', size='sm')
                        fault_btn = gr.Button('고장 구간 예시', size='sm')
                with gr.Group(visible=False) as manual_box:
                    case_dd = gr.Dropdown(CASE_NAMES, value=CASE_NAMES[0], label='조건 (대표 구간의 10초 변동 유지)')
                    sliders = []
                    low, high = detector.slider_range(CASE_NAMES[0])
                    init = detector.preset_values(CASE_NAMES[0], 'normal')
                    for i, s in enumerate(tm.SENSORS):
                        sliders.append(gr.Slider(float(low[i]), float(high[i]), value=round(float(init[i]), 3),
                                                 label=f'{config.SENSOR_KO[s]} ({config.SENSOR_UNIT[s]})'))
                with gr.Row():
                    rag_cb = gr.Checkbox(value=True, label='RAG 사용 (정비 문서 검색)')
                    rag_mode = gr.Dropdown(['hybrid', 'vector', 'bm25'], value='hybrid', label='검색 방식')
                run_btn = gr.Button('판정 + 정비 안내', variant='primary')
            with gr.Column(scale=6):
                status = gr.Markdown()
                tl_plot = gr.Plot(label='시점별 이상점수', value=timeline_plot(EXP_LABELS[first], row0))
                contrib_plot = gr.Plot(label='센서별 기여도')
                votes = gr.Dataframe(label='모델별 판정 (비교 모델 5종 포함)', interactive=False)
        with gr.Row():
            with gr.Column(scale=6):
                advice_md = gr.Markdown(label='AI 정비 안내')
            with gr.Column(scale=4):
                gr.Markdown('#### 근거 문서')
                src_md = gr.Markdown()
                report_btn = gr.Button('정비 리포트 저장 (Markdown + PDF)')
                report_files = gr.File(label='리포트 파일', file_count='multiple')

    with gr.Tab('후속 질문 챗봇'):
        gr.Markdown('현재 판정 결과를 바탕으로 질문하세요. 답변은 정비 문서 검색 결과를 근거로 합니다.')
        gr.ChatInterface(chat, additional_inputs=[result_state],
                         examples=[['진동이 높으면 무엇부터 점검해야 해?', None],
                                   ['고장 조치 후에도 온도 때문에 경보가 계속 나면?', None],
                                   ['점검 전에 지켜야 할 안전수칙은?', None]])

    with gr.Tab('모델 성능 (DL)'):
        gr.Markdown('### 33개 실험 공식 성능 (시드 42)\n평가 구간의 54.2%가 이상이라, 항상 경보를 울려도 F1 0.703입니다. '
                    'F1은 이 기준선과 함께 읽어야 합니다.')
        gr.Dataframe(OVERALL, interactive=False)

    # 이벤트 연결
    mode.change(lambda m: (gr.update(visible=m == '실험 데이터 재생'), gr.update(visible=m != '실험 데이터 재생')),
                mode, [replay_box, manual_box])
    exp_dd.change(on_exp_change, exp_dd, [row_sl, tl_plot])
    row_sl.release(lambda e, r: timeline_plot(EXP_LABELS[e], int(r)), [exp_dd, row_sl], tl_plot)
    normal_btn.click(lambda e: jump(e, 'normal'), exp_dd, row_sl).then(
        lambda e, r: timeline_plot(EXP_LABELS[e], int(r)), [exp_dd, row_sl], tl_plot)
    fault_btn.click(lambda e: jump(e, 'fault'), exp_dd, row_sl).then(
        lambda e, r: timeline_plot(EXP_LABELS[e], int(r)), [exp_dd, row_sl], tl_plot)
    case_dd.change(on_case_change, case_dd, sliders)
    run_btn.click(run, [mode, exp_dd, row_sl, case_dd, rag_cb, rag_mode, *sliders],
                  [result_state, status, contrib_plot, votes, tl_plot, advice_md, src_md, advice_state])
    report_btn.click(save_report, [result_state, advice_state], report_files)


if __name__ == '__main__':
    share = os.getenv('GRADIO_SHARE', '0') == '1'
    demo.queue().launch(server_name='0.0.0.0', server_port=int(os.getenv('PORT', '7861')), share=share)
