"""화면 공통 요소: 그래프, 판정 요약, 표. Gradio(app.py)와 Streamlit(streamlit_app.py)이 같이 쓴다."""
import logging

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

import train_models as tm

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)   # 글꼴 후보 경고 숨김
# Windows는 맑은 고딕, Streamlit Cloud(Linux)는 packages.txt의 fonts-nanum으로 설치한 나눔고딕
plt.rcParams['font.family'] = ['Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
BLUE, RED, GRAY = '#2b4fa3', '#c0392b', '#c5cbd3'

# 딥러닝 프로젝트의 33개 실험 공식 성능 (시드 42)
OVERALL = pd.DataFrame([
    ['LSTM AE (대표)', 0.785, 40.42, 13.29, 0.815, '3,848'],
    ['Conv1D AE', 0.779, 40.43, 14.28, 0.845, '1,584'],
    ['Conv2D AE', 0.793, 32.54, 16.23, 0.834, '899'],
    ['Dense AE', 0.755, 35.38, 21.26, 0.782, '428'],
    ['Isolation Forest', 0.533, 13.68, 59.44, 0.745, '신경망 아님'],
    ['평균 이탈(10초)', 0.766, 39.58, 17.05, 0.814, '학습 없음'],
], columns=['모델', 'F1', 'FAR(%)', 'MAR(%)', '평균 AUC', '파라미터 수'])


def timeline_plot(detector, name, row):
    tl = detector.score_timeline(name)
    thr = detector.thresholds(name)[tm.REPRESENTATIVE]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.fill_between(tl['row'], 0, 1, where=tl['anomaly'] == 1, transform=ax.get_xaxis_transform(),
                    color=RED, alpha=0.08, label='실제 이상 구간')
    ax.axvspan(0, tm.TRAIN_SIZE, color=GRAY, alpha=0.25, label='학습 구간(정상)')
    ax.plot(tl['row'], tl['score'], color=BLUE, lw=1, label='이상점수 (LSTM AE)')
    ax.axhline(thr, color='black', ls='--', lw=1, label=f'임계값 {thr:.2f}')
    ax.axvline(row, color=RED, lw=1.5)
    ax.set_yscale('log')
    ax.set_xlabel('시점 (초)')
    ax.legend(fontsize=8, loc='upper left', frameon=False, ncol=2)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def contribution_plot(result):
    contribs = result.contributions[::-1]
    color = RED if result.is_anomaly else BLUE
    top = result.contributions[0].sensor
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.barh([c.name_ko for c in contribs], [c.share for c in contribs],
            color=[color if c.sensor == top else GRAY for c in contribs])
    ax.set_xlabel('복원오차 비중 (%)')
    ax.set_title('어떤 센서가 평소와 다른가? (LSTM AE)', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def status_md(result, heading='##'):
    icon = '🔴 이상' if result.is_anomaly else '🟢 정상'
    top = result.contributions[0]
    alarms = sum(a for _, _, a in result.model_votes.values())
    truth = ''
    if result.true_label is not None:
        truth = f"\n\n실제 라벨: **{'이상' if result.true_label else '정상'}** (평가용, LLM에는 전달하지 않음)"
    return (f'{heading} {icon} · 긴급도 {result.severity}\n'
            f'LSTM AE 점수 **{result.score:.3f}** / 임계값 {result.threshold:.3f} (×{result.ratio:.2f}) · '
            f'6개 모델 중 {alarms}개 이상 판정 · 가장 크게 벗어난 센서: **{top.name_ko}** ({top.direction}){truth}')


def votes_df(result):
    return pd.DataFrame([{'모델': m, '판정': '이상' if a else '정상', '점수': round(s, 3), '임계값': round(t, 3)}
                         for m, (s, t, a) in result.model_votes.items()])


def sources_md(advice):
    if not advice.sources:
        return '_RAG 사용 안 함: 검색 문서 없이 LLM만으로 답변했습니다._'
    lines = [f'**검색 질의:** {advice.query}', '']
    lines += [f'{i}. {c.cite()} (점수 {c.score:.2f})' for i, c in enumerate(advice.sources, 1)]
    return '\n'.join(lines)


def representative_row(detector, name, kind):
    """정상/고장 구간의 대표 시점(중앙값)."""
    tl = detector.score_timeline(name)
    if kind == 'fault':
        rows = tl.loc[tl['anomaly'] == 1, 'row']
    else:
        start = int(tl['anomaly'].values.argmax())
        rows = tl.loc[(tl['row'] >= tm.TRAIN_SIZE) & (tl['row'] < start), 'row']
    return int(rows.median())


def friendly_error(e: Exception) -> str:
    """LLM 호출 오류를 사용자가 이해할 수 있는 문장으로 바꾼다."""
    msg = str(e)
    if '429' in msg or 'RESOURCE_EXHAUSTED' in msg or 'quota' in msg.lower():
        return 'Gemini API 사용량 한도를 넘었습니다. 1분쯤 뒤에 다시 시도해 주세요.'
    if 'API key' in msg or 'API_KEY' in msg or '401' in msg or '403' in msg:
        return 'Gemini API 키가 올바르지 않습니다. Streamlit Cloud의 Secrets에 있는 GEMINI_API_KEY를 확인해 주세요.'
    if 'Connection' in msg or 'connect' in msg.lower():
        return 'LLM 서버에 연결하지 못했습니다. 로컬은 Ollama 실행 여부, 배포 서버는 GEMINI_API_KEY 설정을 확인해 주세요.'
    return f'정비 안내를 만들지 못했습니다: {msg[:200]}'
