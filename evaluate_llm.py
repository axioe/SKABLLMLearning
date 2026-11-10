"""LLM 비교 실험: 같은 이상탐지 결과에 대해 (LLM 모델 × RAG 사용 여부)별 안내 품질을 자동 채점한다.

채점 기준 (사람이 읽지 않아도 계산되는 지표)
    - 기여센서_언급: 가장 크게 기여한 센서를 답변에서 언급했는가 (1/0)
    - 점검항목수: 체크리스트(- [ ]) 항목 개수 (RAG 없이는 '센서 값 확인' 수준에 그치는지 비교)
    - 근거_인용수: [1], [2] 같은 문서 번호 인용 개수 (RAG를 켰을 때 의미 있음)
    - 지어낸_수치: 답변의 숫자 중 입력(판정 결과 + 참고 문서)에 없는 숫자 개수 → 환각 지표, 낮을수록 좋음
    - 단정_표현: '원인입니다', '확실합니다'처럼 원인을 확정하는 표현 개수, 낮을수록 좋음
    - 응답시간(초)

실행: .venv\\Scripts\\python.exe evaluate_llm.py            (기본: Ollama gemma2, gemma4:e2b)
      .venv\\Scripts\\python.exe evaluate_llm.py --gemini   (GEMINI_API_KEY가 있으면 Gemini도 비교)
"""
import argparse
import json
import re

import pandas as pd

import config
from anomaly_detector import PumpAnomalyDetector
from knowledge_base import MaintenanceKnowledgeBase
from llm_advisor import ADVICE_TEMPLATE, NORMAL_TEMPLATE, LLMClient, MaintenanceAdvisor

ASSERTIVE = ['원인입니다', '원인이다', '확실합니다', '확실히', '틀림없', '분명히 고장']


def numbers(text: str) -> set:
    return {n.rstrip('.').lstrip('0') or '0' for n in re.findall(r'\d+(?:\.\d+)?', text)}


class AdvisorEvaluator:
    def __init__(self, detector: PumpAnomalyDetector, kb: MaintenanceKnowledgeBase):
        self.detector = detector
        self.kb = kb

    def cases(self) -> list:
        """고장 4종의 고장 구간 대표 시점 + 정상 구간 1개."""
        out = []
        for name, fault in self.detector.experiments.items():
            tl = self.detector.score_timeline(name)
            out.append((fault, self.detector.detect_at(name, int(tl.loc[tl['anomaly'] == 1, 'row'].median()))))
        tl = self.detector.score_timeline('other/7')
        start = int(tl['anomaly'].values.argmax())
        normal_rows = tl.loc[(tl['row'] >= 400) & (tl['row'] < start), 'row']
        out.append(('정상 운전', self.detector.detect_at('other/7', int(normal_rows.median()))))
        return out

    @staticmethod
    def score(result, advice) -> dict:
        text = advice.text
        top = result.contributions[0].name_ko
        key = '진동' if top.startswith('진동') else top
        template = ADVICE_TEMPLATE if result.is_anomaly else NORMAL_TEMPLATE
        given = json.dumps(result.to_llm_dict(), ensure_ascii=False) + template + \
            ' '.join(c.text for c in advice.sources)
        invented = numbers(text) - numbers(given) - {str(i) for i in range(1, 11)}   # 목록 번호 1~10 제외
        return {
            '기여센서_언급': int(key in text),
            '점검항목수': text.count('- [ ]'),
            '근거_인용수': len(re.findall(r'\[\d\]', text)),
            '지어낸_수치': len(invented),
            '단정_표현': sum(text.count(w) for w in ASSERTIVE),
            '응답시간(초)': round(advice.seconds, 1),
        }

    def run(self, llms: list, rag_options=(True, False)) -> pd.DataFrame:
        rows, logs = [], []
        cases = self.cases()
        for llm in llms:
            advisor = MaintenanceAdvisor(self.kb, llm)
            for use_rag in rag_options:
                for fault, result in cases:
                    try:
                        advice = advisor.advise(result, use_rag=use_rag)
                    except Exception as e:   # 메모리 부족 등: 실패로 기록하고 다음 모델로 넘어간다
                        print(f'{llm.name} 실패: {e}')
                        rows.append({'모델': llm.name, 'RAG': '사용' if use_rag else '미사용', '케이스': fault,
                                     '오류': str(e)[:120]})
                        break
                    row = {'모델': llm.name, 'RAG': '사용' if use_rag else '미사용', '케이스': fault,
                           **self.score(result, advice)}
                    rows.append(row)
                    logs.append({**row, '답변': advice.text, '출처': [c.cite() for c in advice.sources]})
                    print(row)
        config.OUTPUT_DIR.mkdir(exist_ok=True)
        with open(config.OUTPUT_DIR / 'llm_eval_responses.jsonl', 'w', encoding='utf-8') as f:
            for log in logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')
        df = pd.DataFrame(rows)
        df.to_csv(config.OUTPUT_DIR / 'llm_eval.csv', index=False, encoding='utf-8-sig')
        return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gemini', action='store_true')
    parser.add_argument('--models', default='gemma2,gemma4:e2b')
    args = parser.parse_args()

    llms = [LLMClient('ollama', m) for m in args.models.split(',')]
    if args.gemini:
        llms.append(LLMClient('gemini'))
    df = AdvisorEvaluator(PumpAnomalyDetector(), MaintenanceKnowledgeBase()).run(llms)
    metrics = ['기여센서_언급', '점검항목수', '근거_인용수', '지어낸_수치', '단정_표현', '응답시간(초)']
    summary = df.groupby(['모델', 'RAG'])[[m for m in metrics if m in df]].mean()
    print('\n', summary.round(2))
    summary.round(2).to_csv(config.OUTPUT_DIR / 'llm_eval_summary.csv', encoding='utf-8-sig')
