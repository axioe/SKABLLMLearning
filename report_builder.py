"""정비 리포트 생성: 이상탐지 결과 + LLM 안내를 Markdown과 PDF 파일로 저장한다."""
import re
from datetime import datetime
from pathlib import Path

import config
from anomaly_detector import DetectionResult
from llm_advisor import Advice

# 한글 PDF 글꼴 후보 (Windows → Linux/Colab 순서)
FONT_CANDIDATES = [
    Path('C:/Windows/Fonts/malgun.ttf'),
    Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
]


class ReportBuilder:
    def __init__(self, output_dir: Path = config.OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.font = next((p for p in FONT_CANDIDATES if p.exists()), None)

    def to_markdown(self, result: DetectionResult, advice: Advice) -> str:
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        rows = '\n'.join(
            f'| {c.name_ko} | {c.share:.1f}% | {c.current:.3f} {c.unit} | {c.normal_mean:.3f} {c.unit} | {c.z:+.1f} |'
            for c in result.contributions)
        votes = '\n'.join(f'| {m} | {"이상" if a else "정상"} | {s:.3f} | {t:.3f} |'
                          for m, (s, t, a) in result.model_votes.items())
        sources = '\n'.join(f'{i}. {c.cite()}' for i, c in enumerate(advice.sources, 1)) or '(검색 문서 없음)'
        return f"""# 펌프 설비 정비 리포트

- 작성 시각: {now}
- 판정: **{'이상' if result.is_anomaly else '정상'}** (긴급도 {result.severity})
- LSTM AE 이상점수 {result.score:.3f} / 임계값 {result.threshold:.3f} (비율 {result.ratio:.2f})
- 안내 생성 모델: {advice.model} ({advice.seconds:.1f}초)

## 센서별 복원오차 기여도

| 센서 | 오차 비중 | 현재값 | 정상 평균 | 표준편차 배수 |
| --- | --- | --- | --- | --- |
{rows}

## 모델별 판정

| 모델 | 판정 | 점수 | 임계값 |
| --- | --- | --- | --- |
{votes}

## AI 정비 안내

{advice.text.strip()}

## 참고 문서

{sources}

---
이 리포트는 점검 우선순위를 정하는 참고 자료이며, 고장 원인을 확정하지 않습니다.
"""

    def save(self, result: DetectionResult, advice: Advice) -> dict:
        """Markdown과 PDF를 저장하고 경로를 돌려준다. 한글 글꼴이 없으면 Markdown만 저장한다."""
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        md = self.to_markdown(result, advice)
        md_path = self.output_dir / f'maintenance_report_{stamp}.md'
        md_path.write_text(md, encoding='utf-8')
        paths = {'markdown': md_path}
        if self.font:
            paths['pdf'] = self._to_pdf(md, self.output_dir / f'maintenance_report_{stamp}.pdf')
        return paths

    def build(self, result: DetectionResult, advice: Advice) -> tuple:
        """파일로 저장하지 않고 (Markdown 문자열, PDF 바이트 또는 None)을 돌려준다 (Streamlit 다운로드 버튼용)."""
        md = self.to_markdown(result, advice)
        return md, (bytes(self._pdf(md).output()) if self.font else None)

    def _to_pdf(self, md: str, path: Path) -> Path:
        self._pdf(md).output(str(path))
        return path

    def _pdf(self, md: str):
        """Markdown을 단순 변환한다: 제목, 표(텍스트 행), 목록, 본문."""
        from fpdf import FPDF

        pdf = FPDF(format='A4')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.add_font('KR', '', str(self.font))
        width = pdf.w - pdf.l_margin - pdf.r_margin
        for line in md.splitlines():
            text = re.sub(r'\*\*(.+?)\*\*', r'\1', line).replace('- [ ] ', '□ ')
            if re.match(r'^\|\s*-', text):
                continue                                   # 표 구분선
            if text.startswith('# '):
                pdf.set_font('KR', size=16)
                pdf.multi_cell(width, 9, text[2:])
                pdf.ln(2)
            elif text.startswith('## ') or text.startswith('### '):
                pdf.ln(2)
                pdf.set_font('KR', size=12.5)
                pdf.multi_cell(width, 7, text.lstrip('#').strip())
            elif text.startswith('|'):
                pdf.set_font('KR', size=9)
                cells = [c.strip() for c in text.strip('|').split('|')]
                col = width / len(cells)
                for c in cells:
                    pdf.cell(col, 6, c[:40], border=1)
                pdf.ln()
            elif text.strip() == '---':
                pdf.ln(3)
            else:
                pdf.set_font('KR', size=10)
                pdf.multi_cell(width, 6, text if text.strip() else ' ')
        return pdf
