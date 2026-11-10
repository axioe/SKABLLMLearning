"""[DL] 펌프 이상탐지: 딥러닝 프로젝트에서 학습한 LSTM 오토인코더를 감싸는 클래스.

재학습 없이 models/ 폴더의 모델을 불러와
    1) 이상 여부 판정 (복원오차 > 임계값)
    2) 센서별 복원오차 기여도
    3) 비교 모델 5종의 판정
을 계산하고, LLM에 넘길 수 있는 구조화된 결과(dict)로 바꾼다.

두 가지 입력 방식을 지원한다.
    - detect_at(): 실험 CSV의 특정 시점(최근 10초)을 그대로 재생 (실제 데이터)
    - detect_from_values(): 기존 Streamlit 앱처럼 센서 평균값을 슬라이더로 바꾼 모의 입력
"""
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

import config
import train_models as tm   # 딥러닝 프로젝트 원본 코드 (모델 구조, 점수 계산)


@dataclass
class SensorContribution:
    """센서 하나의 복원오차와 현재 상태."""
    sensor: str
    name_ko: str
    unit: str
    error: float          # 10초 평균 제곱 복원오차 (표준화 단위)
    share: float          # 전체 오차에서 차지하는 비율 (%)
    current: float        # 현재(마지막 1초) 실제 측정값
    normal_mean: float    # 학습 구간(정상 운전) 평균
    z: float              # 정상 평균에서 표준편차 몇 배만큼 벗어났는지

    @property
    def direction(self) -> str:
        return '정상보다 높음' if self.z > 0 else '정상보다 낮음'


@dataclass
class DetectionResult:
    """이상탐지 결과. to_llm_dict()로 LLM 프롬프트에 넣는다."""
    experiment: str                   # 실험 이름 (예: valve1/0)
    is_anomaly: bool
    score: float
    threshold: float
    severity: str                     # 정상 / 주의 / 높음
    contributions: list               # SensorContribution, 오차 큰 순
    model_votes: dict                 # 모델명 -> (점수, 임계값, 이상 여부)
    window: np.ndarray = field(repr=False)   # 표준화된 입력 (10, 8)
    recon: np.ndarray = field(repr=False)    # LSTM AE 복원값 (10, 8)
    row: int | None = None            # 재생 모드의 시점 (행 번호)
    true_label: int | None = None     # 재생 모드의 실제 라벨 (평가·화면 표시용, LLM에는 넘기지 않음)

    @property
    def ratio(self) -> float:
        return self.score / self.threshold

    @property
    def top_sensors(self) -> list:
        return self.contributions[:3]

    def to_llm_dict(self) -> dict:
        """LLM에 넘길 정보만 담는다. 실험 이름·고장 종류·정답 라벨은 넣지 않는다 (답을 미리 알려주지 않기 위해)."""
        alarms = [m for m, (_, _, a) in self.model_votes.items() if a]
        return {
            '판정': '이상' if self.is_anomaly else '정상',
            '긴급도': self.severity,
            '이상점수': round(self.score, 3),
            '임계값': round(self.threshold, 3),
            '점수_임계값_비율': round(self.ratio, 2),
            '이상으로_판정한_모델수': f'{len(alarms)}/{len(self.model_votes)}',
            '기여_센서_상위3': [
                {'센서': c.name_ko, '오차_비중(%)': round(c.share, 1),
                 '현재값': f'{c.current:.3f} {c.unit}', '정상평균': f'{c.normal_mean:.3f} {c.unit}',
                 '표준편차_배수': round(c.z, 1), '방향': c.direction}
                for c in self.top_sensors],
        }


class PumpAnomalyDetector:
    """학습된 모델을 불러와 펌프 센서 10초 구간의 이상 여부를 판정한다."""

    def __init__(self):
        self.demo = joblib.load(tm.DEMO_DATA)   # 임계값, 슬라이더 범위, 기준 구간 (딥러닝 프로젝트에서 저장)
        self._bundles = {}
        self._frames = {}
        self._timelines = {}

    # ---------- 조회 ----------
    @property
    def cases(self) -> dict:
        """화면에 보여줄 조건명 -> 실험 이름 (예: '입구 밸브 닫힘' -> 'valve1/0')."""
        return dict(tm.CASES)

    @property
    def experiments(self) -> dict:
        """재생 가능한 실험 -> 고장 종류."""
        return dict(tm.DEMO)

    def thresholds(self, name: str) -> dict:
        return self.demo['experiments'][name]['thresholds']

    def bundle(self, name: str) -> dict:
        """실험 하나의 스케일러와 모델 5종 (처음 한 번만 불러온다)."""
        if name not in self._bundles:
            self._bundles[name] = {**tm.load_models(name), 'thresholds': self.thresholds(name)}
        return self._bundles[name]

    def frame(self, name: str) -> pd.DataFrame:
        """실험 CSV (센서 8개 + anomaly 라벨)."""
        if name not in self._frames:
            self._frames[name] = tm.read_experiment(name)
        return self._frames[name]

    def preset_values(self, case: str, kind: str = 'fault') -> np.ndarray:
        """조건별 대표 구간(정상 'normal' / 고장 'fault')의 센서 평균값. 슬라이더 초기값으로 쓴다."""
        return self.demo['experiments'][self.cases[case]][kind].mean(axis=0)

    def slider_range(self, case: str):
        exp = self.demo['experiments'][self.cases[case]]
        return exp['low'], exp['high']

    # ---------- 판정 ----------
    def detect_at(self, name: str, row: int) -> DetectionResult:
        """재생 모드: 실험 CSV의 row 시점까지 최근 10초로 판정한다."""
        df = self.frame(name)
        row = int(np.clip(row, tm.WINDOW - 1, len(df) - 1))
        raw = df[tm.SENSORS].values[row - tm.WINDOW + 1:row + 1]
        label = int(df['anomaly'].values[row])
        return self._analyze(name, raw, row=row, true_label=label)

    def detect_from_values(self, case: str, values, kind: str | None = None) -> DetectionResult:
        """모의 입력 모드: 대표 구간의 10초 변동은 유지하고 센서별 평균만 values로 옮겨 판정한다 (기존 앱과 같은 방식).

        kind를 주지 않으면 '정상 운전'은 정상 구간, 고장 조건은 고장 구간의 변동을 쓴다.
        """
        name = self.cases[case]
        kind = kind or ('normal' if case == tm.NORMAL else 'fault')
        base = self.demo['experiments'][name][kind]
        raw = base - base.mean(axis=0) + np.asarray(values, dtype=float)
        return self._analyze(name, raw)

    def score_timeline(self, name: str) -> pd.DataFrame:
        """실험 전체의 시점별 LSTM AE 이상점수. 화면의 시계열 그래프에 쓴다."""
        if name not in self._timelines:
            df, b = self.frame(name), self.bundle(name)
            W = tm.make_windows(b['scaler'].transform(df[tm.SENSORS].values))
            score = ((W - tm.reconstruct(b, W)) ** 2).mean(axis=(1, 2))
            self._timelines[name] = pd.DataFrame({
                'row': np.arange(len(df)),
                'score': np.r_[np.full(tm.WINDOW - 1, np.nan), score],
                'anomaly': df['anomaly'].values.astype(int),
            })
        return self._timelines[name]

    def _analyze(self, name, raw, row=None, true_label=None) -> DetectionResult:
        b = self.bundle(name)
        scaler = b['scaler']
        W = scaler.transform(raw)[None]                  # (1, 10, 8)
        scores = {m: float(s[0]) for m, s in tm.score_all(b, W).items()}
        recon = tm.reconstruct(b, W)[0]
        err = ((W[0] - recon) ** 2).mean(axis=0)         # 센서별 오차 (8,)

        rep = tm.REPRESENTATIVE
        score, threshold = scores[rep], b['thresholds'][rep]
        ratio = score / threshold
        severity = next(label for limit, label in config.SEVERITY_LEVELS if ratio >= limit)

        z = (raw[-1] - scaler.mean_) / scaler.scale_
        contributions = sorted([
            SensorContribution(
                sensor=s, name_ko=config.SENSOR_KO[s], unit=config.SENSOR_UNIT[s],
                error=float(err[i]), share=float(err[i] / err.sum() * 100),
                current=float(raw[-1, i]), normal_mean=float(scaler.mean_[i]), z=float(z[i]))
            for i, s in enumerate(tm.SENSORS)], key=lambda c: c.error, reverse=True)

        votes = {m: (scores[m], b['thresholds'][m], scores[m] > b['thresholds'][m]) for m in tm.COMPARE}
        return DetectionResult(
            experiment=name, is_anomaly=score > threshold, score=score, threshold=threshold,
            severity=severity, contributions=contributions, model_votes=votes,
            window=W[0], recon=recon, row=row, true_label=true_label)


if __name__ == '__main__':
    import json
    detector = PumpAnomalyDetector()
    for name, fault in detector.experiments.items():
        tl = detector.score_timeline(name)
        fault_row = int(tl.loc[tl['anomaly'] == 1, 'row'].median())
        r = detector.detect_at(name, fault_row)
        print(f'\n[{fault}] row {fault_row} 실제 라벨={r.true_label}')
        print(json.dumps(r.to_llm_dict(), ensure_ascii=False, indent=1))
