# 시연용 실험 4개의 모델을 미리 학습해 models 폴더에 저장한다
# 대표 모델 LSTM AE는 파일이 있으면 다시 학습하지 않는다. 비교 모델과 화면용 데이터만 새로 만든다.
# 실행: .\.venv\Scripts\python.exe train_models.py
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA_DIR = HERE / 'dataset' / 'SKAB'
MODEL_DIR = HERE / 'models'
DEMO_DATA = MODEL_DIR / 'demo_data.pkl'
SENSORS = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
           'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']
TRAIN_SIZE = 400   # 각 실험의 앞 400행 = 정상 운전 구간 → 학습용
WINDOW = 10        # LSTM 입력 윈도우 길이 (10초)
EPOCHS = 50
SEED = 42
K = 1.5            # 임계값 = 학습 구간 오차의 99퍼센타일 × 1.5 (Isolation Forest는 × 1.0)
n = len(SENSORS)

# 시연 케이스: 서로 다른 센서에서 고장이 드러나는 실험
DEMO = {
    'valve1/0': '입구 밸브 닫힘',
    'other/7': '로터 불균형(계단)',
    'other/12': '캐비테이션(배수)',
    'other/14': '고온수 공급',
}
NORMAL = '정상 운전'
NORMAL_EXP = 'other/7'   # 고장 전 정상 구간에서 오경보가 없는 실험
# 카드 순서: 정상 → 고장 케이스
CASES = {NORMAL: NORMAL_EXP, **{fault: name for name, fault in sorted(DEMO.items(), key=lambda kv: kv[0] != 'other/7')}}

REPRESENTATIVE = 'LSTM AE'
KERAS_MODELS = ['LSTM AE', 'Conv1D AE', 'Conv2D AE', 'Dense AE']
COMPARE = [*KERAS_MODELS, 'Isolation Forest', '평균 이탈(10초)']


def read_experiment(name):
    return pd.read_csv(DATA_DIR / f'{name}.csv', sep=';', index_col='datetime', parse_dates=True)


def make_windows(Xs, w=WINDOW):
    return np.stack([Xs[i:i + w] for i in range(len(Xs) - w + 1)])


def model_path(name):
    return MODEL_DIR / f"{name.replace('/', '_')}.keras"


def meta_path(name):
    return MODEL_DIR / f"{name.replace('/', '_')}.pkl"


def compare_path(name, model):
    """비교 모델 파일 경로. LSTM AE는 기존 대표 모델 파일을 그대로 쓴다."""
    if model == REPRESENTATIVE:
        return model_path(name)
    tag = model.split()[0].lower()
    return MODEL_DIR / f"{name.replace('/', '_')}__{tag}.keras"


def iforest_path(name):
    return MODEL_DIR / f"{name.replace('/', '_')}__iforest.pkl"


def build_lstm_ae():
    import tensorflow as tf
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input(shape=(WINDOW, n)),
        L.LSTM(16),
        L.RepeatVector(WINDOW),
        L.LSTM(16, return_sequences=True),
        L.TimeDistributed(L.Dense(n))])


def build_conv1d_ae():
    import tensorflow as tf
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input(shape=(WINDOW, n)),
        L.Conv1D(16, 3, padding='same', activation='relu'),
        L.MaxPooling1D(2),
        L.Conv1D(8, 3, padding='same', activation='relu'),
        L.UpSampling1D(2),
        L.Conv1D(16, 3, padding='same', activation='relu'),
        L.Conv1D(n, 3, padding='same')])


def build_conv2d_ae():
    import tensorflow as tf
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input(shape=(WINDOW, n, 1)),
        L.Conv2D(16, (3, 3), padding='same', activation='relu'),
        L.MaxPooling2D((2, 2)),
        L.Conv2D(2, (3, 3), padding='same', activation='relu'),
        L.UpSampling2D((2, 2)),
        L.Conv2D(16, (3, 3), padding='same', activation='relu'),
        L.Conv2D(1, (3, 3), padding='same')])


def build_dense_ae():
    import tensorflow as tf
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.Input(shape=(n,)),
        L.Dense(16, activation='relu'), L.Dropout(0.2),
        L.Dense(4, activation='relu'),
        L.Dense(16, activation='relu'),
        L.Dense(n)])


def fit_ae(model, X):
    import tensorflow as tf
    model.compile(optimizer='adam', loss='mse')
    early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    hist = model.fit(X, X, epochs=EPOCHS, batch_size=32, validation_split=0.2, callbacks=[early], verbose=0)
    return len(hist.history['loss'])


def train(name):
    """대표 모델 LSTM AE를 학습해 저장한다."""
    import joblib
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler

    tf.keras.utils.set_random_seed(SEED)
    X = read_experiment(name)[SENSORS].values
    scaler = StandardScaler().fit(X[:TRAIN_SIZE])
    W = make_windows(scaler.transform(X[:TRAIN_SIZE]))
    model = build_lstm_ae()
    epochs = fit_ae(model, W)
    train_err = ((W - model.predict(W, verbose=0)) ** 2).mean(axis=(1, 2))
    model.save(model_path(name))
    joblib.dump({'scaler': scaler, 'train_err': train_err}, meta_path(name))
    return epochs


def train_compare(name):
    """비교 모델을 대표 모델과 같은 스케일러, 같은 학습 구간으로 학습해 저장한다."""
    import joblib
    import tensorflow as tf
    from sklearn.ensemble import IsolationForest

    scaler = joblib.load(meta_path(name))['scaler']
    Xs = scaler.transform(read_experiment(name)[SENSORS].values[:TRAIN_SIZE])
    W = make_windows(Xs)
    for label, build, X_train in [('Conv1D AE', build_conv1d_ae, W),
                                  ('Conv2D AE', build_conv2d_ae, W[..., None]),
                                  ('Dense AE', build_dense_ae, Xs)]:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = build()
        fit_ae(model, X_train)
        model.save(compare_path(name, label))
    joblib.dump(IsolationForest(n_estimators=200, random_state=SEED).fit(Xs), iforest_path(name))


def load_models(name):
    """실험 하나의 스케일러와 모델을 불러온다. 데이터셋 파일은 필요 없다."""
    import joblib
    import tensorflow as tf

    return {'scaler': joblib.load(meta_path(name))['scaler'], 'iforest': joblib.load(iforest_path(name)),
            'keras': {m: tf.keras.models.load_model(compare_path(name, m)) for m in KERAS_MODELS}}


def load_bundle(name):
    """모델을 불러오고 원본 데이터의 학습 구간으로 모델별 임계값을 계산한다 (학습할 때만 사용)."""
    bundle = load_models(name)
    scaler = bundle['scaler']
    Xs = scaler.transform(read_experiment(name)[SENSORS].values[:TRAIN_SIZE])
    W = make_windows(Xs)
    # Dense AE와 Isolation Forest는 1초 입력이라 앞 400행 전체로 임계값을 정한다 (메인 노트북과 같음)
    train_scores = score_all(bundle, W, Xs_rows=Xs)
    bundle['thresholds'] = {m: np.percentile(s, 99) * (1.0 if m == 'Isolation Forest' else K)
                            for m, s in train_scores.items()}
    return bundle


def reconstruct(bundle, W):
    """대표 모델의 복원 값. 센서별 오차와 복원 그래프에 쓴다."""
    return bundle['keras'][REPRESENTATIVE].predict(W, verbose=0)


def score_all(bundle, W, Xs_rows=None):
    """표준화된 윈도우(개수, 10, 8)를 받아 모델별 점수를 구한다. 클수록 이상이다.

    Dense AE와 Isolation Forest는 1초 입력이라 윈도우의 마지막 행을 쓴다.
    Xs_rows를 주면 1초 입력 모델은 그 행들로 계산한다.
    """
    keras = bundle['keras']
    rows = W[:, -1, :] if Xs_rows is None else Xs_rows
    return {
        'LSTM AE': ((W - keras['LSTM AE'].predict(W, verbose=0)) ** 2).mean(axis=(1, 2)),
        'Conv1D AE': ((W - keras['Conv1D AE'].predict(W, verbose=0)) ** 2).mean(axis=(1, 2)),
        'Conv2D AE': ((W[..., None] - keras['Conv2D AE'].predict(W[..., None], verbose=0)) ** 2).mean(axis=(1, 2, 3)),
        'Dense AE': ((rows - keras['Dense AE'].predict(rows, verbose=0)) ** 2).mean(axis=1),
        'Isolation Forest': -bundle['iforest'].score_samples(rows),
        '평균 이탈(10초)': (W ** 2).mean(axis=(1, 2)),
    }


def median_row(score, rows):
    """점수가 중앙값인 행. 가장 좋거나 나쁜 시점만 고르지 않기 위해 쓴다."""
    return int(rows[np.argsort(score[rows])[len(rows) // 2]])


def build_demo_data():
    """화면에서 쓸 슬라이더 범위, 기준 구간, 조건별 모델 성능을 미리 계산해 저장한다."""
    import joblib

    experiments, by_case = {}, {}
    for name in DEMO:
        bundle = load_bundle(name)
        df = read_experiment(name)
        X, y = df[SENSORS].values, df['anomaly'].values.astype(int)
        scores = score_all(bundle, make_windows(bundle['scaler'].transform(X)))
        # 윈도우 i의 점수는 마지막 행 i + WINDOW - 1의 점수다
        scores = {m: np.r_[np.full(WINDOW - 1, np.nan), s] for m, s in scores.items()}
        start = int(y.argmax())
        pre_rows, fault_rows = np.arange(TRAIN_SIZE, start), np.where(y == 1)[0]
        normal_row = median_row(scores[REPRESENTATIVE], pre_rows)
        fault_row = median_row(scores[REPRESENTATIVE], fault_rows)
        mean, std = bundle['scaler'].mean_, bundle['scaler'].scale_
        experiments[name] = {
            'mean': mean, 'std': std,
            'low': np.minimum(X.min(axis=0), mean - 10 * std),
            'high': np.maximum(X.max(axis=0), mean + 10 * std),
            'normal': X[normal_row - WINDOW + 1:normal_row + 1],
            'fault': X[fault_row - WINDOW + 1:fault_row + 1],
            'normal_row': normal_row, 'fault_row': fault_row,
            'thresholds': {m: float(v) for m, v in bundle['thresholds'].items()},
        }
        by_case[DEMO[name]] = {
            m: {'detect': float((scores[m][fault_rows] > bundle['thresholds'][m]).mean() * 100),
                'false_alarm': float((scores[m][pre_rows] > bundle['thresholds'][m]).mean() * 100)}
            for m in COMPARE}
    joblib.dump({'experiments': experiments, 'by_case': by_case}, DEMO_DATA)
    return experiments, by_case


if __name__ == '__main__':
    MODEL_DIR.mkdir(exist_ok=True)
    for name, fault in DEMO.items():
        if not (model_path(name).exists() and meta_path(name).exists()):
            print(f'{name} ({fault}): LSTM AE {train(name)} epoch 학습')
        train_compare(name)
        print(f'{name} ({fault}): 비교 모델 저장')
    _, table = build_demo_data()
    for fault, row in table.items():
        print(fault, {m: (round(v['detect'], 1), round(v['false_alarm'], 1)) for m, v in row.items()})
    print(f'{DEMO_DATA.name} 저장')
