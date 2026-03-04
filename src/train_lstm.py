import os
import gc
import glob
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

# 保持整体思路：用过去10天的“+-”字符序列作为特征，滚动3个月训练，逐日预测并累乘收益
FEATURE_LOOKBACK = 10
TRAIN_LOOKBACK_MONTHS = 3
INITIAL_INVESTMENT = 100.0
CASH_FILE = "../data/process/cash.csv"

# 三种信号列：oc=开→收，cc=昨收→今收，co=昨收→今开
# SIGNAL_COLS = ["oc", "cc", "co"]
SIGNAL_COLS = ["oc"]


def build_feature_table(df: pd.DataFrame, signal_col: str = "oc") -> pd.DataFrame:
    df = df.copy()
    # 统一日期列
    if "Date" in df.columns:
        df["date"] = pd.to_datetime(df["Date"])
    else:
        raise ValueError("缺少日期列 Date/date")

    # 验证并提取指定信号列作为当日收益率；cash 收益为 0
    if signal_col not in df.columns:
        raise ValueError(f"缺少 {signal_col} 列，无法计算当日收益率")
    df["return_stock"] = df[signal_col].astype(float)

    # "+/-"标签与 FEATURE_LOOKBACK 日字符特征
    label_up_down = np.where(df["return_stock"] > 0, "+", "-")
    label_cash = np.where(df["return_stock"] == 0, "+", "-")
    feats = []
    targets = []
    for i in range(len(labels)):
        if i >= FEATURE_LOOKBACK:
            feats.append("".join(labels[i - FEATURE_LOOKBACK: i]))
            targets.append(labels[i])
        else:
            feats.append(None)
            targets.append(None)
    df["feature_up_down"] = feats
    df["target"] = targets
    df = df.dropna(subset=["feature_up_down"])
    return df[["date", "feature_up_down", "target", "return_stock"]]


def _backtest_single_signal(df: pd.DataFrame, cash_df: pd.DataFrame, signal_col: str) -> pd.DataFrame:
    # 确保日期列格式为datetime
    df['date'] = pd.to_datetime(df['date'])
    cash_df['date'] = pd.to_datetime(cash_df['date'])

    # 创建 return_label 列
    df['return_label'] = df[signal_col].apply(lambda x: '+' if x > 0 else '-')
    cash_df['return_label'] = cash_df['return'].apply(lambda x: '+' if x > 0 else '-')

    data = build_feature_table(df, signal_col=signal_col)
    if data.empty:
        return pd.DataFrame(columns=["date", "value", "predicted", "signal"])

    current_value = INITIAL_INVESTMENT
    records = []
    total_correct = 0  # 累计预测正确次数
    total_pred = 0  # 累计预测总次数
    DEBUG = True
    data_min = pd.to_datetime("2025-10-03") if DEBUG else data["date"].min()
    start_date = data_min + pd.DateOffset(months=TRAIN_LOOKBACK_MONTHS)
    end_date = data["date"].max()
    current_date = start_date

    while current_date <= end_date:
        print(f"current_date==> {current_date}   end_date==> {end_date}")
        train_start = current_date - pd.DateOffset(months=TRAIN_LOOKBACK_MONTHS)
        train_end = current_date - pd.Timedelta(days=1)
        train_mask = (data["date"] >= train_start) & (data["date"] <= train_end)
        test_mask = (data["date"] == current_date)
        train_set = data[train_mask]
        test_set = data[test_mask]
        train_set = train_set.dropna(subset=['feature_up_down', "target"])
        test_set = test_set.dropna(subset=['feature_up_down', "target"])

        if not test_set.empty and not train_set.empty:
            X_train = train_set["feature_up_down"].values
            y_train = train_set["target"].values
            X_test = test_set["feature_up_down"].values

            tokenizer = Tokenizer(char_level=True)
            tokenizer.fit_on_texts(X_train)
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)

            max_len = max(len(s) for s in X_train_seq)
            X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding="post")
            X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding="post")
            X_train_pad = np.expand_dims(X_train_pad, axis=-1)
            X_test_pad = np.expand_dims(X_test_pad, axis=-1)

            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_train_cat = to_categorical(y_train_enc)

            model = Sequential()
            model.add(LSTM(50, input_shape=(max_len, 1)))
            model.add(Dense(2, activation='softmax'))  # 假设有两个输出类别

            # 编译模型
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # 训练模型
            model.fit(X_train_pad, y_train_cat, epochs=10, batch_size=32)

            # 进行模型预测
            y_pred = model.predict(X_test_pad)
            print("y_pred==>", y_pred)
            decisions = np.argmax(y_pred, axis=1)
            pred_labels = le.inverse_transform(decisions)
            print(f"decisions c==> {pred_labels}, target==> {test_set['target'].values}")

            # 累计准确率统计
            y_test_enc = le.transform(test_set["target"].values)
            print(f"y_test_enc {y_test_enc}   decisions：{decisions}")
            total_correct += int(np.sum(decisions == y_test_enc))
            total_pred += len(decisions)
            for i, decision in enumerate(decisions):
                stock_rate = float(test_set.iloc[i]["return_stock"])
                pred_class = le.inverse_transform([decision])[0]
                if pred_class == "+":
                    current_value *= (1.0 + stock_rate)
                else:
                    current_value *= 1.0
                records.append([test_set.iloc[i]["date"], current_value, pred_class, y_test_enc, signal_col])

            del model
            K.clear_session()
            gc.collect()

        current_date += pd.Timedelta(days=1)

    if total_pred > 0:
        overall_acc = total_correct / total_pred
        print(f"[{signal_col}] 整体准确率: {overall_acc:.2%} ({total_correct}/{total_pred})")

    return pd.DataFrame(records, columns=["date", "value", "predicted", "signal"])


def backtest_one_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """对 oc / cc / co 三种信号分别回测，合并结果返回。"""
    available = [col for col in SIGNAL_COLS if col in df.columns]
    if not available:
        raise ValueError(f"数据中缺少所有信号列 {SIGNAL_COLS}，请检查输入")

    # 生成现金基准文件：若已存在则先删除，再按 df 的日期列重建
    cash_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), CASH_FILE)
    if os.path.exists(cash_path):
        os.remove(cash_path)
    date_col = "date" if "date" in df.columns else "Date"
    cash_df = pd.DataFrame({
        "Date":   pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
        "price":  1,
        "return": 1e-6,
    })
    os.makedirs(os.path.dirname(cash_path), exist_ok=True)
    cash_df.to_csv(cash_path, index=False)
    print(f"cash file saved: {cash_path} ({len(cash_df)} rows)")

    cash_df = pd.read_csv(cash_path)

    parts = []
    for sig in available:
        print(f"  回测信号: {sig}")
        result = _backtest_single_signal(df, cash_df, signal_col=sig)
        parts.append(result)

    return pd.concat(parts, ignore_index=True)


def main():
    base = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base, "data", "process")
    out_dir = os.path.join(base, "data", "result")
    os.makedirs(out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for p in paths:
        try:
            df = pd.read_csv(p)
            ticker = os.path.splitext(os.path.basename(p))[0]
            print(f"处理: {ticker}")
            result = backtest_one_ticker(df)
            out_path = os.path.join(out_dir, f"{ticker}_backtest.csv")
            result.to_csv(out_path, index=False)
            print(f"saved: {out_path}")
        except Exception as e:
            print(f"failed on {p}: {e}")


if __name__ == "__main__":
    main()
