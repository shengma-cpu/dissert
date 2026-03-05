import os
import gc
import glob
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, Dense
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

# 保持整体思路：用过去10天的“+-”字符序列作为特征，滚动3个月训练，逐日预测并累乘收益
FEATURE_LOOKBACK = 10
TRAIN_LOOKBACK_MONTHS = 12
INITIAL_INVESTMENT = 100.0
MODEL_TYPE = ["lstm", "gru"]
# START_TIME = "2026-2-01"
START_TIME = "2024-05-01"

# 三种信号列：oc=开→收，cc=昨收→今收，co=昨收→今开
# SIGNAL_COLS = ["oc", "cc", "co"]
SIGNAL_COLS = ["oc"]

base = os.path.dirname(os.path.dirname(__file__))

def build_feature_table(df: pd.DataFrame, signal_col: str = "oc"):
    # 验证并提取指定信号列作为当日收益率；cash 收益为 0
    if signal_col not in df.columns:
        raise ValueError(f"缺少 {signal_col} 列，无法计算当日收益率")

    feats = []
    for i in range(len(df)):
        if i >= 9:
            feats.append("".join(df['return_label_df'][i - FEATURE_LOOKBACK: i]))
        else:
            feats.append(None)
    df["feature_df"] = feats


def _backtest_single_signal(df: pd.DataFrame, cash_df: pd.DataFrame, signal_col: str, model_type: str) -> pd.DataFrame:
    # 确保日期列格式为datetime
    df['date'] = pd.to_datetime(df['Date'])
    cash_df['date'] = pd.to_datetime(cash_df['Date'])

    # 创建 return_label 列
    df['return_label_df'] = df[signal_col].apply(lambda x: '+' if x > 0 else '-')
    cash_df['return_label_cash'] = cash_df['return'].apply(lambda x: '+' if x > 0 else '-')
    feats = []
    for i in range(len(cash_df)):
        if i >= 9:
            feats.append("".join(cash_df['return_label_cash'][i - FEATURE_LOOKBACK: i]))
        else:
            feats.append(None)
    cash_df["feature_cash"] = feats

    build_feature_table(df, signal_col=signal_col)

    combined_data = pd.merge(df, cash_df, on='date')
    combined_data['combined_feature'] = combined_data['feature_df']  # 只使用ixic的特征
    combined_data['target'] = combined_data.apply(lambda row: 'stock' if row[signal_col] > row['return'] else 'cash', axis=1)
    
    current_value = INITIAL_INVESTMENT
    records = []
    total_correct = 0  # 累计预测正确次数
    total_pred = 0  # 累计预测总次数
    DEBUG = True
    start_date = pd.to_datetime(START_TIME) if DEBUG else combined_data["date"].min()
    end_date = combined_data["date"].max()
    current_date = start_date

    while current_date <= end_date:
        print("================================================================================")
        print(f"current_date==> {current_date}   end_date==> {end_date}")
        train_start = current_date - pd.DateOffset(months=TRAIN_LOOKBACK_MONTHS)
        train_end = current_date - pd.Timedelta(days=1)
        train_mask = (combined_data["date"] >= train_start) & (combined_data["date"] <= train_end)
        test_mask = (combined_data["date"] == current_date)
        train_set = combined_data[train_mask]
        test_set = combined_data[test_mask]

        train_set = train_set.dropna(subset=['combined_feature'])
        test_set = test_set.dropna(subset=['combined_feature'])

        if not test_set.empty:
            X_train = train_set['combined_feature']
            y_train = train_set['target']
            # print("X_train:", X_train.head(10))
            # print("y_train:", y_train.head(10))
            X_test = test_set['combined_feature']
            y_test = test_set['target']
            # print("X_test:", X_test.head(10))
            # print("y_test:", y_test.head(10))

            # 将字符串序列转换为数值
            tokenizer = Tokenizer(char_level=True)
            tokenizer.fit_on_texts(X_train)
            X_train_seq = tokenizer.texts_to_sequences(X_train)
            X_test_seq = tokenizer.texts_to_sequences(X_test)
            # print("X_train_seq:", X_train_seq[:10])
            # 确保所有序列具有相同的长度
            max_length = max([len(seq) for seq in X_train_seq])
            X_train_padded = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
            X_test_padded = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
            # print("X_train_padded:", X_train_padded[:10])
            # print("X_test_padded:", X_test_padded[:10])

            # LSTM 输入形状需要为 (样本数, 时间步长, 特征数)，需转换为 float32
            X_train_padded = np.expand_dims(X_train_padded, axis=-1).astype(np.float32)
            X_test_padded = np.expand_dims(X_test_padded, axis=-1).astype(np.float32)
            
            # 将目标变量转换为分类编码
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            # y_test_encoded = label_encoder.transform(y_test)
            y_train_categorical = to_categorical(y_train_encoded)
            # y_test_categorical = to_categorical(y_test_encoded)

            # 创建 LSTM 模型
            model = Sequential()
            if model_type == "lstm":
                model.add(LSTM(50, input_shape=(max_length, 1)))
            elif model_type == "bilstm":
                model.add(Bidirectional(LSTM(50, input_shape=(max_length, 1))))
            elif model_type == "gru":
                model.add(GRU(50, input_shape=(max_length, 1)))
            elif model_type == "bigru":
                model.add(Bidirectional(GRU(50, input_shape=(max_length, 1))))
            model.add(Dense(2, activation='softmax'))  # 假设有两个输出类别

            # 编译模型
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # 训练模型
            model.fit(X_train_padded, y_train_categorical, epochs=10, batch_size=32)

            # 进行模型预测
            y_pred = model.predict(X_test_padded)
            print("y_pred==>", y_pred)
            decisions = np.argmax(y_pred, axis=1)
            pred_labels = label_encoder.inverse_transform(decisions)
            print(f"decisions c==> {pred_labels}, target==> {test_set['target'].values}")

            # 累计准确率统计
            y_test_enc = label_encoder.transform(test_set["target"].values)
            # print(f"y_test_enc {y_test_enc}   decisions：{decisions}")
            total_correct += int(np.sum(decisions == y_test_enc))
            total_pred += len(decisions)
            for i, decision in enumerate(decisions):
                # 获取对应日期的收益率
                date = test_set.iloc[i]['date']
                ixic_rate = combined_data.loc[combined_data['date'] == date, signal_col].values
                cash_rate = combined_data.loc[combined_data['date'] == date, 'return'].values
                print(f"date==> {date}   ixic_rate==> {ixic_rate}   cash_rate==> {cash_rate}")
                # 检查收益率是否为 None，如果是则设为 0
                ixic_rate = ixic_rate[0] if len(ixic_rate) > 0 else 0
                cash_rate = cash_rate[0] if len(cash_rate) > 0 else 0
                print(f"b current_value==> {current_value}")
                # 根据预测决策更新投资金额
                pred_class = label_encoder.inverse_transform([decision])[0]
                if pred_class == "stock":
                    current_value *= (1 + ixic_rate)
                else:  # 'cash'
                    current_value *= (1 + cash_rate)
                print(f"a current_value==> {current_value}")
                real_label = label_encoder.inverse_transform([y_test_enc[i]])[0]
                records.append([test_set.iloc[i]["date"], current_value, pred_class, real_label, signal_col, y_pred[i].tolist()])

            del model
            K.clear_session()
            gc.collect()

        current_date += pd.Timedelta(days=1)

    if total_pred > 0:
        overall_acc = total_correct / total_pred
        print(f"[{signal_col}] 整体准确率: {overall_acc:.2%} ({total_correct}/{total_pred})")

    return pd.DataFrame(records, columns=["date", "value", "predicted", "real", "signal", "y_pred"])

def backtest_one_ticker(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """对 oc / cc / co 三种信号分别回测，合并结果返回。"""
    available = [col for col in SIGNAL_COLS if col in df.columns]
    if not available:
        raise ValueError(f"数据中缺少所有信号列 {SIGNAL_COLS}，请检查输入")

    # 生成现金基准
    date_col = "date" if "date" in df.columns else "Date"
    cash_df = pd.DataFrame({
        "Date":   pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
        "price":  1,
        "return": 1e-6,
    })
    # print(f"cash_df: {cash_df.head(5)} \n {cash_df.head(-5)}")
    out_dir = os.path.join(base, "data", "result")
    os.makedirs(out_dir, exist_ok=True)

    for model_type in MODEL_TYPE:
        for sig in available:
            print(f"  回测信号: {sig}")
            result = _backtest_single_signal(df, cash_df, signal_col=sig, model_type=model_type)
            
            out_path = os.path.join(out_dir, f"{ticker}_{sig}_{model_type}_backtest.csv")
            result.to_csv(out_path, index=False)
            print(f"saved: {out_path}")

    # result = pd.concat(parts, ignore_index=True)

def main():
    data_dir = os.path.join(base, "data", "process")
    # paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    paths = ["/Users/ms/Desktop/ixic.csv"]
    for p in paths:
        df = pd.read_csv(p)
        ticker = os.path.splitext(os.path.basename(p))[0]
        print(f"处理: {ticker}")
        backtest_one_ticker(df, ticker)

if __name__ == "__main__":
    main()
