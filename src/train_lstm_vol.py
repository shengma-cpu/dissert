import os
import gc
import glob
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, GRU, Dense, Flatten, RepeatVector, Permute, Multiply, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

# 保持整体思路：用过去10天的“+-”字符序列作为特征，滚动3个月训练，逐日预测并累乘收益
FEATURE_LOOKBACK = 10
TRAIN_LOOKBACK_MONTHS = 12
INITIAL_INVESTMENT = 100.0
MODEL_TYPE = ["gru"]
START_TIME = "2026-2-01"
# START_TIME = "2024-05-01"

# 三种信号列：oc=开→收，cc=昨收→今收，co=昨收→今开
# SIGNAL_COLS = ["oc", "cc", "co"]
SIGNAL_COLS = ["oc"]

base = os.path.dirname(os.path.dirname(__file__))

def build_feature_table(df: pd.DataFrame, signal_col: str = "oc"):
    """
    不仅构建标签序列，还准备好数值特征。
    """
    if signal_col not in df.columns:
        raise ValueError(f"缺少 {signal_col} 列")
    
    # 1. 预处理：将标签转换为 0/1 数值
    df['return_label_df'] = df[signal_col].apply(lambda x: 1 if x > 0 else 0)
    
    # 2. 准备特征矩阵：[标签, 成交量百分比]
    # 假设 df["v"] 已经是百分比（如 0.02 代表 2%）
    label_data = df['return_label_df'].values
    volume_data = df['v'].fillna(0).values 

    feats = []
    for i in range(len(df)):
        if i >= FEATURE_LOOKBACK:
            # 提取过去 10 天的 [标签, 成交量] 组合
            # shape: (10, 2)
            seq = np.column_stack((
                label_data[i - FEATURE_LOOKBACK : i],
                volume_data[i - FEATURE_LOOKBACK : i]
            ))
            feats.append(seq)
        else:
            feats.append(None)
    
    df["feature_matrix"] = feats


def _backtest_single_signal(df: pd.DataFrame, cash_df: pd.DataFrame, signal_col: str, model_type: str) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['Date'])
    cash_df['date'] = pd.to_datetime(cash_df['Date'])

    # 构建特征表（包含成交量）
    build_feature_table(df, signal_col=signal_col)

    combined_data = pd.merge(df, cash_df, on='date')
    # 目标： stock (ixic > cash) or cash
    combined_data['target'] = combined_data.apply(lambda row: 'stock' if row[signal_col] > row['return'] else 'cash', axis=1)
    
    current_value = INITIAL_INVESTMENT
    records = []
    total_correct = 0
    total_pred = 0
    start_date = pd.to_datetime(START_TIME)
    end_date = combined_data["date"].max()
    current_date = start_date

    while current_date <= end_date:
        print("================================================================================")
        print(f"current_date==> {current_date} | 模型: {model_type}")
        
        train_start = current_date - pd.DateOffset(months=TRAIN_LOOKBACK_MONTHS)
        train_end = current_date - pd.Timedelta(days=1)
        
        train_set = combined_data[(combined_data["date"] >= train_start) & (combined_data["date"] <= train_end)].dropna(subset=['feature_matrix'])
        test_set = combined_data[combined_data["date"] == current_date].dropna(subset=['feature_matrix'])

        if not test_set.empty and len(train_set) > 10:
            # --- 数据准备 ---
            # 转换 list of arrays 为 3D numpy array: (样本数, 10, 2)
            X_train = np.stack(train_set['feature_matrix'].values).astype(np.float32)
            X_test = np.stack(test_set['feature_matrix'].values).astype(np.float32)
            
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(train_set['target'])
            y_train_categorical = to_categorical(y_train_encoded, num_classes=2)

            # --- 模型定义: BiGRU-Attention (双特征输入) ---
            # 这里的输入维度变为 (10, 2)
            inputs = Input(shape=(FEATURE_LOOKBACK, 2))
            
            gru_out = Bidirectional(GRU(50, return_sequences=True))(inputs)

            # Attention 模块
            attention = Dense(1, activation='tanh')(gru_out) 
            attention = Flatten()(attention)
            attention = Dense(FEATURE_LOOKBACK, activation='softmax')(attention)
            attention = RepeatVector(100)(attention) # 50*2 = 100
            attention = Permute([2, 1])(attention)

            sent_representation = Multiply()([gru_out, attention])
            sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)

            probabilities = Dense(2, activation='softmax')(sent_representation)
            model = Model(inputs=inputs, outputs=probabilities)

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            # 训练 (增加成交量后，模型能学到更多动能特征)
            model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, verbose=0)

            # 预测
            y_pred = model.predict(X_test, verbose=0)
            print("y_pred:", y_pred)
            decision = np.argmax(y_pred, axis=1)[0]
            pred_label = label_encoder.inverse_transform([decision])[0]
            
            # 净值更新
            date = test_set.iloc[0]['date']
            ixic_rate = test_set.iloc[0][signal_col]
            cash_rate = test_set.iloc[0]['return']
            
            if pred_label == "stock":
                current_value *= (1 + ixic_rate)
            else:
                current_value *= (1 + cash_rate)
            
            real_label = test_set.iloc[0]['target']
            if pred_label == real_label: total_correct += 1
            total_pred += 1
            
            records.append([date, current_value, pred_label, real_label, signal_col, y_pred[0].tolist()])
            print(f"Pred: {pred_label} | Real: {real_label} | Value: {current_value:.2f} | Accuracy: {total_correct/total_pred:.2%}")

            del model
            K.clear_session()
            gc.collect()

        current_date += pd.Timedelta(days=1)

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
            
            out_path = os.path.join(out_dir, f"{ticker}_{sig}_{model_type}_vol_backtest.csv")
            result.to_csv(out_path, index=False)
            print(f"saved: {out_path}")

    # result = pd.concat(parts, ignore_index=True)

def main():
    data_dir = os.path.join(base, "data", "process")
    # paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    # paths = ["/Users/ms/Desktop/ixic.csv"]
    paths = [data_dir+"/AAPL.csv"]
    for p in paths:
        df = pd.read_csv(p)
        ticker = os.path.splitext(os.path.basename(p))[0]
        print(f"处理: {ticker}")
        backtest_one_ticker(df, ticker)

if __name__ == "__main__":
    main()
