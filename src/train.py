import os
import gc
import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, GRU, Dense, Flatten, RepeatVector, Permute, Multiply, Lambda, Dropout
from tensorflow.keras.utils import to_categorical
from keras import backend as K
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 全局实验配置 (Experiment Configurations)
# ==========================================
WRITE_RESULT_FOURCE = True
FEATURE_LOOKBACK = 10
INITIAL_INVESTMENT = 100.0
START_TIME = "2026-01-01"
# START_TIME = "2024-05-01"

# 论文需要对比的四个维度：
# SIGNAL_COLS = ["oc", "cc", "co"]
# MODEL_TYPES = ["lstm", "attention"]
# USE_VOLUME_LIST = [False, True]
# TRAIN_MONTHS_LIST = [3, 6, 12, 36]

SIGNAL_COLS = ["oc"]
MODEL_TYPES = ["attention"]
USE_VOLUME_LIST = [False, True]
TRAIN_MONTHS_LIST = [36]

base = os.path.dirname(os.path.dirname(__file__))

# ==========================================
# 模块 1: 特征工程 (Feature Engineering)
# ==========================================
def build_feature_table(df: pd.DataFrame, signal_col: str, use_volume: bool):
    """
    统一构建特征矩阵：
    - use_volume=False: 返回 shape 为 (10, 1) 的标签序列 (只包含0/1)
    - use_volume=True:  返回 shape 为 (10, 2) 的复合序列 (标签 + 成交量百分比)
    """
    if signal_col not in df.columns:
        raise ValueError(f"数据中缺少信号列: {signal_col}")
    if use_volume and "v" not in df.columns:
        print("警告: 开启了成交量特征但数据中缺少 'v' 列，自动填充为0")
        df["v"] = 0.0
    
    # 1. 将涨跌转化为 1/0 数值分类
    df['return_label_df'] = df[signal_col].apply(lambda x: 1 if x > 0 else 0)
    
    label_data = df['return_label_df'].values
    volume_data = df['v'].fillna(0).values if use_volume else None

    feats = []
    for i in range(len(df)):
        if i >= FEATURE_LOOKBACK:
            if use_volume:
                # 包含成交量，形状: (10, 2)
                seq = np.column_stack((
                    label_data[i - FEATURE_LOOKBACK : i],
                    volume_data[i - FEATURE_LOOKBACK : i]
                ))
            else:
                # 不含成交量，形状: (10, 1)
                seq = np.expand_dims(label_data[i - FEATURE_LOOKBACK : i], axis=-1)
            feats.append(seq)
        else:
            feats.append(None)
    
    df["feature_matrix"] = feats
    return df

# ==========================================
# 模块 2: 模型工厂 (Model Factory)
# ==========================================
def build_model(model_type: str, input_shape: tuple):
    """
    根据配置动态生成网络结构
    """
    if model_type == "lstm":
        # 传统 LSTM 模型
        model = Sequential()
        model.add(LSTM(50, input_shape=input_shape))
        model.add(Dense(2, activation='softmax'))
    elif model_type == "gru":
        model = Sequential()
        model.add(GRU(50, input_shape=input_shape))
        model.add(Dense(2, activation='softmax'))
    elif model_type == "attention":
        # 改进的 BiGRU-Attention 模型
        inputs = Input(shape=input_shape)
        gru_out = Bidirectional(GRU(50, return_sequences=True))(inputs)
        gru_out = Dropout(0.2)(gru_out)

        # Attention 机制计算权重
        attention = Dense(1, activation='tanh')(gru_out) 
        attention = Flatten()(attention)
        attention = Dense(input_shape[0], activation='softmax')(attention)  # input_shape[0] 即 LOOKBACK
        attention = RepeatVector(100)(attention)  # BiGRU(50) 拼接后维度为 100
        attention = Permute([2, 1])(attention)

        # 加权求和
        sent_representation = Multiply()([gru_out, attention])
        sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)

        probabilities = Dense(2, activation='softmax')(sent_representation)
        model = Model(inputs=inputs, outputs=probabilities)
    
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 模块 3: 核心回测循环 (Backtesting Engine)
# ==========================================
def _backtest_single_experiment(df: pd.DataFrame, cash_df: pd.DataFrame, 
                                signal_col: str, model_type: str, 
                                use_volume: bool, train_months: int) -> pd.DataFrame:
        
    df['date'] = pd.to_datetime(df['Date'])
    cash_df['date'] = pd.to_datetime(cash_df['Date'])

    # 提取特征
    df = build_feature_table(df, signal_col, use_volume)

    combined_data = pd.merge(df, cash_df, on='date')
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
        print(f"current_date==> {current_date} | [信号: {signal_col}] | [模型: {model_type}] | [加量: {use_volume}] | [窗口: {train_months}月]")
        train_start = current_date - pd.DateOffset(months=train_months)
        train_end = current_date - pd.Timedelta(days=1)
        
        train_set = combined_data[(combined_data["date"] >= train_start) & (combined_data["date"] <= train_end)].dropna(subset=['feature_matrix'])
        test_set = combined_data[combined_data["date"] == current_date].dropna(subset=['feature_matrix'])

        if not test_set.empty:
            if len(train_set) <= 10:
                print(f"=====>训练集样本数不足: {len(train_set)}，跳过")
                current_date += pd.Timedelta(days=1)
                continue
            # 1. 准备 Numpy 格式的特征与标签
            X_train = np.stack(train_set['feature_matrix'].values).astype(np.float32)
            X_test = np.stack(test_set['feature_matrix'].values).astype(np.float32)
            
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(train_set['target'])
            y_train_categorical = to_categorical(y_train_encoded, num_classes=2)

            # 2. 构建与训练模型
            input_shape = (FEATURE_LOOKBACK, 2 if use_volume else 1)
            model = build_model(model_type, input_shape)
            
            # 使用 verbose=0 保持打印清爽，避免大量进度条刷屏
            model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, verbose=0)

            # 3. 进行预测
            # y_pred = model.predict(X_test, verbose=0)
            y_pred = model(X_test, training=False).numpy()
            decision = np.argmax(y_pred, axis=1)[0]
            pred_label = label_encoder.inverse_transform([decision])[0]
            
            # 4. 统计与计算当日收益
            date = test_set.iloc[0]['date']
            ixic_rate = test_set.iloc[0][signal_col]
            cash_rate = test_set.iloc[0]['return']
            
            if pred_label == "stock":
                current_value *= (1 + ixic_rate)
            else:
                current_value *= (1 + cash_rate)
            
            real_label = test_set.iloc[0]['target']
            if pred_label == real_label: 
                total_correct += 1
            total_pred += 1
            
            # 打印调试信息
            print(
                  f"当前净值: {current_value:>8.2f} | acc: {total_correct/total_pred:.2%},  概率: [cash:{y_pred[0][0]:.3f}, stock:{y_pred[0][1]:.5f}]"
                  f"[预测: {pred_label:<5} | 实际: {real_label:<5} | "
                  )
            
            records.append([date, current_value, pred_label, real_label, y_pred[0].tolist()])

            # 清理内存，防止内存泄漏
            del model
            K.clear_session()
            gc.collect()

        current_date += pd.Timedelta(days=1)

    if total_pred > 0:
        overall_acc = total_correct / total_pred
        print(f"=== 实验结束 === 整体准确率: {overall_acc:.2%} ({total_correct}/{total_pred}) 最终净值: {current_value:.2f}\n")

    return pd.DataFrame(records, columns=["date", "value", "predicted", "real", "y_pred"])

# ==========================================
# 模块 4: 主调度器 (Experiment Scheduler)
# ==========================================
def backtest_one_ticker(df: pd.DataFrame, ticker: str):
    available_signals = [col for col in SIGNAL_COLS if col in df.columns]
    if not available_signals:
        raise ValueError(f"数据中缺少所有配置的信号列 {SIGNAL_COLS}，请检查输入")

    # 生成基准无风险现金收益 (0%)
    date_col = "date" if "date" in df.columns else "Date"
    cash_df = pd.DataFrame({
        "Date":   pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
        "price":  1,
        "return": 1e-6,
    })
    out_dir = os.path.join(base, "data", "result")
    os.makedirs(out_dir, exist_ok=True)

    # 嵌套循环：全自动跑完所有交叉对比实验
    for sig in available_signals:
        for model_type in MODEL_TYPES:
            for use_volume in USE_VOLUME_LIST:
                for train_months in TRAIN_MONTHS_LIST:

                    # 生成明确的文件名： 股票名_信号_模型_是否加量_训练窗口.csv
                    vol_tag = "vol" if use_volume else "novol"
                    out_filename = f"{ticker}_{sig}_{model_type}_{vol_tag}_{train_months}m.csv"
                    out_path = os.path.join(out_dir, out_filename)
                    
                    # 如果已经跑过，可以跳过 (方便中断后继续运行)
                    if os.path.exists(out_path):
                        if WRITE_RESULT_FOURCE:
                            os.remove(out_path)
                        else:
                            print(f"文件已存在，跳过: {out_filename}")
                            continue
                    
                    # 运行单次回测
                    result_df = _backtest_single_experiment(
                        df.copy(), cash_df.copy(), 
                        signal_col=sig, 
                        model_type=model_type, 
                        use_volume=use_volume, 
                        train_months=train_months
                    )
                    
                    # 保存结果
                    if not result_df.empty:
                        result_df.to_csv(out_path, index=False)
                        print(f"✅ 结果已保存: {out_path}\n")

def main():
    data_dir = os.path.join(base, "data", "process")
    # 支持遍历目录下所有 CSV，或指定单个测试文件
    # paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    paths = ["/Users/ms/Desktop/ixic.csv"]
    paths = [data_dir+"/AAPL.csv"]
    
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            ticker = os.path.splitext(os.path.basename(p))[0]
            print(f"========= 开始处理标的: {ticker} =========")
            backtest_one_ticker(df, ticker)
        else:
            print(f"文件不存在: {p}")

if __name__ == "__main__":
    main()