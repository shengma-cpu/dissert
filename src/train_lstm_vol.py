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

# Keep the general idea: use the past 10 days of "+-" character sequences as features, roll 3 months for training, predict daily and accumulate returns
FEATURE_LOOKBACK = 10
TRAIN_LOOKBACK_MONTHS = 12
INITIAL_INVESTMENT = 100.0
MODEL_TYPE = ["gru"]
START_TIME = "2026-2-01"
# START_TIME = "2024-05-01"

# Three signal columns: oc=open->close, cc=prev_close->close, co=prev_close->open
# SIGNAL_COLS = ["oc", "cc", "co"]
SIGNAL_COLS = ["oc"]

base = os.path.dirname(os.path.dirname(__file__))

def build_feature_table(df: pd.DataFrame, signal_col: str = "oc"):
    """
    Not only build label sequences, but also prepare numeric features.
    """
    if signal_col not in df.columns:
        raise ValueError(f"Missing {signal_col} column")
    
    # 1. Preprocessing: convert labels into 0/1 numeric values
    df['return_label_df'] = df[signal_col].apply(lambda x: 1 if x > 0 else 0)
    
    # 2. Prepare feature matrix: [label, volume percentage]
    # Assuming df["v"] is already a percentage (e.g., 0.02 means 2%)
    label_data = df['return_label_df'].values
    volume_data = df['v'].fillna(0).values 

    feats = []
    for i in range(len(df)):
        if i >= FEATURE_LOOKBACK:
            # Extract past 10 days of [label, volume] combination
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

    # Build feature table (including volume)
    build_feature_table(df, signal_col=signal_col)

    combined_data = pd.merge(df, cash_df, on='date')
    # Target: stock (ixic > cash) or cash
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
        print(f"current_date==> {current_date} | Model: {model_type}")
        
        train_start = current_date - pd.DateOffset(months=TRAIN_LOOKBACK_MONTHS)
        train_end = current_date - pd.Timedelta(days=1)
        
        train_set = combined_data[(combined_data["date"] >= train_start) & (combined_data["date"] <= train_end)].dropna(subset=['feature_matrix'])
        test_set = combined_data[combined_data["date"] == current_date].dropna(subset=['feature_matrix'])

        if not test_set.empty and len(train_set) > 10:
            # --- Data preparation ---
            # Convert list of arrays into 3D numpy array: (samples, 10, 2)
            X_train = np.stack(train_set['feature_matrix'].values).astype(np.float32)
            X_test = np.stack(test_set['feature_matrix'].values).astype(np.float32)
            
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(train_set['target'])
            y_train_categorical = to_categorical(y_train_encoded, num_classes=2)

            # --- Model definition: BiGRU-Attention (dual feature input) ---
            # Here input dimension becomes (10, 2)
            inputs = Input(shape=(FEATURE_LOOKBACK, 2))
            
            gru_out = Bidirectional(GRU(50, return_sequences=True))(inputs)

            # Attention Module
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
            
            # Train (With volume added, model can learn more momentum features)
            model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, verbose=0)

            # Predict
            y_pred = model.predict(X_test, verbose=0)
            print("y_pred:", y_pred)
            decision = np.argmax(y_pred, axis=1)[0]
            pred_label = label_encoder.inverse_transform([decision])[0]
            
            # Update net value
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
    """Backtest separately for oc / cc / co signals and return combined results."""
    available = [col for col in SIGNAL_COLS if col in df.columns]
    if not available:
        raise ValueError(f"All signal columns missing {SIGNAL_COLS}, please check input")

    # Generate cash benchmark
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
            print(f"  Backtesting Signal: {sig}")
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
        print(f"Processing: {ticker}")
        backtest_one_ticker(df, ticker)

if __name__ == "__main__":
    main()
