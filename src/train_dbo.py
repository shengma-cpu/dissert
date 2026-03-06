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
# Module 1: Experiment Configurations
# ==========================================
WRITE_RESULT_FOURCE = True
FEATURE_LOOKBACK = 10
INITIAL_INVESTMENT = 100.0
START_TIME = "2026-01-01"
# START_TIME = "2024-05-01"

# The four dimensions to be compared in the thesis:
# SIGNAL_COLS = ["oc", "cc", "co"]
# MODEL_TYPES = ["lstm", "attention"]
# USE_VOLUME_LIST = [False, True]
# TRAIN_MONTHS_LIST = [3, 6, 12, 36]

SIGNAL_COLS = ["oc"]
MODEL_TYPES = ["attention"]
USE_VOLUME_LIST = [True]
TRAIN_MONTHS_LIST = [3, 6, 12]

base = os.path.dirname(os.path.dirname(__file__))

# ==========================================
# Module 1: Feature Engineering
# ==========================================
def build_feature_table(df: pd.DataFrame, signal_col: str, use_volume: bool):
    """
    Uniformly build the feature matrix:
    - use_volume=False: Return a label sequence of shape (10, 1) (containing only 0/1)
    - use_volume=True:  Return a composite sequence of shape (10, 2) (label + volume percentage)
    """
    if signal_col not in df.columns:
        raise ValueError(f"Signal column missing in data: {signal_col}")
    if use_volume and "v" not in df.columns:
        print("Warning: Volume feature enabled but 'v' column missing, auto-filling with 0")
        df["v"] = 0.0
    
    # 1. Convert fluctuations into 1/0 numerical classification
    df['return_label_df'] = df[signal_col].apply(lambda x: 1 if x > 0 else 0)
    
    label_data = df['return_label_df'].values
    volume_data = df['v'].fillna(0).values if use_volume else None

    feats = []
    for i in range(len(df)):
        if i >= FEATURE_LOOKBACK:
            if use_volume:
                # Includes volume, shape: (10, 2)
                seq = np.column_stack((
                    label_data[i - FEATURE_LOOKBACK : i],
                    volume_data[i - FEATURE_LOOKBACK : i]
                ))
            else:
                # No volume, shape: (10, 1)
                seq = np.expand_dims(label_data[i - FEATURE_LOOKBACK : i], axis=-1)
            feats.append(seq)
        else:
            feats.append(None)
    
    df["feature_matrix"] = feats
    return df

# ==========================================
# Module 2: Model Factory
# ==========================================
def build_model(model_type: str, input_shape: tuple, units: int = 50, dropout: float = 0.2, lr: float = 0.001):
    """
    Dynamically generate network structure based on configuration
    """
    if model_type == "lstm":
        # Traditional LSTM model
        model = Sequential()
        model.add(LSTM(units, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(2, activation='softmax'))
    elif model_type == "gru":
        model = Sequential()
        model.add(GRU(units, input_shape=input_shape))
        model.add(Dropout(dropout))
        model.add(Dense(2, activation='softmax'))
    elif model_type == "attention":
        # Improved BiGRU-Attention model
        inputs = Input(shape=input_shape)
        gru_out = Bidirectional(GRU(units, return_sequences=True))(inputs)
        gru_out = Dropout(dropout)(gru_out)

        # Attention mechanism to calculate weights
        attention = Dense(1, activation='tanh')(gru_out) 
        attention = Flatten()(attention)
        attention = Dense(input_shape[0], activation='softmax')(attention)  # input_shape[0] is LOOKBACK
        attention = RepeatVector(units * 2)(attention)  # Dimension after BiGRU concatenation is units * 2
        attention = Permute([2, 1])(attention)

        # Weighted sum
        sent_representation = Multiply()([gru_out, attention])
        sent_representation = Lambda(lambda x: K.sum(x, axis=1))(sent_representation)

        probabilities = Dense(2, activation='softmax')(sent_representation)
        model = Model(inputs=inputs, outputs=probabilities)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    from tensorflow.keras.optimizers import Adam
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# Module 3: Backtesting Engine
# ==========================================
def _backtest_single_experiment(df: pd.DataFrame, cash_df: pd.DataFrame, 
                                signal_col: str, model_type: str, 
                                use_volume: bool, train_months: int) -> pd.DataFrame:
        
    df['date'] = pd.to_datetime(df['Date'])
    cash_df['date'] = pd.to_datetime(cash_df['Date'])

    # Extract features
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
        print(f"current_date==> {current_date} | [Signal: {signal_col}] | [Model: {model_type}] | [Vol: {use_volume}] | [Window: {train_months}m]")
        train_start = current_date - pd.DateOffset(months=train_months)
        train_end = current_date - pd.Timedelta(days=1)
        
        train_set = combined_data[(combined_data["date"] >= train_start) & (combined_data["date"] <= train_end)].dropna(subset=['feature_matrix'])
        test_set = combined_data[combined_data["date"] == current_date].dropna(subset=['feature_matrix'])

        if not test_set.empty:
            if len(train_set) <= 10:
                print(f"=====> Not enough samples in training set: {len(train_set)}, skipping")
                current_date += pd.Timedelta(days=1)
                continue
            # 1. Prepare features and labels in Numpy format
            X_train = np.stack(train_set['feature_matrix'].values).astype(np.float32)
            X_test = np.stack(test_set['feature_matrix'].values).astype(np.float32)
            
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(train_set['target'])
            y_train_categorical = to_categorical(y_train_encoded, num_classes=2)

            # 2. Prepare for DBO optimization and construct model
            input_shape = (FEATURE_LOOKBACK, 2 if use_volume else 1)
            
            # Split validation set for DBO fitness evaluation
            val_split = int(len(X_train) * 0.8)
            X_dbo_train, X_dbo_val = X_train[:val_split], X_train[val_split:]
            y_dbo_train, y_dbo_val = y_train_categorical[:val_split], y_train_categorical[val_split:]
            
            def dbo_obj(params):
                u = int(np.round(params[0]))
                d = float(params[1])
                l = float(params[2])
                try:
                    m = build_model(model_type, input_shape, units=u, dropout=d, lr=l)
                    m.fit(X_dbo_train, y_dbo_train, epochs=3, batch_size=32, verbose=0)
                    loss, _ = m.evaluate(X_dbo_val, y_dbo_val, verbose=0)
                    del m
                    K.clear_session()
                    return loss  # Minimize loss
                except Exception as e:
                    return 999.0
            
            # bounds: units=[10, 100], dropout=[0.1, 0.5], lr=[0.0001, 0.01]
            lb = [10, 0.1, 0.0001]
            ub = [100, 0.5, 0.01]
            
            import sys
            import os
            sys.path.append(os.path.dirname(__file__))
            from dbo import DBO
            
            # Use small parameters to avoid running too long, you can adjust pop_size and max_iter for better parameters but more time
            dbo = DBO(dbo_obj, lb, ub, dim=3, pop_size=3, max_iter=2)
            best_params, best_loss = dbo.optimize()
            
            best_units = int(np.round(best_params[0]))
            best_dropout = float(best_params[1])
            best_lr = float(best_params[2])
            
            # Construct and train final model with best hyperparameters optimized by optimizer
            model = build_model(model_type, input_shape, units=best_units, dropout=best_dropout, lr=best_lr)
            model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, verbose=0)

            # 3. Perform prediction
            # y_pred = model.predict(X_test, verbose=0)
            y_pred = model(X_test, training=False).numpy()
            decision = np.argmax(y_pred, axis=1)[0]
            pred_label = label_encoder.inverse_transform([decision])[0]
            
            # 4. Statistics and calculate daily return
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
            
            # Print debug information
            print(
                  f"Net Value: {current_value:>8.2f} | acc: {total_correct/total_pred:.2%},  Prob: [cash:{y_pred[0][0]:.3f}, stock:{y_pred[0][1]:.5f}] "
                  f"[Pred: {pred_label:<5} | Real: {real_label:<5}] "
                  f"(DBO params: units={best_units}, drop={best_dropout:.2f}, lr={best_lr:.4f})"
                  )
            
            records.append([date, current_value, pred_label, real_label, y_pred[0].tolist()])

            # Clear memory to prevent memory leaks
            del model
            K.clear_session()
            gc.collect()

        current_date += pd.Timedelta(days=1)

    if total_pred > 0:
        overall_acc = total_correct / total_pred
        print(f"=== Experiment Ended === Overall Acc: {overall_acc:.2%} ({total_correct}/{total_pred}) Final Value: {current_value:.2f}\n")

    return pd.DataFrame(records, columns=["date", "value", "predicted", "real", "y_pred"])

# ==========================================
# Module 4: Experiment Scheduler
# ==========================================
def backtest_one_ticker(df: pd.DataFrame, ticker: str):
    available_signals = [col for col in SIGNAL_COLS if col in df.columns]
    if not available_signals:
        raise ValueError(f"All configured signal columns missing {SIGNAL_COLS}, please check input")

    # Generate benchmark risk-free cash return (0%)
    date_col = "date" if "date" in df.columns else "Date"
    cash_df = pd.DataFrame({
        "Date":   pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d"),
        "price":  1,
        "return": 1e-6,
    })
    out_dir = os.path.join(base, "data", "result")
    os.makedirs(out_dir, exist_ok=True)

    # Nested loop: Fully automatically run all cross-comparison experiments
    for sig in available_signals:
        for model_type in MODEL_TYPES:
            for use_volume in USE_VOLUME_LIST:
                for train_months in TRAIN_MONTHS_LIST:

                    # Generate clear filename: Ticker_Signal_Model_Volume_Window.csv
                    vol_tag = "vol" if use_volume else "novol"
                    out_filename = f"{ticker}_{sig}_{model_type}_{vol_tag}_{train_months}m.csv"
                    out_path = os.path.join(out_dir, out_filename)
                    
                    # If already run, skip (easy to resume after interruption)
                    if os.path.exists(out_path):
                        if WRITE_RESULT_FOURCE:
                            os.remove(out_path)
                        else:
                            print(f"File already exists, skipping: {out_filename}")
                            continue
                    
                    # Run single backtest
                    result_df = _backtest_single_experiment(
                        df.copy(), cash_df.copy(), 
                        signal_col=sig, 
                        model_type=model_type, 
                        use_volume=use_volume, 
                        train_months=train_months
                    )
                    
                    # Save results
                    if not result_df.empty:
                        result_df.to_csv(out_path, index=False)
                        print(f"✅ Result saved: {out_path}\n")

def main():
    data_dir = os.path.join(base, "data", "process")
    # Supports iterating through all CSVs in the dir, or specifying a single test file
    # paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    paths = ["/Users/ms/Desktop/ixic.csv"]
    paths = [data_dir+"/AAPL.csv"]
    
    for p in paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            ticker = os.path.splitext(os.path.basename(p))[0]
            print(f"========= Started processing ticker: {ticker} =========")
            backtest_one_ticker(df, ticker)
        else:
            print(f"File does not exist: {p}")

if __name__ == "__main__":
    main()