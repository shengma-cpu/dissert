import yfinance as yf
import os

def collect_data():
    # 1. Define the stock ticker list
    tickers = ['AAPL', 'TSLA', 'XOM', 'JPM']

    # 2. Create the data storage directory (if it does not exist)
    output_dir = './../data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' has been created.")

    # 3. Download the data and save it
    for ticker in tickers:
        print(f"Fetching data for {ticker} (from 2000 to present)...")
        
        # Get data
        # start='2000-01-01' specifies the start date
        # yfinance will automatically handle stocks like TSLA that went public after 2000
        data = yf.download(ticker, start='2000-01-01')
        
        if not data.empty:
            # 4. Construct the file path and save as CSV
            file_path = os.path.join(output_dir, f"{ticker}.csv")
            data.to_csv(file_path)
            print(f"Successfully saved: {file_path}")
        else:
            print(f"Failed to fetch data for {ticker}, please check your network or code.")
    print("\nAll tasks completed!")

collect_data()