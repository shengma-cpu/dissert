import yfinance as yf
import os

def collect_data():
    # 1. 定义股票代码列表
    tickers = ['AAPL', 'TSLA', 'XOM', 'JPM']

    # 2. 创建存储目录 (如果不存在)
    output_dir = './../data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"目录 '{output_dir}' 已创建。")

    # 3. 下载数据并保存
    for ticker in tickers:
        print(f"正在获取 {ticker} 的数据 (2000年至今)...")
        
        # 获取数据
        # start='2000-01-01' 指定起始日期
        # yfinance 会自动处理 TSLA 这种在 2000 年后才上市的股票
        data = yf.download(ticker, start='2000-01-01')
        
        if not data.empty:
            # 4. 构造文件路径并保存为 CSV
            file_path = os.path.join(output_dir, f"{ticker}.csv")
            data.to_csv(file_path)
            print(f"成功保存: {file_path}")
        else:
            print(f"未能获取 {ticker} 的数据，请检查网络或代码。")
    print("\n所有任务已完成！")

collect_data()