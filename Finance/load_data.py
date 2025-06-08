import yfinance as yf
import pandas as pd
import numpy as np
import torch
import os
import time
import logging
from tqdm import tqdm


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sequence_length = 256
start_date = "2017-01-01"
end_date = "2024-12-31"
output_dir = "financial_data/sequences"
cache_dir = "financial_data/cache"
cache_file = os.path.join(cache_dir, "sp500_data.parquet")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(cache_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sequences_256.pt")
standardize = False
target_std = 1

def safe_yf_download(tickers, max_retries=3, delay=5):
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                group_by='ticker',
                threads=True
            )
            if not data.empty:
                return data
            logger.warning(f"attempt {attempt+1}: empty")
        except Exception as e:
            logger.warning(f"attempt {attempt+1} failed: {str(e)[:100]}...")
        
        if attempt < max_retries - 1:
            sleep_time = delay * (attempt + 1) 
            logger.info(f"wait {sleep_time} ...")
            time.sleep(sleep_time)
    
    raise Exception(f"failed, tried {max_retries}")

try:
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500['Symbol'].str.replace('.', '-', regex=False).tolist()
    logger.info(f"Loaded {len(tickers)} S&P 500 tickers")
except Exception as e:
    logger.error(f"Failed to fetch S&P 500 tickers: {e}")
    exit(1)


BATCH_SIZE = 50
total_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE


if os.path.exists(cache_file):
    try:
        data = pd.read_parquet(cache_file)
        logger.info(f"Loaded cached data from {cache_file}")
    except Exception as e:
        logger.warning(f"Invalid cache file {cache_file}: {e}, redownloading")
        data = None
else:
    data = None

if data is None:
    logger.info("Downloading stock data in batches...")
    all_data = []
    failed_batches = []
    
    for batch_num in tqdm(range(total_batches), desc="batch"):
        batch_tickers = tickers[batch_num*BATCH_SIZE : (batch_num+1)*BATCH_SIZE]
        try:
            batch_data = safe_yf_download(batch_tickers)
            all_data.append(batch_data)
            logger.info(f"batch {batch_num+1}/{total_batches} completed")
        except Exception as e:
            failed_batches.extend(batch_tickers)
            logger.error(f"batch {batch_num+1} failed: {e}")
        
        # 批次间延迟
        time.sleep(2)
    
    if all_data:
        data = pd.concat(all_data, axis=1)
        data.to_parquet(cache_file)
        logger.info(f"Saved data to {cache_file}")
    else:
        logger.error("所有批次下载均失败")
        exit(1)

    if failed_batches:
        logger.warning(f"{len(failed_batches)} failed")
        with open("failed_downloads.txt", "w") as f:
            f.write("\n".join(failed_batches))


sequences = []
start_dates = []
market_caps = []
failed_tickers = []

for ticker in tqdm(tickers, desc=""):
    try:

        df = data[ticker]['Close'] if ticker in data.columns.levels[0] else pd.Series()
        if df.empty or len(df) < sequence_length:
            failed_tickers.append(ticker)
            logger.warning(f"Skipping {ticker}: insufficient data ({len(df)} days)")
            continue
        

        market_cap = None
        for attempt in range(3):
            try:
                ticker_obj = yf.Ticker(ticker)
                market_cap = ticker_obj.info.get('marketCap', None)
                if market_cap is not None:
                    break
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed to get market cap for {ticker}: {e}")
                time.sleep(2 * (attempt + 1))
        
        if market_cap is None:
            failed_tickers.append(ticker)
            logger.warning(f"Using default market cap for {ticker}")
            market_cap = 1e9 
        

        df = df.dropna()
        returns = np.log(df / df.shift(1)).dropna()
        if len(returns) < sequence_length:
            failed_tickers.append(ticker)
            logger.warning(f"Skipping {ticker}: insufficient returns ({len(returns)})")
            continue
        

        df.index = pd.to_datetime(df.index)
        monthly_starts = df.groupby([df.index.year, df.index.month]).head(1).index
        for start_date in monthly_starts:
            try:
                start_idx = returns.index.get_loc(start_date)
                if start_idx + sequence_length <= len(returns):
                    seq = returns.iloc[start_idx:start_idx + sequence_length].values
                    if len(seq) == sequence_length and not np.any(np.isnan(seq)):
                        sequences.append(seq.flatten())
                        date_vec = [
                            (start_date.year - 2017) / 8.0,
                            (start_date.month - 1) / 12.0,
                            (start_date.day - 1) / 31.0
                        ]
                        start_dates.append(date_vec)
                        market_caps.append(market_cap)
            except Exception as e:
                logger.warning(f"Skipping sequence for {ticker} at {start_date}: {e}")
        logger.info(f"Processed {ticker}: {len(monthly_starts)} sequences")
    except Exception as e:
        failed_tickers.append(ticker)
        logger.error(f"Failed {ticker}: {e}")


if failed_tickers:
    logger.warning(f"Failed tickers ({len(failed_tickers)}): {failed_tickers}")


if len(sequences) == 0:
    logger.error("No sequences generated")
    raise ValueError("No sequences generated")
sequences = np.array(sequences, dtype=np.float32)
start_dates = np.array(start_dates, dtype=np.float32)
market_caps = np.array(market_caps, dtype=np.float32)


if len(market_caps) > 0:
    market_caps = (market_caps - np.min(market_caps)) / (np.max(market_caps) - np.min(market_caps) + 1e-8)
    market_caps = market_caps.reshape(-1, 1)


if standardize:
    mean = np.mean(sequences, axis=1, keepdims=True)
    std = np.std(sequences, axis=1, keepdims=True)
    sequences = (sequences - mean) / (std + 1e-8) * target_std


sequences = torch.tensor(sequences)
start_dates = torch.tensor(start_dates)
market_caps = torch.tensor(market_caps)


if sequences.shape[0] == 0:
    logger.error("No sequences generated")
    raise ValueError("No sequences generated")
if sequences.dim() != 2 or sequences.shape[1] != sequence_length:
    logger.error(f"Expected shape (N, {sequence_length}), got {sequences.shape}")
    raise ValueError(f"Expected shape (N, {sequence_length}), got {sequences.shape}")
if start_dates.shape != (sequences.shape[0], 3):
    logger.error(f"Expected start_dates shape ({sequences.shape[0]}, 3), got {start_dates.shape}")
    raise ValueError(f"Expected start_dates shape ({sequences.shape[0]}, 3), got {start_dates.shape}")
if market_caps.shape != (sequences.shape[0], 1):
    logger.error(f"Expected market_caps shape ({sequences.shape[0]}, 1), got {market_caps.shape}")
    raise ValueError(f"Expected market_caps shape ({sequences.shape[0]}, 1), got {market_caps.shape}")
logger.info(f"Sequences shape: {sequences.shape}")
logger.info(f"Start dates shape: {start_dates.shape}")
logger.info(f"Market caps shape: {market_caps.shape}")
logger.info(f"Sequences mean: {sequences.mean().item():.6f}")
logger.info(f"Sequences std: {sequences.std().item():.6f}")

torch.save({
    "sequences": sequences,
    "start_dates": start_dates,
    "market_caps": market_caps
}, output_file)
logger.info(f"Saved {sequences.shape[0]} sequences to {output_file}")