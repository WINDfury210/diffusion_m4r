import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


output_dir = "financial_data/figures"
os.makedirs(output_dir, exist_ok=True)
input_file = "financial_data/sequences/sequences_256.pt"
output_file = os.path.join(output_dir, "sample_sequences.png")


try:
    data = torch.load(input_file)
    sequences = data["sequences"].numpy()
    start_dates = data["start_dates"].numpy()
    logger.info(f"Loaded data: {sequences.shape} sequences, {start_dates.shape} start dates")
except Exception as e:
    logger.error(f"Failed to load data from {input_file}: {e}")
    raise

num_samples = min(5, len(sequences)) 
sample_indices = np.random.choice(len(sequences), num_samples, replace=False)
sample_sequences = sequences[sample_indices]
sample_dates = start_dates[sample_indices]


def denormalize_date(normalized_date):
    year = int(normalized_date[0] * 8 + 2017)
    month = int(normalized_date[1] * 12 + 1)
    day = int(normalized_date[2] * 31 + 1)
    
    try:
        return datetime(year, month, day).strftime("%Y-%m-%d")
    except ValueError:
        
        if month == 2 and day > 28:
            day = 28 if year % 4 != 0 else 29
        elif day == 31 and month in [4, 6, 9, 11]:
            day = 30
        return datetime(year, month, day).strftime("%Y-%m-%d")

means = np.mean(sample_sequences, axis=1)
stds = np.std(sample_sequences, axis=1)

colors = plt.cm.Blues(np.linspace(0.3, 0.9, num_samples))  


plt.figure(figsize=(12, 8))
for i in range(num_samples):
    date_str = denormalize_date(sample_dates[i])
    mean_val = means[i]
    std_val = stds[i]
    label = f'Sequence {i+1} (Start: {date_str}, Mean: {mean_val:.4f}, Std: {std_val:.4f})'
    plt.plot(range(256), sample_sequences[i], label=label, color=colors[i], linewidth=1.5, alpha=0.8)

plt.title('Sample Daily Log-Return Sequences (256 Days)', fontsize=14, pad=15)
plt.xlabel('Day', fontsize=12)
plt.ylabel('Log-Return', fontsize=12)
plt.legend(loc='upper right', fontsize=10, framealpha=0.8)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()


plt.savefig(output_file, dpi=300)
logger.info(f"Saved sample sequences plot to {output_file}")
plt.close()


logger.info("Statistical Summary of Sampled Sequences:")
for i in range(num_samples):
    date_str = denormalize_date(sample_dates[i])
    logger.info(f"Sequence {i+1} (Start: {date_str}): Mean = {means[i]:.4f}, Std = {stds[i]:.4f}")