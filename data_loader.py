"""
Sample data loading and exploration script for CICIDS2017 dataset.
Use this script to understand the dataset structure and create sample data.
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_dataset(output_path='data/cicids2017_sample.csv', num_samples=1000):
    """
    Create a sample CICIDS2017-like dataset for demonstration.

    In production, download the real dataset from:
    https://www.kaggle.com/datasets/cicdataset/cicids2017

    Args:
        output_path: Where to save the sample CSV
        num_samples: Number of samples to generate
    """
    logger.info(f"Creating sample dataset with {num_samples} samples...")

    np.random.seed(42)

    # Generate synthetic data with 84 features similar to CICIDS2017
    features = {
        'Flow Duration': np.random.randint(1000, 1000000, num_samples),
        'Total Fwd Packets': np.random.randint(1, 500, num_samples),
        'Total Backward Packets': np.random.randint(1, 500, num_samples),
        'Total Length of Fwd Packets': np.random.randint(100, 100000, num_samples),
        'Total Length of Bwd Packets': np.random.randint(100, 100000, num_samples),
        'Fwd Packet Length Max': np.random.randint(1, 65535, num_samples),
        'Fwd Packet Length Min': np.random.randint(1, 1000, num_samples),
        'Fwd Packet Length Mean': np.random.randint(1, 5000, num_samples),
        'Fwd Packet Length Std': np.random.randint(0, 1000, num_samples),
        'Bwd Packet Length Max': np.random.randint(1, 65535, num_samples),
        'Bwd Packet Length Min': np.random.randint(1, 1000, num_samples),
        'Bwd Packet Length Mean': np.random.randint(1, 5000, num_samples),
        'Bwd Packet Length Std': np.random.randint(0, 1000, num_samples),
        'Flow Bytes/s': np.random.randint(0, 10000000, num_samples),
        'Flow Packets/s': np.random.randint(0, 100000, num_samples),
        'Flow IAT Mean': np.random.randint(0, 1000000, num_samples),
        'Flow IAT Std': np.random.randint(0, 1000000, num_samples),
        'Flow IAT Max': np.random.randint(0, 10000000, num_samples),
        'Flow IAT Min': np.random.randint(0, 10000000, num_samples),
        'Fwd IAT Total': np.random.randint(0, 1000000000, num_samples),
        'Fwd IAT Mean': np.random.randint(0, 1000000, num_samples),
        'Fwd IAT Std': np.random.randint(0, 1000000, num_samples),
        'Fwd IAT Max': np.random.randint(0, 10000000, num_samples),
        'Fwd IAT Min': np.random.randint(0, 10000000, num_samples),
        'Bwd IAT Total': np.random.randint(0, 1000000000, num_samples),
        'Bwd IAT Mean': np.random.randint(0, 1000000, num_samples),
        'Bwd IAT Std': np.random.randint(0, 1000000, num_samples),
        'Bwd IAT Max': np.random.randint(0, 10000000, num_samples),
        'Bwd IAT Min': np.random.randint(0, 10000000, num_samples),
        'Fwd PSH Flags': np.random.randint(0, 100, num_samples),
        'Bwd PSH Flags': np.random.randint(0, 100, num_samples),
        'Fwd URG Flags': np.random.randint(0, 100, num_samples),
        'Bwd URG Flags': np.random.randint(0, 100, num_samples),
        'Fwd Header Length': np.random.randint(0, 1000, num_samples),
        'Bwd Header Length': np.random.randint(0, 1000, num_samples),
        'Fwd Packets/s': np.random.randint(0, 100000, num_samples),
        'Bwd Packets/s': np.random.randint(0, 100000, num_samples),
        'Min Packet Length': np.random.randint(0, 10000, num_samples),
        'Max Packet Length': np.random.randint(0, 65535, num_samples),
        'Packet Length Mean': np.random.randint(0, 10000, num_samples),
        'Packet Length Std': np.random.randint(0, 5000, num_samples),
        'Packet Length Variance': np.random.randint(0, 5000, num_samples),
        'FIN Flag Count': np.random.randint(0, 10, num_samples),
        'SYN Flag Count': np.random.randint(0, 10, num_samples),
        'RST Flag Count': np.random.randint(0, 10, num_samples),
        'PSH Flag Count': np.random.randint(0, 10, num_samples),
        'ACK Flag Count': np.random.randint(0, 500, num_samples),
        'URG Flag Count': np.random.randint(0, 10, num_samples),
        'CWE Flag Count': np.random.randint(0, 10, num_samples),
        'ECE Flag Count': np.random.randint(0, 10, num_samples),
        'Down/Up Ratio': np.random.uniform(0, 10, num_samples),
        'Average Packet Size': np.random.randint(0, 10000, num_samples),
        'Avg Fwd Segment Size': np.random.randint(0, 10000, num_samples),
        'Avg Bwd Segment Size': np.random.randint(0, 10000, num_samples),
        'Fwd Header Length': np.random.randint(0, 1000, num_samples),
        'Fwd Avg Bytes/Bulk': np.random.randint(0, 10000, num_samples),
        'Fwd Avg Packets/Bulk': np.random.randint(0, 100, num_samples),
        'Fwd Avg Bulk Rate': np.random.randint(0, 10000, num_samples),
        'Bwd Avg Bytes/Bulk': np.random.randint(0, 10000, num_samples),
        'Bwd Avg Packets/Bulk': np.random.randint(0, 100, num_samples),
        'Bwd Avg Bulk Rate': np.random.randint(0, 10000, num_samples),
        'Subflow Fwd Packets': np.random.randint(0, 100, num_samples),
        'Subflow Fwd Bytes': np.random.randint(0, 100000, num_samples),
        'Subflow Bwd Packets': np.random.randint(0, 100, num_samples),
        'Subflow Bwd Bytes': np.random.randint(0, 100000, num_samples),
        'Init Fwd Win Bytes': np.random.randint(0, 65535, num_samples),
        'Init Bwd Win Bytes': np.random.randint(0, 65535, num_samples),
        'Fwd Act Data Pkts': np.random.randint(0, 500, num_samples),
        'Fwd Seg Size Min': np.random.randint(0, 10000, num_samples),
        'Active Mean': np.random.randint(0, 1000000, num_samples),
        'Active Std': np.random.randint(0, 1000000, num_samples),
        'Active Max': np.random.randint(0, 10000000, num_samples),
        'Active Min': np.random.randint(0, 10000000, num_samples),
        'Idle Mean': np.random.randint(0, 1000000, num_samples),
        'Idle Std': np.random.randint(0, 1000000, num_samples),
        'Idle Max': np.random.randint(0, 10000000, num_samples),
        'Idle Min': np.random.randint(0, 10000000, num_samples),
    }

    df = pd.DataFrame(features)

    # Add label column - 80% normal, 20% attack
    labels = np.random.choice(['BENIGN', 'Attack'], num_samples, p=[0.8, 0.2])
    df['Label'] = labels

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Sample dataset created: {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['Label'].value_counts()}")

    return df


def load_real_cicids2017(filepath):
    """
    Load real CICIDS2017 dataset from Kaggle.

    Dataset download: https://www.kaggle.com/datasets/cicdataset/cicids2017

    Expected columns: 84 network flow features + 'Label' column
    """
    logger.info(f"Loading CICIDS2017 dataset from {filepath}...")

    df = pd.read_csv(filepath)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Label distribution:\n{df['Label'].value_counts()}")

    return df


def explore_dataset(df):
    """Explore dataset statistics and characteristics."""
    logger.info("\n=== Dataset Exploration ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Numeric statistics:\n{df.describe()}")


if __name__ == '__main__':
    # Create sample dataset for demonstration
    sample_df = create_sample_dataset(num_samples=1000)
    explore_dataset(sample_df)

    logger.info("\nDataset created successfully!")
    logger.info("Ready for training with: python -m src.train")
