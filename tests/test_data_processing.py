# tests/test_data_processing.py
import sys
import os

# Add the parent directory to sys.path before importing anything from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
from src.data_processing import calculate_rfm


def test_calculate_rfm_columns():
    data = pd.DataFrame({
        "CustomerId": [1, 1, 2],
        "TransactionId": ["T1", "T2", "T3"],
        "TransactionStartTime": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Amount": [100, 200, 150]
    })
    snapshot_date = "2023-01-04"
    rfm = calculate_rfm(data, snapshot_date)
    expected_columns = ["CustomerId", "Recency", "Frequency", "Monetary"]
    assert list(rfm.columns) == expected_columns


def test_calculate_rfm_values():
    data = pd.DataFrame({
        "CustomerId": [1, 1],
        "TransactionId": ["T1", "T2"],
        "TransactionStartTime": ["2023-01-01", "2023-01-02"],
        "Amount": [100, 200]
    })
    snapshot_date = "2023-01-04"
    rfm = calculate_rfm(data, snapshot_date)
    assert rfm.loc[rfm["CustomerId"] == 1, "Recency"].iloc[0] == 2
    assert rfm.loc[rfm["CustomerId"] == 1, "Frequency"].iloc[0] == 2
    assert rfm.loc[rfm["CustomerId"] == 1, "Monetary"].iloc[0] == 300
