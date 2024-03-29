# test_signal_preprocessor.py
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.signal_preprocessor import SignalPreprocessor

import pandas as pd
import numpy as np

def test_butter_lowpass_filter():
    sp = SignalPreprocessor(window_length_ms=100, sampling_rate_hz=10000, cutoff_hz=150)
    # Create a test signal with a mix of low and high frequencies
    t = np.linspace(0, 1, 10000, False)
    data = np.sin(2*np.pi*5*t) + np.sin(2*np.pi*500*t)
    filtered_data = sp.butter_lowpass_filter(data)

    # Assert that the high frequency component is reduced
    assert np.std(filtered_data) < np.std(data), "Filter should reduce the variance of a high-frequency signal"

def test_segment_into_windows():
    sp = SignalPreprocessor(window_length_ms=100, sampling_rate_hz=10000, cutoff_hz=150)
    df = pd.DataFrame({'time': np.linspace(0, 1, 10000), 'data': np.random.rand(10000)})
    windows = sp.segment_into_windows(df)
    assert len(windows) == 10, "Should create 10 windows for 10000 data points with 100ms window length"

def test_preprocess():
    sp = SignalPreprocessor(window_length_ms=100, sampling_rate_hz=10000, cutoff_hz=150)
    df = pd.DataFrame({
        'time': np.linspace(0, 0.1, 1000),
        'data': np.sin(2*np.pi*50*np.linspace(0, 0.1, 1000)) + np.random.normal(0, 1, 1000)
    })
    preprocessed_windows = sp.preprocess(df)
    assert len(preprocessed_windows) == 1, "Preprocess should segment data into 1 window for 1000 data points"


# To run these tests, use the command: pytest test_signal_preprocessor.py
