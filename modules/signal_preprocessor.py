import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, lfilter

class SignalPreprocessor:
    """
    Eine Klasse zur Vorverarbeitung von Zeitsignaldaten für maschinelles Lernen.
    
    Attribute:
        window_length_ms (int): Länge des Fensters in Millisekunden für die Segmentierung.
        sampling_rate_hz (int): Abtastrate in Hertz.
        cutoff_hz (int): Grenzfrequenz für den Tiefpassfilter.
        window_size_points (int): Anzahl der Messpunkte pro Fenster.
        scaler (MinMaxScaler): Instanz des Scalers zur Normalisierung der Daten.
    """
    
    def __init__(self, window_length_ms: int = 100, sampling_rate_hz: int = 10000, cutoff_hz: int = 150):
        """
        Initialisiert den SignalPreprocessor mit den gegebenen Parametern.

        Args:
            window_length_ms (int): Länge des Fensters in Millisekunden.
            sampling_rate_hz (int): Abtastrate in Hz.
            cutoff_hz (int): Grenzfrequenz für den Tiefpassfilter in Hz.
        """
        self.window_length_ms = window_length_ms
        self.sampling_rate_hz = sampling_rate_hz
        self.cutoff_hz = cutoff_hz
        self.window_size_points = int((sampling_rate_hz / 1000) * window_length_ms)
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def butter_lowpass_filter(self, data, order=5):
        """
        Wendet einen Butterworth-Tiefpassfilter auf die Daten an.

        Args:
            data (array_like): Die zu filternden Daten.
            order (int): Die Ordnung des Filters.

        Returns:
            array_like: Die gefilterten Daten.
        """
        self.cutoff_hz = 40
        nyq = 0.5 * self.sampling_rate_hz
        normal_cutoff = self.cutoff_hz / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y


    def segment_into_windows(self, data: pd.DataFrame) -> [pd.DataFrame]:
        """
        Segmentiert die Daten in Fenster basierend auf der Fensterlänge.

        Args:
            data (pd.DataFrame): Die zu segmentierenden Daten.

        Returns:
            List[pd.DataFrame]: Liste von DataFrames, jedes repräsentiert ein Fenster.
        """
        windows = []
        total_samples = len(data)
        for start_index in range(0, total_samples, self.window_size_points):
            end_index = min(start_index + self.window_size_points, total_samples)
            window = data.iloc[start_index:end_index]
            if not window.empty:
                windows.append(window)
        return windows

    def preprocess(self, data: pd.DataFrame) -> [pd.DataFrame]:
        """
        Verarbeitet die gegebenen Daten durch Filterung und Normalisierung.

        Args:
            data (pd.DataFrame): Die zu verarbeitenden Daten.

        Returns:
            List[pd.DataFrame]: Liste von DataFrames, jedes repräsentiert ein vorverarbeitetes Fenster.
        """
        if 'data' not in data.columns or 'time' not in data.columns:
            raise ValueError("Data for preprocessing must include 'data' and 'time' columns")

        preprocessed_windows = []
        windows = self.segment_into_windows(data)
    
        for window in windows:
            filtered = self.butter_lowpass_filter(window['data'].values)
            normalized = self.scaler.fit_transform(filtered.reshape(-1, 1)).flatten()
            preprocessed_window = pd.DataFrame({
                'time': np.round(window['time'].reset_index(drop=True), 4),
                'data_filtered_normalized': normalized
            })
            preprocessed_windows.append(preprocessed_window)
        return preprocessed_windows

    def save_preprocessed_data(self, preprocessed_windows, experiment_name, measurement_name):
        """
        Speichert die vorverarbeiteten Daten in einer CSV-Datei.

        Args:
            preprocessed_windows (List[pd.DataFrame]): Die Liste der vorverarbeiteten Fenster.
            experiment_name (str): Der Name des Experiments.
            measurement_name (str): Der Name der Messung.
        """
        save_directory = '.data/preprocessed'
        os.makedirs(save_directory, exist_ok=True)
        save_path = f'{save_directory}/{experiment_name}_{measurement_name}.csv'
        pd.concat(preprocessed_windows, ignore_index=True).to_csv(save_path, index=False)
        print(f"Preprocessed data saved to {save_path}")