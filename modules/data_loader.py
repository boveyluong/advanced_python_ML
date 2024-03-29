import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Any, Dict, Union
from pathlib import Path, PureWindowsPath  # Importiere pathlib

# Konfiguriere das Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    """
    DataLoader Klasse zum Laden und Vereinigen von Datensätzen für verschiedene Experimente.
    """
    def __init__(self, config_path: str):
        """
        Initialisiert den DataLoader mit einer Konfigurationsdatei.
        """
        try:
            with open(config_path, 'r') as config_file:
                self.config = json.load(config_file)
            self.validate_config()
            logging.info("Konfiguration erfolgreich geladen und validiert.")
        except FileNotFoundError:
            logging.error(f"Konfigurationsdatei unter {config_path} nicht gefunden.")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Ungültiges JSON-Format in der Konfiguration: {e}")
            raise
    
    def validate_config(self):
        """
        Überprüft die geladene Konfiguration auf notwendige Felder und Formate.
        """
        required_fields = ['experiments']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Fehlendes erforderliches Feld in der Konfiguration: {field}")

    def load_experiment_data(self, experiment_name: Union[str, None] = None) -> pd.DataFrame:
        """
        Lädt Datensätze für ein gegebenes Experiment oder alle Experimente.
        """
        if experiment_name and experiment_name not in self.config['experiments']:
            raise ValueError(f"Experiment {experiment_name} nicht in der Konfiguration gefunden.")

        experiment_data = []
        experiments_to_load = self.config['experiments'] if not experiment_name else {experiment_name: self.config['experiments'][experiment_name]}

        for experiment, files in experiments_to_load.items():
            for file_info in files:
                # Konvertiere Windows-Pfad zu einem systemunabhängigen Path-Objekt
                file_path = Path(PureWindowsPath(file_info['path']))
                file_type = file_info['type']
                try:
                    data = self.load_file(file_path, file_type, experiment)
                    file_name = file_path.name.split('.')[0]
                    
                    # Füge 'experiment' und 'measurement' Spalten hinzu
                    data['experiment'] = experiment
                    data['measurement'] = file_name

                    # Behalte 'time', 'experiment', und 'measurement' als reguläre Spalten
                    experiment_data.append(data)
                except Exception as e:
                    logging.error(f"Fehler beim Laden der Datei {file_path}: {e}")
                    continue
        
        if not experiment_data:
            logging.warning(f"Keine Daten für Experiment(e) geladen.")
            return pd.DataFrame()
        # Verbinde Daten wie sie sind, ohne den Index zu ändern
        data = pd.concat(experiment_data, ignore_index=True)
        # Erzeuge einen Zeitindex in Sekunden
        actual_number_of_samples = len(data)
        sampling_rate = 10e3  # 10 kHz
        time_in_seconds = np.arange(1, actual_number_of_samples + 1) / sampling_rate
        data['time'] = time_in_seconds  # Füge Zeit-Spalte hinzu, aber nicht als Index
        return data


    def load_file(self, file_path: Path, file_type: str, experiment_name: str) -> pd.DataFrame:
        """
        Lädt Daten aus einer Datei basierend auf ihrem Typ.
        Ersetzt Kommas durch Punkte in numerischen Werten zur Konsistenz.
        """
        if file_type == 'csv':
            data = pd.read_csv(file_path, converters={'RawData': lambda x: x.replace(',', '.')})
        elif file_type == 'tsv':
            data = pd.read_csv(file_path, sep='\t', converters={'RawData': lambda x: x.replace(',', '.')})
        elif file_type == 'pkl':
            data = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Nicht unterstützter Dateityp: {file_type}")

        if data.empty:
            logging.warning(f"Keine Daten in der Datei gefunden: {file_path}")

        # Für Experiment 4, stelle sicher, dass die 'RawData' Spalte als 'data' geladen wird
        if 'RawData' in data.columns and experiment_name == 'experiment4':
            data.rename(columns={'RawData': 'data'}, inplace=True)
        elif 'data' not in data.columns:
            # Assume first column is the data column
            first_column = data.columns[0]
            data.rename(columns={first_column: 'data'}, inplace=True)

        # Stelle sicher, dass alle numerischen Werte einen Punkt als Dezimaltrennzeichen verwenden
        data['data'] = data['data'].astype(str).str.replace(',', '.').astype(float)

        return data

# # example usage
# # Initialisiere den DataLoader
# data_loader = DataLoader('config.json')
# # Lade Daten für ein bestimmtes Experiment
# experiment_name = 'experiment1'
# data = data_loader.load_experiment_data(experiment_name)
# print(data.head())
# # Lade Daten für alle Experimente
# data = data_loader.load_experiment_data()
# print(data.head())
