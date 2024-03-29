from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
import pandas as pd
from tsfresh.utilities.dataframe_functions import impute

class FeatureExtractor:
    """
    Extrahiere Merkmale aus einer univariaten Zeitreihe mit der tsfresh-Bibliothek.
    """

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extrahiere Zeitreihenmerkmale aus den Daten.
        :param data: Pandas DataFrame mit den Spalten 'id', 'time' und 'value'.
        :return: DataFrame mit extrahierten Merkmalen.
        """
        # Verwende die Spalte 'id', um separate Zeitreihen für jedes Fenster anzugeben
        extraction_settings = ComprehensiveFCParameters()
        extracted_features = extract_features(data,
                                              column_id='id', column_sort='time',
                                              default_fc_parameters=extraction_settings,
                                              impute_function=impute)
        return extracted_features
