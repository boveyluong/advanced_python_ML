import sys
import logging
sys.path.append('scripts')

import data_exploration
import data_extraction
import signal_preprocessing
import feature_engineering
import labelling
import model_training_and_evaluation
import optimization

def main(script_name=None):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        if script_name:
            logger.info(f"Starting {script_name}...")
            if script_name == "data_exploration":
                data_exploration.main()
            elif script_name == "data_extraction":
                data_extraction.main()
            elif script_name == "signal_preprocessing":
                signal_preprocessing.main()
            elif script_name == "feature_engineering":
                feature_engineering.main()
            elif script_name == "labelling":
                labelling.main()
            elif script_name == "model_training_and_evaluation":
                model_training_and_evaluation.main()
            elif script_name == "optimization":
                optimization.main()
            else:
                logger.warning(f"Unknown script name: {script_name}")
        else:
            logger.info("Starting data exploration...")
            data_exploration.main()
            logger.info("Data exploration completed.")
            
            logger.info("Starting data extraction...")
            data_extraction.main()
            logger.info("Data extraction completed.")
            
            logger.info("Starting signal preprocessing...")
            signal_preprocessing.main()
            logger.info("Signal preprocessing completed.")
            
            logger.info("Starting feature engineering...")
            feature_engineering.main()
            logger.info("Feature engineering completed.")
            
            logger.info("Starting labelling...")
            labelling.main()
            logger.info("Labelling completed.")
            
            logger.info("Starting model training and evaluation...")
            model_training_and_evaluation.main()
            logger.info("Model training and evaluation completed.")
            
            logger.info("Starting optimization...")
            optimization.main()
            logger.info("Optimization completed.")
        
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))

if __name__ == "__main__":
    script_name = input("Enter the name of the script to run (or press Enter to run all scripts): ")
    main(script_name)
