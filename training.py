import spacy
import os
from spacy.tokens import DocBin
import subprocess
import logging

class ModelTrainer:
    def __init__(self, config_path):
        self.config_path = config_path
        self.logger = logging.getLogger('main_logger')

    def train_model(self, output_path, train_path, dev_path):
        command = f"python -m spacy train {self.config_path} --output {output_path} --paths.train {train_path} --paths.dev {dev_path}"
        self.logger.info(f"Running command: {command}")
        
        try:
            result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.logger.info(f"Model training completed successfully. Output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Model training failed. Error: {e.stderr}")
            raise RuntimeError(f"Model training failed with exit code {e.returncode}")

    def save_tok2vec_layer(self, model_path):
        if not os.path.exists(model_path):
            error_msg = f"The specified model path does not exist: {model_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            nlp = spacy.load(model_path)
            ner = nlp.get_pipe("ner")
            tok2vec = ner.model.layers[0]
            tok2vec_bytes = tok2vec.to_bytes()

            tok2vec_path = os.path.join(model_path, "tok2vec.bin")
            with open(tok2vec_path, "wb") as file_:
                file_.write(tok2vec_bytes)

            self.logger.info(f"Tok2Vec layer saved to: {tok2vec_path}")
            return tok2vec_path
        except Exception as e:
            self.logger.error(f"Error saving tok2vec layer: {str(e)}", exc_info=True)
            raise

    def update_config_init_tok2vec(self, config_path, tok2vec_path):
        try:
            with open(config_path, 'r') as file:
                config_content = file.read()

            updated_content = config_content.replace(
                '[initialize]',
                f'[initialize]\ninit_tok2vec = "{tok2vec_path}"'
            )

            with open(config_path, 'w') as file:
                file.write(updated_content)

            self.logger.info(f"Updated config file with new tok2vec path: {tok2vec_path}")
        except Exception as e:
            self.logger.error(f"Error updating config file: {str(e)}", exc_info=True)
            raise
