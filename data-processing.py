import pandas as pd
import os
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from sklearn.model_selection import train_test_split
from logging_utils import setup_logger

# Set up logger
logger = setup_logger(__name__, 'data_processing.log')

class DataProcessor:
    def __init__(self):
        self.nlp = spacy.blank("en")
        logger.info("DataProcessor initialized with blank English language model")

    def load_data_from_csv(self, file_path):
        logger.info(f"Attempting to load data from CSV: {file_path}")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"The specified file path does not exist: {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from CSV. Shape: {data.shape}")
        return data

    def add_entities_from_tags(self, dataframe, tag_column):
        logger.info(f"Adding entities from tags in column: {tag_column}")
        def parse_tags(tag_string):
            entities = []
            if tag_string:
                for tag in tag_string.split(','):
                    parts = tag.split(':')
                    if len(parts) == 3:
                        start, end, label = int(parts[0]), int(parts[1]), parts[2]
                        entities.append((start-1, end-1, label))
            return entities
        
        dataframe['entities'] = dataframe[tag_column].apply(parse_tags)
        logger.info(f"Entities added to dataframe. New shape: {dataframe.shape}")
        return dataframe

    def split_data_into_sets(self, data, test_size=0.2, validation_size=0.5, random_state=0):
        logger.info(f"Splitting data into sets. Test size: {test_size}, Validation size: {validation_size}")
        train_data, temp_data = train_test_split(data, test_size=test_size, random_state=random_state)
        validation_size_adjusted = validation_size / (1 - test_size)
        test_data, validation_data = train_test_split(temp_data, test_size=validation_size_adjusted, random_state=random_state)
        logger.info(f"Data split complete. Train: {len(train_data)}, Test: {len(test_data)}, Validation: {len(validation_data)}")
        return train_data, test_data, validation_data

    def normalize_label(self, label):
        label_map = {
            'treatment': 'Treatment',
            'chronic_disease': 'Chronic Disease',
            'cancer': 'Cancer',
            'allergy_name': 'Allergy'
        }
        normalized = label_map.get(label, label)
        if normalized != label:
            logger.debug(f"Normalized label: {label} -> {normalized}")
        return normalized

    def adjust_span(self, doc, start, end):
        original_start, original_end = start, end
        for token in doc:
            if start >= token.idx and start < token.idx + len(token):
                start = token.idx
                break
        for token in doc:
            if end > token.idx and end <= token.idx + len(token):
                end = token.idx + len(token)
                break
        if (start, end) != (original_start, original_end):
            logger.debug(f"Adjusted span: [{original_start}, {original_end}] -> [{start}, {end}]")
        return start, end

    def process_data(self, data, doc_bin_path):
        logger.info(f"Processing data and saving to {doc_bin_path}")
        doc_bin = DocBin()
        for index, row in data.iterrows():
            doc = self.nlp.make_doc(row['text'])
            ents = []
            has_label = False
            for start, end, label in row['entities']:
                label = self.normalize_label(label)
                adjusted_start, adjusted_end = self.adjust_span(doc, start, end)
                if adjusted_start >= adjusted_end:
                    logger.warning(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
                    continue
                span = doc.char_span(adjusted_start, adjusted_end, label=label)
                if span is not None:
                    ents.append(span)
                    has_label = True
                else:
                    logger.warning(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
            if not has_label:
                entire_doc_span = doc.char_span(0, len(doc), label='Other')
                if entire_doc_span is not None:
                    ents.append(entire_doc_span)
                    logger.info(f"Row {index}: Added 'Other' label for entire document")

            doc.ents = filter_spans(ents)
            doc_bin.add(doc)
        doc_bin.to_disk(doc_bin_path)
        logger.info(f"Data processing complete. Doc bin saved to {doc_bin_path}")
        return doc_bin
