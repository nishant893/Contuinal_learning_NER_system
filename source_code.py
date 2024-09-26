


import pandas as pd
import os


from thinc.api import get_current_ops


import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans


import pandas as pd
import os
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

#!python -m spacy download en_core_web_lg


def load_data_from_csv(file_path):
    """
    This function takes a file path to a CSV file and returns a pandas DataFrame.

    Parameters:
    file_path (str): The file path to the CSV file.
    
    Returns:
    DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    # Check if the file path exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")

    # Load the CSV data into a pandas DataFrame
    data = pd.read_csv(file_path)

    return data


def add_entities_from_tags(dataframe, tag_column):
    """
    This function takes a pandas DataFrame and a column name that contains tag data.
    It parses the tag data to create entities and adds them as a new column to the DataFrame.

    Parameters:
    dataframe (DataFrame): The pandas DataFrame containing the tag data.
    tag_column (str): The name of the column containing the tag strings.
    
    Returns:
    DataFrame: The modified pandas DataFrame with an added 'entities' column.
    """
    
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
    
    return dataframe



def split_data_into_sets(data, test_size=0.2, validation_size=0.5, random_state=0):
    """
    This function splits the data into training, testing, and validation sets.
    
    Parameters:
    data (DataFrame): The data to be split.
    test_size (float): The proportion of the dataset to include in the test split.
    validation_size (float): The proportion of the temp dataset to include in the validation split.
    random_state (int): Controls the shuffling applied to the data before applying the split.
    
    Returns:
    tuple: A tuple containing the training, testing, and validation DataFrames.
    """
    
    # First, split into training and temp data
    train_data, temp_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # Adjust validation size to be proportionate to the remaining data
    validation_size_adjusted = validation_size / (1 - test_size)

    # Then, split the temp data into testing and validation sets
    test_data, validation_data = train_test_split(temp_data, test_size=validation_size_adjusted, random_state=random_state)
    
    return train_data, test_data, validation_data

# Example usage:
# train_data_g1, test_data_g1, validation_data_g1 = split_data_into_sets(g1_data)



def save_tok2vec_layer(model_path):
    """
    This function takes the path to a spaCy model and saves the Tok2Vec layer to disk.

    Parameters:
    model_path (str): The file path to the trained spaCy model.
    
    Returns:
    str: The path to the saved Tok2Vec layer file.
    """
    # Check if the model directory exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The specified model path does not exist: {model_path}")

    # Load the trained model from the given path
    nlp = spacy.load(model_path)

    # Access the NER pipe
    ner = nlp.get_pipe("ner")

    # Assuming the first layer of the NER pipeline is Tok2Vec
    tok2vec = ner.model.layers[0]

    # Save the Tok2Vec layer
    ops = spacy.util.get_current_ops()
    tok2vec_bytes = tok2vec.to_bytes()

    # Define the path for saving the tok2vec.bin file
    tok2vec_path = os.path.join(model_path, "tok2vec.bin")

    # Save the file
    with open(tok2vec_path, "wb") as file_:
        file_.write(tok2vec_bytes)

    print(f"Tok2Vec layer saved to: {tok2vec_path}")
    return tok2vec_path



def normalize_label(label):
    label_map = {
        'treatment': 'Treatment',
        'chronic_disease': 'Chronic Disease',
        'cancer': 'Cancer',
        'allergy_name':'Allergy'
    }
    return label_map.get(label, label)

def adjust_span(doc, start, end):
    # Adjust start index to the start of its token
    for token in doc:
        if start >= token.idx and start < token.idx + len(token):
            start = token.idx
            break

    # Adjust end index to the end of its token
    for token in doc:
        if end > token.idx and end <= token.idx + len(token):
            end = token.idx + len(token)
            break

    return start, end

import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans


def process_data(data,nlp,doc_bin_path):
  doc_bin = DocBin()
  for index, row in data.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)
    doc_bin.to_disk(doc_bin_path)





!python -m spacy init fill-config base_config.cfg config.cfg

!python -m spacy train config.cfg --output ./output1 --paths.train ./train_g1.spacy --paths.dev ./validation_g1.spacy



# for t1

g1_data = load_data_from_csv('/content/G2 - G2.csv.csv')
g1_data = add_entities_from_tags(g1_data, 'tags')

train_data_g1, test_data_g1, validation_data_g1 = split_data_into_sets(g1_data)

nlp = spacy.blank("en")  # Load a blank English model
doc_bin = DocBin()


for index, row in train_data_g1.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)
doc_bin.to_disk("/content/train_g1.spacy")


for index, row in test_data_g1.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)
doc_bin.to_disk("/content/test_g1.spacy")


for index, row in validation_data_g1.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)
doc_bin.to_disk("/content/validation_g1.spacy")


!python -m spacy init fill-config base_config.cfg config.cfg

!python -m spacy train config.cfg --output ./output1 --paths.train ./train_g1.spacy --paths.dev ./validation_g1.spacy


import spacy
from thinc.api import get_current_ops
import os

# Specify the path to your trained model
model_path = "/content/output1/model-last"

# Check if the model directory exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The specified model path does not exist: {model_path}")

# Load your trained model
nlp = spacy.load(model_path)

# Access the NER pipe
ner = nlp.get_pipe("ner")

# Assuming the first layer of the NER pipeline is Tok2Vec
tok2vec = ner.model.layers[0]

# Save the Tok2Vec layer
ops = get_current_ops()
tok2vec_bytes = tok2vec.to_bytes()

# Define the path for saving the tok2vec.bin file
tok2vec_path = os.path.join(model_path, "tok2vec.bin")

# Save the file
with open(tok2vec_path, "wb") as file_:
    file_.write(tok2vec_bytes)

print(f"Tok2Vec layer saved to: {tok2vec_path}")



# for t2 


g2_data = load_data_from_csv('/content/G2 - G2.csv.csv')
g2_data = add_entities_from_tags(g2_data, 'tags')

train_data_g2, test_data_g2, validation_data_g2 = split_data_into_sets(g2_data)

nlp = spacy.blank("en")  
doc_bin = DocBin()# 

random_samples_g1 = train_data_g1.sample(n=100, random_state=0)
combined_train_data_t2 = pd.concat([train_data_g2, random_samples_g1])


for index, row in combined_train_data_t2.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)

doc_bin.to_disk("/content/combined_t2.spacy")

for index, row in validation_data_g2.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)

doc_bin.to_disk("/content/validation_data_g2.spacy")


for index, row in test_data_g2.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)

doc_bin.to_disk("/content/test_data_g2.spacy")

# CHANGE IN CONFIG FILE 
""" 
[initialize]
init_tok2vec = "/content/output1/model-last/tok2vec.bin"

"""

!python -m spacy train config.cfg --output ./output_finetuned_t2 --paths.train ./combined_t2.spacy --paths.dev ./vakidation_data_g2.spacy


# for t3 

g3_data = load_data_from_csv('/content/G3 - G3.csv.csv')
g3_data = add_entities_from_tags(g3_data, 'tags')

train_data_g3, test_data_g3, validation_data_g3 = split_data_into_sets(g3_data)

nlp = spacy.blank("en")  
doc_bin = DocBin()

random_samples_g2 = train_data_g2.sample(n=100, random_state=0)
combined_train_data_t3 = pd.concat([train_data_g3, random_samples_g2,random_samples_g1])


for index, row in combined_train_data_t3.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)

doc_bin.to_disk("/content/combined_t3.spacy")

for index, row in validation_data_g3.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)

doc_bin.to_disk("/content/validation_data_g3.spacy")


for index, row in test_data_g3.iterrows():
    doc = nlp.make_doc(row['text'])
    ents = []
    has_label = False
    for start, end, label in row['entities']:
        label = normalize_label(label)
        adjusted_start, adjusted_end = adjust_span(doc, start, end)
        if adjusted_start >= adjusted_end:
            print(f"Invalid adjusted span in text '{row['text']}' [{adjusted_start}, {adjusted_end}) with label '{label}'")
            continue
        span = doc.char_span(adjusted_start, adjusted_end, label=label)
        if span is not None:
            ents.append(span)
            has_label = True
            #print(f"Row {index}:Added span '{span.text}' in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
        else:
            print(f"Row {index}: Skipped span in text: '{row['text']}' [{adjusted_start}, {adjusted_end}] with label '{label}'")
    if not has_label:
        entire_doc_span = doc.char_span(0, len(doc), label='Other')
        if entire_doc_span is not None:
          ents.append(entire_doc_span)

    doc.ents = filter_spans(ents)
    doc_bin.add(doc)

doc_bin.to_disk("/content/test_data_g3.spacy")


# CHANGE IN CONFIG FILE 
""" 
[initialize]
init_tok2vec = "/content/output_finetuned_t2/model-last/tok2vec.bin"

"""

!python -m spacy train config.cfg --output ./output_finetuned_t3 --paths.train ./combined_t3.spacy --paths.dev ./validation_data_g3.spacy



# test_docs

from spacy.tokens import DocBin

nlp_t1 = spacy.blank("en")  # 
test_bin_t1 = DocBin().from_disk("/content/test_g1.spacy")
test_docs_t1 = list(test_bin_t1.get_docs(nlp_t1.vocab))

nlp_t2 = spacy.blank("en")  
test_bin_t2 = DocBin().from_disk("/content/test_g2.spacy")
test_docs_t2 = list(test_bin_t2.get_docs(nlp_t2.vocab))

nlp_t3 = spacy.blank("en")  
test_bin_t3 = DocBin().from_disk("/content/test_t3.spacy")
test_docs_t3 = list(test_bin_t3.get_docs(nlp_t3.vocab))



def evaluate_model(model_path, test_docs):
    # Load the trained model
    nlp_g1_trained = spacy.load(model_path)

    # Initialize counters for each category
    category_stats = {category: {"tp": 0, "fp": 0, "fn": 0} for category in ["Treatment", "Chronic Disease ", "Cancer", "Allergy", "Other"]}

    global_tp, global_fp, global_fn = 0, 0, 0

    for gold_doc in test_docs:
        # Create a predicted document
        pred_doc = nlp_g1_trained(gold_doc.text)

        # Update category stats
        for ent in pred_doc.ents:
            if ent.label_ in category_stats:
                if ent.label_ in [gold_ent.label_ for gold_ent in gold_doc.ents]:
                    category_stats[ent.label_]["tp"] += 1
                    global_tp += 1
                else:
                    category_stats[ent.label_]["fp"] += 1
                    global_fp += 1

        for gold_ent in gold_doc.ents:
            if gold_ent.label_ not in [ent.label_ for ent in pred_doc.ents]:
                category_stats[gold_ent.label_]["fn"] += 1
                global_fn += 1

    # Calculate F1 score for each category
    f1_scores = {}
    weighted_f1_scores = {}
    total_true_instances = sum([stats["total"] for stats in category_stats.values()])
    for category, stats in category_stats.items():
        precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
        recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[category] = f1
        weighted_f1 = f1 * (stats["total"] / total_true_instances) if total_true_instances > 0 else 0
        weighted_f1_scores[category] = weighted_f1

    # Calculate global precision, recall, and F1
    global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
    weighted_average_f1 = sum(weighted_f1_scores.values())
    

    return f1_scores, global_precision, global_recall, global_f1,weighted_average_f1







# function that takes a data set and a model as the input arguments and produces the metrics 

def process_and_evaluate_model(data_file_path, model_path):
    # Load data from CSV
    data = load_data_from_csv(data_file_path)

    # Add entities from tags
    data = add_entities_from_tags(data, 'tags')

    # Initialize a blank Spacy model
    nlp = spacy.blank("en")

    # Process the data
    doc_bin = process_data(data, nlp)

    # Convert doc_bin to a list of docs
    test_docs = list(doc_bin.get_docs(nlp.vocab))

    # Use the existing function to evaluate the model
    f1_scores, global_precision, global_recall, global_f1,weighted_average_f1 =  evaluate_model(model_path, test_docs)
    
    
    return f1_scores , weighted_average_f1
    