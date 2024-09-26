# NER Model Training and Evaluation with Continual Learning

This project implements a Named Entity Recognition (NER) model training and evaluation pipeline using spaCy, with a focus on continual learning techniques.

## Features

- Load and preprocess data from CSV files
- Split data into training, testing, and validation sets
- Train NER models using spaCy
- Implement continual learning through fine-tuning
- Evaluate model performance with detailed metrics

## Continual Learning

Continual learning is an approach in machine learning where a model learns continuously from a stream of data, adapting to new tasks while retaining knowledge from previous tasks. In this project, we implement continual learning with the following key properties:

1. **Knowledge retention**: The model is not prone to catastrophic forgetting, maintaining performance on previously learned tasks.
2. **Forward transfer**: The model learns new tasks while reusing knowledge acquired from previous tasks.
3. **Backward transfer**: The model achieves improved performance on previous tasks after learning a new task.
4. **Fixed model capacity**: Memory size remains constant regardless of the number of tasks and the length of the data stream.

Our implementation demonstrates these properties through sequential fine-tuning on different datasets (G1, G2, G3) and evaluating performance across all datasets after each training phase.

## The Dataset and Objective

Our dataset consists of three separate CSV files (G1, G2, G3) with minimal overlap between them. Each dataset contains text and corresponding entity tags. Here's an example of what the data might look like:

```
            text                                                         tags
The patient was prescribed Metformin for their diabetes.                ,18:27:Treatment,36:44:Chronic Disease
A new immunotherapy treatment shows promise for lung cancer patients.   ,4:26:Treatment,51:62:Cancer
Peanut allergy sufferers should avoid products containing traces of nuts.,0:13:Allergy,54:58:Allergy
```

The objective is to correctly identify and classify entities in the text into one of five categories: Treatment, Chronic Disease, Cancer, Allergy, or Other.

These three datasets define three different NER tasks (T1, T2, T3) with the same set of labels but different entities. For example:

- Task T1: Recognize entities in dataset G1
- Task T2: Recognize entities in dataset G2
- Task T3: Recognize entities in dataset G3

We'll keep aside 20% of each dataset as a test set for evaluation purposes.

## Prerequisites

- Python 3.x
- spaCy
- pandas
- scikit-learn
- thinc

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install spacy pandas scikit-learn thinc
   ```
3. Download the spaCy English model:
   ```
   python -m spacy download en_core_web_lg
   ```

## Usage

1. Prepare your data in CSV format with 'text' and 'tags' columns.
2. Update the file paths in the script to point to your data files.
3. Run the script to train and evaluate the models:
   ```
   python ner_model_pipeline.py
   ```

## Pipeline Overview

1. **Data Loading**: Load data from CSV files using `load_data_from_csv()`.
2. **Preprocessing**: Add entities from tags using `add_entities_from_tags()`.
3. **Data Splitting**: Split data into train, test, and validation sets using `split_data_into_sets()`.
4. **Model Training**: 
   - Train initial model (T1) on G1 dataset
   - Fine-tune model (T2) on G2 dataset with samples from G1
   - Further fine-tune model (T3) on G3 dataset with samples from G1 and G2
5. **Evaluation**: Evaluate models using `evaluate_model()` function.

## Evaluation Results

The evaluation results demonstrate the effectiveness of our continual learning approach:

### T1 on G1
- Treatment: F1 Score = 0.9887
- Chronic Disease: F1 Score = 0.9791
- Cancer: F1 Score = 0.9712
- Allergy: F1 Score = 0.9614

### T2 on G2
- Treatment: F1 Score = 0.9839
- Chronic Disease: F1 Score = 0.9862
- Cancer: F1 Score = 0.9783
- Allergy: F1 Score = 0.9450

### T3 on G3
- Treatment: F1 Score = 0.9828
- Chronic Disease: F1 Score = 0.9856
- Cancer: F1 Score = 0.9793
- Allergy: F1 Score = 0.9581

### T2 on G1
- Treatment: F1 Score = 0.9880
- Chronic Disease: F1 Score = 0.9888
- Cancer: F1 Score = 0.9825
- Allergy: F1 Score = 0.9451

### T3 on G2
- Treatment: F1 Score = 0.9857
- Chronic Disease: F1 Score = 0.9878
- Cancer: F1 Score = 0.9801
- Allergy: F1 Score = 0.9667

### T3 on G1
- Treatment: F1 Score = 0.9849
- Chronic Disease: F1 Score = 0.9879
- Cancer: F1 Score = 0.9799
- Allergy: F1 Score = 0.9536

These results show that the fine-tuned models through continual learning maintain or improve performance across different categories and datasets, demonstrating knowledge retention, backward transfer, and fixed model capacity (1.1GB).

## Functions

- `load_data_from_csv(file_path)`: Load data from a CSV file.
- `add_entities_from_tags(dataframe, tag_column)`: Parse tag data to create entities.
- `split_data_into_sets(data, test_size, validation_size, random_state)`: Split data into train, test, and validation sets.
- `save_tok2vec_layer(model_path)`: Save the Tok2Vec layer of a trained model.
- `process_data(data, nlp, doc_bin_path)`: Process data and save as a DocBin.
- `evaluate_model(model_path, test_docs)`: Evaluate a trained model on test documents.
- `process_and_evaluate_model(data_file_path, model_path)`: End-to-end function to process data and evaluate a model.

## Note

Make sure to adjust file paths and configurations as needed for your specific setup and data.

