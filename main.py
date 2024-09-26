import logging
from data_processing import DataProcessor
from training import ModelTrainer
from evaluation import ModelEvaluator
from utils import combine_data_with_samples
from logging_utils import setup_logger

def main():
    # Setup logging
    logger = setup_logger('main_logger', 'main.log')
    
    data_processor = DataProcessor()
    model_trainer = ModelTrainer("config.cfg")
    model_evaluator = ModelEvaluator()

    try:
        # Process G1 data
        logger.info("Processing G1 data")
        g1_data = data_processor.load_data_from_csv('/path/to/G1.csv')
        g1_data = data_processor.add_entities_from_tags(g1_data, 'tags')
        train_data_g1, test_data_g1, validation_data_g1 = data_processor.split_data_into_sets(g1_data)

        # Process and save G1 data
        data_processor.process_data(train_data_g1, "/content/train_g1.spacy")
        data_processor.process_data(test_data_g1, "/content/test_g1.spacy")
        data_processor.process_data(validation_data_g1, "/content/validation_g1.spacy")

        # Train model for T1
        logger.info("Training model for T1")
        model_trainer.train_model("./output1", "/content/train_g1.spacy", "/content/validation_g1.spacy")
        tok2vec_path = model_trainer.save_tok2vec_layer("/content/output1/model-last")

        # Process G2 data
        logger.info("Processing G2 data")
        g2_data = data_processor.load_data_from_csv('/path/to/G2.csv')
        g2_data = data_processor.add_entities_from_tags(g2_data, 'tags')
        train_data_g2, test_data_g2, validation_data_g2 = data_processor.split_data_into_sets(g2_data)

        # Combine G2 data with G1 samples
        combined_train_data_t2 = combine_data_with_samples(train_data_g2, [train_data_g1], sample_size=100)

        # Process and save combined T2 data
        data_processor.process_data(combined_train_data_t2, "/content/combined_t2.spacy")
        data_processor.process_data(validation_data_g2, "/content/validation_data_g2.spacy")
        data_processor.process_data(test_data_g2, "/content/test_data_g2.spacy")

        # Update config and train model for T2
        logger.info("Training model for T2")
        model_trainer.update_config_init_tok2vec("config.cfg", tok2vec_path)
        model_trainer.train_model("./output_finetuned_t2", "/content/combined_t2.spacy", "/content/validation_data_g2.spacy")
        tok2vec_path = model_trainer.save_tok2vec_layer("/content/output_finetuned_t2/model-last")

        # Process G3 data
        logger.info("Processing G3 data")
        g3_data = data_processor.load_data_from_csv('/path/to/G3.csv')
        g3_data = data_processor.add_entities_from_tags(g3_data, 'tags')
        train_data_g3, test_data_g3, validation_data_g3 = data_processor.split_data_into_sets(g3_data)

        # Combine G3 data with G1 and G2 samples
        combined_train_data_t3 = combine_data_with_samples(train_data_g3, [train_data_g2, train_data_g1], sample_size=100)

        # Process and save combined T3 data
        data_processor.process_data(combined_train_data_t3, "/content/combined_t3.spacy")
        data_processor.process_data(validation_data_g3, "/content/validation_data_g3.spacy")
        data_processor.process_data(test_data_g3, "/content/test_data_g3.spacy")

        # Update config and train model for T3
        logger.info("Training model for T3")
        model_trainer.update_config_init_tok2vec("config.cfg", tok2vec_path)
        model_trainer.train_model("./output_finetuned_t3", "/content/combined_t3.spacy", "/content/validation_data_g3.spacy")

        # Evaluate models
        logger.info("Evaluating models")
        test_docs_t1 = model_evaluator.load_test_docs("/content/test_g1.spacy")
        test_docs_t2 = model_evaluator.load_test_docs("/content/test_data_g2.spacy")
        test_docs_t3 = model_evaluator.load_test_docs("/content/test_data_g3.spacy")

        # Evaluate T1 model
        f1_scores_t1, global_precision_t1, global_recall_t1, global_f1_t1, weighted_average_f1_t1 = model_evaluator.evaluate_model("/content/output1/model-last", test_docs_t1)
        logger.info(f"T1 Model Evaluation: Weighted Average F1: {weighted_average_f1_t1}")

        # Evaluate T2 model
        f1_scores_t2, global_precision_t2, global_recall_t2, global_f1_t2, weighted_average_f1_t2 = model_evaluator.evaluate_model("/content/output_finetuned_t2/model-last", test_docs_t2)
        logger.info(f"T2 Model Evaluation: Weighted Average F1: {weighted_average_f1_t2}")

        # Evaluate T3 model
        f1_scores_t3, global_precision_t3, global_recall_t3, global_f1_t3, weighted_average_f1_t3 = model_evaluator.evaluate_model("/content/output_finetuned_t3/model-last", test_docs_t3)
        logger.info(f"T3 Model Evaluation: Weighted Average F1: {weighted_average_f1_t3}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
