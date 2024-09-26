import spacy
from spacy.tokens import DocBin
from logging_utils import setup_logger

# Set up logger
logger = setup_logger(__name__, 'evaluation.log')

class ModelEvaluator:
    def __init__(self):
        self.categories = ["Treatment", "Chronic Disease", "Cancer", "Allergy", "Other"]
        logger.info("ModelEvaluator initialized with categories: %s", self.categories)

    def load_test_docs(self, test_path):
        logger.info(f"Loading test documents from {test_path}")
        nlp = spacy.blank("en")
        test_bin = DocBin().from_disk(test_path)
        docs = list(test_bin.get_docs(nlp.vocab))
        logger.info(f"Loaded {len(docs)} test documents")
        return docs

    def evaluate_model(self, model_path, test_docs):
        logger.info(f"Evaluating model from {model_path}")
        nlp_trained = spacy.load(model_path)
        category_stats = {category: {"tp": 0, "fp": 0, "fn": 0, "total": 0} for category in self.categories}
        global_tp, global_fp, global_fn = 0, 0, 0

        for i, gold_doc in enumerate(test_docs):
            logger.debug(f"Evaluating document {i+1}/{len(test_docs)}")
            pred_doc = nlp_trained(gold_doc.text)

            for ent in pred_doc.ents:
                if ent.label_ in category_stats:
                    if ent.label_ in [gold_ent.label_ for gold_ent in gold_doc.ents]:
                        category_stats[ent.label_]["tp"] += 1
                        global_tp += 1
                    else:
                        category_stats[ent.label_]["fp"] += 1
                        global_fp += 1

            for gold_ent in gold_doc.ents:
                category_stats[gold_ent.label_]["total"] += 1
                if gold_ent.label_ not in [ent.label_ for ent in pred_doc.ents]:
                    category_stats[gold_ent.label_]["fn"] += 1
                    global_fn += 1

        logger.info("Evaluation complete. Calculating scores...")
        f1_scores, weighted_f1_scores = self.calculate_scores(category_stats)
        global_precision, global_recall, global_f1 = self.calculate_global_scores(global_tp, global_fp, global_fn)
        weighted_average_f1 = sum(weighted_f1_scores.values())

        logger.info(f"Global Precision: {global_precision:.4f}, Recall: {global_recall:.4f}, F1: {global_f1:.4f}")
        logger.info(f"Weighted Average F1: {weighted_average_f1:.4f}")
        
        return f1_scores, global_precision, global_recall, global_f1, weighted_average_f1

    def calculate_scores(self, category_stats):
        logger.debug("Calculating category-wise scores")
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
            logger.debug(f"{category}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Weighted F1={weighted_f1:.4f}")

        return f1_scores, weighted_f1_scores

    def calculate_global_scores(self, global_tp, global_fp, global_fn):
        logger.debug("Calculating global scores")
        global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
        global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
        global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
        return global_precision, global_recall, global_f1
