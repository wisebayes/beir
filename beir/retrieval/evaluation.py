import pytrec_eval
import logging
from .search.dense import DenseRetrievalExactSearch as DRES

logger = logging.getLogger(__name__)

class EvaluateRetrieval:
    
    def __init__(self, retriever=None, k_values=[1,3,5,10,100,1000]):
        self.k_values = k_values
        self.top_k = max(k_values)
        self.retriever = retriever
            
    def retrieve(self, corpus, queries):
        if not self.retriever:
            raise ValueError("Model/Technique has not been provided!")
        return self.retriever.search(corpus, queries, self.top_k)
    
    def retrieve_and_evaluate(self, corpus, queries, qrels):
        results = self.retrieve(corpus, queries, qrels)
        return evaluate(qrels, results, self.k_values)
    
    @staticmethod
    def evaluate(qrels, results, k_values):
    
        ndcg = {}
        _map = {}
        recall = {}
        precision = {}
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = 0.0
            _map[f"MAP@{k}"] = 0.0
            recall[f"Recall@{k}"] = 0.0
            precision[f"P@{k}"] = 0.0
        
        map_string = "map_cut." + ",".join([str(k) for k in k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
        recall_string = "recall." + ",".join([str(k) for k in k_values])
        precision_string = "P." + ",".join([str(k) for k in k_values])
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string})
        scores = evaluator.evaluate(results)
        
        for query_id in scores.keys():
            for k in k_values:
                ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
                _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
                recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
                precision[f"P@{k}"] += scores[query_id]["P_"+ str(k)]
        
        for k in k_values:
            ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"]/len(scores), 5)
            _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"]/len(scores), 5)
            recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"]/len(scores), 5)
            precision[f"P@{k}"] = round(precision[f"P@{k}"]/len(scores), 5)
        
        for eval in [ndcg, _map, recall, precision]:
            logging.info("\n")
            for k in eval.keys():
                logging.info("{}: {:.4f}".format(k, eval[k]))

        return ndcg, _map, recall, precision
    