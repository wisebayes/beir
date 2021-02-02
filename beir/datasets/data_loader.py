import json
import os
import logging
import csv

logger = logging.getLogger(__name__)

class GenericDataLoader:
    
    def __init__(self, data_folder, prefix=None, corpus_file="corpus.jsonl", query_file="queries.jsonl", qrels_folder="qrels"):
        self.corpus = {}
        self.queries = {}
        self.qrels = {}
        self.data_folder = data_folder
        
        if prefix:
            query_file = prefix + "-" + query_file
            qrels_folder = prefix + "-" + qrels_folder

        self.corpus_file = os.path.join(self.data_folder, corpus_file)
        self.query_file = os.path.join(self.data_folder, query_file)
        self.qrels_folder = os.path.join(self.data_folder, qrels_folder)
        self.qrels_file = ""
        
    
    def load(self, split="test"):
        
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        
        if not len(self.queries):
            self._load_queries()
        
        if not len(self.corpus):
            self._load_corpus()
            logger.info("Loaded %d %s Documents.", len(self.corpus), split.upper())
            logger.info("Doc Example: %s", list(self.corpus.values())[0])
        
        
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.corpus, self.queries, self.qrels
    
    def load_corpus(self):
        
        if not len(self.corpus):
            self._load_corpus()
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        return self.corpus
    
    def _load_corpus(self):
        
        with open(self.corpus_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.corpus[line.get("_id")] = {
                    "text": line.get("text"),
                    "title": line.get("title"),
                }
    
    def _load_queries(self):
        
        with open(self.query_file, encoding='utf8') as fIn:
            for line in fIn:
                line = json.loads(line)
                self.queries[line.get("_id")] = {
                    "text": line.get("text")
                }
        
    def _load_qrels(self):
        
        reader = csv.reader(open(self.qrels_file, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)
        
        for id, row in enumerate(reader):
            query_id, corpus_id, score = row[0], row[1], int(row[2])
            
            if query_id not in self.qrels and score > 0:
                self.qrels[query_id] = {corpus_id: score}
            elif score > 0:
                self.qrels[query_id][corpus_id] = score