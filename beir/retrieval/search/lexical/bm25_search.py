from .elastic_search import ElasticSearch
import tqdm
from tqdm import trange

class BM25Search:
    def __init__(self, index_name, hostname="localhost", keys={"title": "title", "body": "txt"}, 
                 batch_size=128, timeout=100, retry_on_timeout=True, maxsize=24):
        self.results = {}
        self.batch_size = batch_size
        self.config = {
            "hostname": hostname, 
            "index_name": index_name,
            "keys": keys,
            "timeout": timeout,
            "retry_on_timeout": retry_on_timeout,
            "maxsize": maxsize
        }
        self.es = ElasticSearch(self.config)
        self.initialise()
    
    def initialise(self):
        self.es.delete_index()
        self.es.create_index()
    
    def search(self, corpus, queries, top_k):
        # Index Corpus within elastic-search
        self.index(corpus)
        
        #retrieve results from BM25 
        query_ids = list(queries.keys())
        queries = [queries[qid]["text"] for qid in query_ids]
        
        for start_idx in trange(0, len(queries), self.batch_size, desc='que'):
            query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
            results = self.es.lexical_multisearch(
                texts=queries[start_idx:start_idx+self.batch_size], 
                top_hits=top_k + 1) # Add 1 extra if query is present with documents
            
            for (query_id, hit) in zip(query_ids_batch, results):
                scores = {}
                for corpus_id, score in hit['hits']:
                    if corpus_id != query_id: # query doesnt return in results
                        scores[corpus_id] = score
                    self.results[query_id] = scores
        
        return self.results
        
    
    def index(self, corpus):
        progress = tqdm.tqdm(unit="docs", total=len(corpus))
        # dictionary structure = {_id: {title_key: title, text_key: text}}
        dictionary = {idx: {
            self.config["keys"]["title"]: corpus[idx].get("title", None), 
            self.config["keys"]["body"]: corpus[idx].get("text", None)
            } for idx in list(corpus.keys())
        }
        self.es.bulk_add_to_index(
                generate_actions=self.es.generate_actions(
                dictionary=dictionary, update=False),
                progress=progress
                )


    def bm25_rank(self, queries, judgements, top_k, batch_size):
        """[summary]

        Args:
            queries ([type]): [description]
            top_k ([type]): [description]
            batch_size ([type]): [description]
        """
        _type = next(iter(list(judgements.values())[0]))
        generator = chunks(list(queries.keys()), batch_size)
        batches = int(len(queries)/batch_size)
        total = batches if len(queries) % batch_size == 0 else batches + 1 
        
        for query_id_chunks in tqdm.tqdm(generator, total=total):
            texts = [queries[query_id] for query_id in query_id_chunks]
            results = self.es.lexical_multisearch(
                texts=texts, top_hits=top_k + 1) 
            # add 1 extra just incase if query within document

            for (query_id, hit) in zip(query_id_chunks, results):
                scores = {}
                for corpus_id, score in hit['hits']:
                    corpus_id = type(_type)(corpus_id)
                    # making sure query doesnt return in results
                    if corpus_id != query_id:
                        scores[corpus_id] = score
                    
                self.rank_results[query_id] = scores
                
        return self.rank_results