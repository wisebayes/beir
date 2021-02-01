from tqdm.autonotebook import trange
from .util import write_to_json, write_to_tsv
import logging, os

logger = logging.getLogger(__name__)

class QueryGenerator:
    def __init__(self, model, **kwargs):
        self.model = model
        self.gen_qrels = {}
        self.gen_queries = {}

    @staticmethod
    def save(output_dir, queries, qrels, prefix):
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, prefix + "-qrels"), exist_ok=True)
        
        query_file = os.path.join(output_dir, prefix + "-queries.jsonl")
        qrels_file = os.path.join(output_dir, prefix + "-qrels", "train.tsv")
        
        write_to_json(output_file=query_file, data=queries)
        write_to_tsv(output_file=qrels_file, data=qrels)

    def generate(self, corpus, output_dir, top_p=0.95, top_k=25, max_length=64,
                    ques_per_passage=1, prefix="gen", batch_size=32, save_after=100000):
        
        logger.info("Starting to Generate Questions...")
        logger.info("Batch Size: --- {} ---".format(batch_size))
        
        count = 0
        corpus_ids = list(corpus.keys())
        corpus = [corpus[doc_id] for doc_id in corpus_ids]

        for start_idx in trange(0, len(corpus), batch_size, desc='pas'):            
            
            size = len(corpus[start_idx:start_idx + batch_size])
            queries = self.model.generate(
                corpus=corpus[start_idx:start_idx + batch_size], 
                ques_per_passage=ques_per_passage,
                max_length=max_length,
                top_p=top_p,
                top_k=top_k
                )
            
            assert len(queries) == size * ques_per_passage

            for idx in range(size):      
                # Saving generated questions after every "save_after" corpus ids
                if (len(self.gen_queries) % save_after == 0 and len(self.gen_queries) >= save_after):
                    logger.info("Saving {} Generated Queries...".format(len(self.gen_queries)))
                    self.save(output_dir, self.gen_queries, self.gen_qrels, prefix)

                corpus_id = corpus_ids[start_idx + idx]
                start_id = idx * ques_per_passage
                end_id = start_id + ques_per_passage
                query_set = set([q.strip() for q in queries[start_id:end_id]])

                for query in query_set:
                    count += 1
                    query_id = "genQ" + str(count)
                    self.gen_queries[query_id] = query
                    self.gen_qrels[query_id] = {corpus_id: 1}
        
        # Saving finally all the questions
        logger.info("Saving {} Generated Queries...".format(len(self.gen_queries)))
        self.save(output_dir, self.gen_queries, self.gen_qrels, prefix)