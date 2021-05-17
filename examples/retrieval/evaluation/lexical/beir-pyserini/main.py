import sys, os
import config

from fastapi import FastAPI, File, UploadFile
from pyserini.search import SimpleSearcher
from typing import Optional, List

settings = config.IndexSettings()
app = FastAPI()

@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = f'{dir_path}/datasets/{file.filename}'
    settings.data_folder = f'{dir_path}/datasets/'
    f = open(f'{filename}', 'wb')
    content = await file.read()
    f.write(content)
    return {"filename": file.filename}

@app.get("/index/")
def index(index_name: str, threads: Optional[int] = 8):
    settings.index_name = index_name

    command = f"python -m pyserini.index -collection JsonCollection \
    -generator DefaultLuceneDocumentGenerator -threads {threads} \
    -input {settings.data_folder} -index {settings.index_name} -storeRaw"
    
    os.system(command)
    
    return {200: "OK"}

@app.get("/search/")
def search(q: str, k: Optional[int] = 1000):
    searcher = SimpleSearcher(settings.index_name)
    searcher.set_bm25(k1=0.9, b=0.4)
    
    hits = searcher.search(q, k=k, fields={"contents": 1.0, "title": 1.0})
    results = []
    for i in range(0, len(hits)):
        results.append({'docid': hits[i].docid, 'score': hits[i].score})

    return {'results': results}

@app.post("/batch_search/")
def batch_search(queries: List[str], qids: List[str], k: Optional[int] = 1000, threads: Optional[int] = 8):
    searcher = SimpleSearcher(settings.index_name)
    searcher.set_bm25(k1=0.9, b=0.4)
    hits = searcher.batch_search(queries=queries, qids=qids, k=k, threads=threads,
                                 fields={"contents": 1.0, "title": 1.0})
    results = {}
    for qid, hit in hits.items():
        results[qid] = {}
        for i in range(0, len(hit)):
            results[qid][hit[i].docid] = hit[i].score

    return {'results': results}