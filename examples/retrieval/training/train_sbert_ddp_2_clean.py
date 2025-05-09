# ---------------------------
# Imports and Setup
# ---------------------------
import os
import pathlib
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sentence_transformers import SentenceTransformer, losses, InputExample, util as util_st
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import models
from datasets import Dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import wandb, os

# Initialize Weights & Biases for experiment tracking
wandb.init(project="HPML-Energy")

from datetime import datetime
import pytz 

# Set timezone and create a timestamp for output directories
ny_tz      = pytz.timezone("America/New_York")
timestamp  = datetime.now(ny_tz).strftime("%Y-%m-%d_%H-%M-%S")

# ----------------------------------------------------------------------
# DDP (Distributed Data Parallel) Setup
# ----------------------------------------------------------------------
def setup_ddp():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    print(f"DDP initialized on GPU {local_rank} of {torch.cuda.device_count()}")
    return local_rank

# Initialize DDP and get local rank
local_rank = setup_ddp() 

# Synchronize timestamp across all processes (DDP only)
if dist.get_world_size() > 1:                         # DDP only
    obj = [timestamp] if local_rank == 0 else [None]
    dist.broadcast_object_list(obj, src=0)
    timestamp = obj[0]

# Set device for this process
# local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

# Print device and CUDA info for debugging
print(device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.device_count())  # Prints the number of available GPUs
print(torch.cuda.current_device())  # Prints the current device index
print(torch.__version__)
#print(sentence_transformers.__version__)

# ---------------------------
# Paths and Dataset Loading
# ---------------------------
model_name = "distilbert-base-uncased"
# Output directory for model checkpoints and logs
save_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name}-hotpotqa-lr3e-5-epochs10_full_dev_norm_1_10_{timestamp}")
os.makedirs(save_dir, exist_ok=True)

# Download and load the HotpotQA dataset using BEIR utilities
dataset = "hotpotqa"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")

# Load train and dev splits
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")
dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_path).load(split="dev")

# Convert BEIR data into Hugging Face Dataset format for training
train_data = [{"query": queries[qid], "positive": corpus[pos_id]["text"]} for qid, pos_doc_ids in qrels.items() for pos_id in pos_doc_ids]
train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})
# Convert dev set to Hugging Face Dataset
dev_data = [{"doc_id": doc_id, "text": doc["text"]} for doc_id, doc in dev_corpus.items()]
dev_dataset = Dataset.from_list(dev_data)  # Convert to Hugging Face Dataset

# ---------------------------
# Evaluator Setup
# ---------------------------
# Define evaluator for IR metrics on dev set
ir_evaluator = InformationRetrievalEvaluator(
    queries=dev_queries, corpus=dev_corpus, relevant_docs=dev_qrels,
    name="hotpotqa-dev", show_progress_bar=True
)

# ---------------------------
# Model Construction
# ---------------------------
# Build the SentenceTransformer model from scratch
word_emb   = models.Transformer(model_name, max_seq_length=512)

# mean-pool + L2-normalise, exactly like the original SBERT papers
pooling    = models.Pooling(
                  word_emb.get_word_embedding_dimension(),
                  pooling_mode_mean_tokens=True,
                  pooling_mode_cls_token=False,
                  pooling_mode_max_tokens=False)

normalise  = models.Normalize()                

model = SentenceTransformer(modules=[word_emb, pooling, normalise])
model.to(device) 
# model = SentenceTransformer("distilbert-base-uncased")
# model.to(device)

print(model.similarity)

# Wrap model in DDP if using multiple GPUs
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

print("Model device:", model.device)

# ---------------------------
# Training Arguments
# ---------------------------
# Set up training arguments for SentenceTransformerTrainer
training_args = SentenceTransformerTrainingArguments(
    output_dir=str(save_dir),

    # scheduling
    num_train_epochs=10,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=int(len(train_dataset) / 16 * 0.1),

    # optimisation
    fp16=True,
    max_grad_norm=1.0,

    # logging / saving
    logging_steps=25,
    save_steps=0,            # save at epoch end is not built‑in → 0 disables
    save_total_limit=10,

    # misc
    ddp_find_unused_parameters=False,
    seed=24,
    dataloader_drop_last=True,
    eval_strategy="epoch",
    metric_for_best_model="loss",
    save_strategy="epoch",
    greater_is_better=False,
    load_best_model_at_end=True,
)

# Set evaluation strategy for the trainer
training_args = training_args.set_evaluate(
    strategy="epoch",   # or "steps"
    batch_size=16,      # eval batch size
    loss_only=True,     # compute full metrics, not just loss
    delay=0.0,
)

# ---------------------------
# Trainer Setup and Training
# ---------------------------
# Initialize the trainer with model, datasets, evaluator, and loss
trainer = SentenceTransformerTrainer(
    model=model.module,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    evaluator=ir_evaluator,
    loss=losses.MultipleNegativesRankingLoss(
    model=model.module,
    similarity_fct=lambda x, y, m: util_st.energy_distance(x, y, m, metric="L1"),
    scale=20.0,
    # evaluation_strategy="epoch", 
    # run_name="hotpotqa-ddp-v1",
    ),
    # callbacks=[]
)

#torch.autograd.set_detect_anomaly(True)

# Start training
trainer.train()

# Cleanup DDP process group
dist.destroy_process_group()

