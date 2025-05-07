import os
import pathlib
import json
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from sentence_transformers import SentenceTransformer, losses, InputExample, util as util_st
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader

# ** Get local rank **
local_rank = int(os.environ.get("LOCAL_RANK", 0))

# ** Initialize DDP **
def setup_ddp():
    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    print(f"DDP initialized: Running on GPU {local_rank} of {torch.cuda.device_count()}")

setup_ddp()

device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

print(device)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.device_count())  # Prints the number of available GPUs
print(torch.cuda.current_device())  # Prints the current device index
print(torch.__version__)
#print(sentence_transformers.__version__)


# ** Paths **
model_name = "distilbert/distilbert-base-uncased"
save_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "output", f"{model_name}-hotpotqa-lr3e-5-epochs10-temperature20_full_dev")
os.makedirs(save_dir, exist_ok=True)

# ** Load dataset **
dataset = "hotpotqa"
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
data_path = util.download_and_unzip(url, "datasets")
corpus, queries, qrels = GenericDataLoader(data_path).load(split="train")

# Create InputExample list
# train_examples = []
# for qid, pos_ids in qrels.items():
#     query = queries[qid]
#     for pos_id in pos_ids:
#         if pos_id in corpus:
#             pos_text = corpus[pos_id]["text"]
#             train_examples.append(InputExample(texts=[query, pos_text]))
all_examples = []
for qid, pos_ids in qrels.items():
    query = queries[qid]                         
    for pid in pos_ids:
        if pid in corpus:
            pos_text = corpus[pid]["text"]  
            all_examples.append(InputExample(texts=[query, pos_text]))
# from sentence_transformers.datasets import SentenceTransformerDataset
from sklearn.model_selection import train_test_split
train_examples, val_examples = train_test_split(
    all_examples, test_size=0.10, random_state=42, shuffle=True
)

# Sentence‑Transformers smart‑batching datasets
# train_dataset = SentenceTransformerDataset(train_examples, model)
# eval_dataset  = SentenceTransformerDataset(val_examples,  model)

# ** Load model **
model = SentenceTransformer("distilbert-base-uncased")
model.to(device)

print(model.similarity)

# ** Wrap in DDP if needed **
if torch.cuda.device_count() > 1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

print("Model device:", model.device)


# Create tokenizing dataset
#train_dataset = SentenceTransformerDataset(train_examples, model.module)  # DDP wraps model
train_data = [{"query": example.texts[0], "text": example.texts[1]} for example in train_examples]
train_dataset = Dataset.from_dict({k: [d[k] for d in train_data] for k in train_data[0]})
eval_data = [{"query": example.texts[0], "text": example.texts[1]} for example in val_examples]
eval_dataset = Dataset.from_dict({k: [d[k] for d in eval_data] for k in eval_data[0]})
#train_dataset = train_examples

# ** Create DDP Sampler **
#train_sampler = DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), rank=local_rank, shuffle=True)
#train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler)


# ** Training Args **
# training_args = SentenceTransformerTrainingArguments(
#     output_dir=save_dir,
#     warmup_steps=int(len(train_dataset)//16*0.1),
#     num_train_epochs=6,
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=4,
#     learning_rate=3e-5,
#     weight_decay=0.01,
#     warmup_ratio=0.1,
#     fp16=True,
#     logging_steps=25,
#     save_strategy="epoch",
#     metric_for_best_model="loss",
#     load_best_model_at_end=True,
#     max_grad_norm=1.0,
#     save_total_limit=10,
#     ddp_find_unused_parameters=False,
#     seed=24,
# )
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
    # evaluation_strategy="epoch",
    # metric_for_best_model="loss",
    # load_best_model_at_end=True,
)
training_args = training_args.set_evaluate(
    strategy="epoch",   # or "steps"
    batch_size=16,      # eval batch size
    loss_only=True,     # compute full metrics, not just loss
    delay=0.0,
)
# ** Train **
trainer = SentenceTransformerTrainer(
    model=model.module,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=losses.MultipleNegativesRankingLoss(
    model=model.module,
    similarity_fct=lambda x, y, m: util_st.energy_distance(x, y, m, metric="L1"),
    scale=20.0,
    # evaluation_strategy="epoch", 
    ),
    callbacks=[]
)

#torch.autograd.set_detect_anomaly(True)


trainer.train()

# Cleanup
dist.destroy_process_group()

