#!/bin/sh

#SBATCH -J train-sentence-transformer
#SBATCH --nodes=1                   # Single node
#SBATCH --gres=gpu:h100:2                # 8 GPUs on this node
#SBATCH --ntasks-per-node=2         # One process per GPU
#SBATCH --cpus-per-task=8           # 8 CPU cores per task
#SBATCH --time=10:00:00            # Extended wall time
#SBATCH --mem-per-cpu=16gb
#SBATCH --account=edu             # Requested QoS IMPORTANT: REPLACE WITH SBATCH --account=edu if using Terremoto cluster
#SBATCH --output=output_distilbert/distilbert-base-uncased_ED-hotpotqa-lr1e-5-epochs10-temperature20_full_dev_3.out	  # Standard output log file (make sure correct LR and scale are set)
#SBATCH --error=error_distilbert/distilbert-base-uncased_ED-hotpotqa-lr1e-5-epochs10-temperature20_full_dev_3.out        # Standard error log file (make sure correct LR and scale are set)

#RPI Cluster
module purge
module load gcc/14.1.0
module load cuda

# Terremoto Cluster
#module load anaconda
#module load cuda92/toolkit


# Activate Conda environment
source ~/.bashrc
source /insomnia001/depts/edu/COMSE6998/ck3255/anaconda3/etc/profile.d/conda.sh #RPI Cluster only
conda activate myenv39
pip list

# Set Python path for correct environment
export PYTHONPATH=$CONDA_PREFIX/lib/python3.9/site-packages:$PYTHONPATH
#export CUDA_VISIBLE_DEVICES=1,2
pip list
# export your rank 0 information (its address and port)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE
echo "MASTER_ADDR="$MASTER_ADDR

pip list
nvidia-smi
# Run training with torchrun for DDP
torchrun --nproc_per_node=2 /insomnia001/depts/edu/COMSE6998/ck3255/beir/examples/retrieval/training/train_sbert_ddp_2.py
# torchrun --nproc_per_node=2 /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/train_sbert_ddp_2.py
#torchrun --nproc_per_node=4 /gpfs/u/home/MSSV/MSSVntsn/barn/beir/examples/retrieval/training/test_ddp.py
