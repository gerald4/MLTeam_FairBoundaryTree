#!/bin/bash
#SBATCH --job-name=Synthetic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=24:00:00
#SBATCH --array=1-1
#SBATCH --mail-type=END
#SBATCH --mail-user=valentin.delchevalerie@unamur.be
# ------------------------- work -------------------------

echo "Job start at $(date)"
source ~/VENV/bin/activate
python3 launch_DI.py ${SLURM_ARRAY_TASK_ID}
echo "Job end at $(date)"
