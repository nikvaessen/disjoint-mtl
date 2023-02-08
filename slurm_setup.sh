# this file is called before `srun` in the sbatch submission files
source export_env.sh
printenv

SLEEP_TIME=${SLURM_ARRAY_TASK_ID:--1}
SLEEP_TIME=$(($SLEEP_TIME + 1))
SLEEP_TIME=$(($SLEEP_TIME * 3))
echo "sleeping for $SLEEP_TIME seconds"
sleep $SLEEP_TIME