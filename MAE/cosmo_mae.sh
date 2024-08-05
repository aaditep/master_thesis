#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=2g
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24gb
#SBATCH --job-name=mae_pretraining
#SBATCH --output=pretraining_24gb2.out
#SBATCH --error=pretraining_24gb2.err


# Start timing
echo "Transfering files to local scratch"
start_time=$(date +%s)


# Copy files to local scratch
mkdir ${TMPDIR}/kids_450_h5_files

#full sample
rsync -aq /cluster/work/refregier/atepper/kids_450/full_data/kids_450_h5 ${TMPDIR}/kids_450_h5_files/
#small sample
#rsync -aq /cluster/work/refregier/atepper/kids_450/small_sample/kids_450_h5 ${TMPDIR}/kids_450_h5_files/

# End timing
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$((end_time - start_time))

# Print the elapsed time
echo "Elapsed time: $elapsed_time seconds"

###################formeta##########
#module purge
#module load stack/2024-05  gcc/13.2.0 python/3.11.6_cuda
#source $HOME/mae_test_env/bin/activate
#module load eth_proxy
###################formeta##########

module purge
source $HOME/mae_test_env/bin/activate
module load stack/2024-05  gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy
##################NEW UBUNTU##########################

#run the model wisth python script
python cosmo_mae.py

