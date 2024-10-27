#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=61
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24gb
#SBATCH --job-name=downstream_test1
#SBATCH --output=downstream2_24gb.out
#SBATCH --error=downstream2_24gb.err


# Start timing
echo "Transfering files to local scratch"ss
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


#################OLD CENTOS ########################
#Laod the script
#module purge
#module load gcc/8.2.0 python_gpu/3.10.4
#module load eth_proxy
#source $HOME/thesis_env3/bin/activate
#cd $HOME/master_thesis/master_thesis/SimCLR
#python -c "import torch; print(torch.version.cuda); print(torch.__version__)"
#nvidia-smi
#nvcc --version
##################OLD CENTOS##########################

#################NEW UBUNTU########################
module purge
source $HOME/thesis_env_ubuntu/bin/activate
module load stack/2024-05  gcc/13.2.0 python/3.11.6_cuda
module load eth_proxy


##################NEW UBUNTU##########################

#run the model with python script
#python downstream_model.py --config ./data/configs/config_downstream_model.yaml
python downstream_model.py --config ./data/configs/config_downstream_model_2.yaml