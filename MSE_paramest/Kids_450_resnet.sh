#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=60
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=2g
#SBATCH --gpus=rtx_4090:1
#SBATCH --gres=gpumem:24gb
#SBATCH --job-name=gpu_test1
#SBATCH --output=gpu_24gb.out
#SBATCH --error=gpu_24gb.err


# Start timing
echo "Transfering files to local scratch"
start_time=$(date +%s)


# Copy files to local scratch
mkdir ${TMPDIR}/kids_450_h5_files

#full sample
#rsync -aq /cluster/work/refregier/atepper/kids_450/full_data/kids_450_h5 ${TMPDIR}/kids_450_h5_files/
#small sample
rsync -aq /cluster/work/refregier/atepper/kids_450/small_sample/kids_450_h5 ${TMPDIR}/kids_450_h5_files/

# End timing
end_time=$(date +%s)

# Calculate the elapsed time
elapsed_time=$((end_time - start_time))

# Print the elapsed time
echo "Elapsed time: $elapsed_time seconds"


#Laod the script
module purge
module load gcc/8.2.0 python_gpu/3.10.4
module load eth_proxy
source $HOME/thesis_env3/bin/activate
#cd $HOME/master_thesis/master_thesis/SimCLR
#python -c "import torch; print(torch.version.cuda); print(torch.__version__)"
#nvidia-smi
#nvcc --version

#multi gpu test
python Kids_450_resnet.py --config ./data/configs/config_Kids_450_resnet_2.yaml
#python Kids_450_resnet.py --config ./data/configs/config_Kids_450_resnet_every_epoch.yaml
#python Simclr_kids450.py --config ./configs_simclr_450/config_Simclr_kids450_40gb.yaml