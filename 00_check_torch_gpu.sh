#!/bin/bash

#################################################
## TEMPLATE VERSION 1.01                       ##
#################################################
## ALL SBATCH COMMANDS WILL START WITH #SBATCH ##
## DO NOT REMOVE THE # SYMBOL                  ##
#################################################

#SBATCH --nodes=1                   # How many nodes required? Usually 1
#SBATCH --cpus-per-task=4           # Number of CPU to request for the job
#SBATCH --mem=8GB                   # How much memory does your job require?
#SBATCH --gres=gpu:1                # Do you require GPUS? If not delete this line
#SBATCH --time=02-00:00:00          # How long to run the job for? Jobs exceed this time will be terminated
                                    # Format <DD-HH:MM:SS> eg. 5 days 05-00:00:00
                                    # Format <DD-HH:MM:SS> eg. 24 hours 1-00:00:00 or 24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL  # When should you receive an email?
#SBATCH --output=<insert path prefix>/job_logs/%u.%j.out        	# Where should the log files go?
                                    								# You must provide an absolute path eg /common/home/module/username/
                                    								# If no paths are provided, the output file will be placed in your current working directory

################################################################
## EDIT AFTER THIS LINE IF YOU ARE OKAY WITH DEFAULT SETTINGS ##
################################################################

#SBATCH --partition=<insert partition>
#SBATCH --account=<insert account>
#SBATCH --qos=<insert qos>
#SBATCH --mail-user=<insert email>
#SBATCH --job-name=<insert job name>

#################################################
##            END OF SBATCH COMMANDS           ##
#################################################

# Purge the environment, load the modules we require.
# Refer to https://violet.smu.edu.sg/origami/module/ for more information
module purge
module load Anaconda3/2022.05
module load CUDA/11.7.0

# Do not remove this line even if you have executed conda init
eval "$(conda shell.bash hook)"

# Activate the environment
conda activate ppg_paper_gpu

# Submit your job to the cluster
srun --gres=gpu:1 python -m src.helper.check_torch_gpu
