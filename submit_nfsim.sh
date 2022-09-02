#!/bin/bash
#SBATCH --job-name=c60uM
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --array=1-100
#SBATCH --partition=xeon
#SBATCH --qos=general
#SBATCH --mail-type=END
#SBATCH --mem=1G
#SBATCH -o /outpath/%x_run%a.out

inpath="/home/FCAM/.../mypath"

sysname="FTC_A5_B5_60uM.xml"

t_end=0.05
numSteps=50

module load BioNetGen/2.5.0

path=${file/.xml/}

# when cb is enabled 
outpath="${path}_with_cb"

mkdir -p $outpath

NFsim "-xml" "$file" "-sim" "$t_end" "-oSteps" "$numSteps" "-cb" "-seed" "$SLURM_ARRAY_TASK_ID" "-o" "$outpath/Run_$SLURM_ARRAY_TASK_ID.gdat" "-ss" "$outpath/Run_$SLURM_ARRAY_TASK_ID.species" 

# when cb is disabled
outpath="${path}_wo_cb"

mkdir -p $outpath

NFsim "-xml" "$file" "-sim" "$t_end" "-oSteps" "$numSteps" "-seed" "$SLURM_ARRAY_TASK_ID" "-o" "$outpath/Run_$SLURM_ARRAY_TASK_ID.gdat" "-ss" "$outpath/Run_$SLURM_ARRAY_TASK_ID.species" 

