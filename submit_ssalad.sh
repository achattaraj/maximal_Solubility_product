#!/bin/bash
#SBATCH --job-name=c40
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --array=0-99
#SBATCH --partition=xeon
#SBATCH --qos=general
#SBATCH --mail-type=END
#SBATCH --mem=1G
#SBATCH -o /home/FCAM/achattaraj/Springsalad/stdout/%x_%A_Run%a.out
#SBATCH -e /home/FCAM/achattaraj/Springsalad/stdout/%x_%A_Run%a.err

module load java/1.8.0_77

file="A5_B5_flex_3nm_2nm_count_40_SIM.txt"

java -Xms64m -Xmx1024m -jar /home/FCAM/achattaraj/Springsalad/jar/LangevinNoVis01.jar "/home/FCAM/achattaraj/Springsalad/sims/$file"  "$SLURM_ARRAY_TASK_ID" 2> /home/FCAM/achattaraj/Springsalad/sims/out

