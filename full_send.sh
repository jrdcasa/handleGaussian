#!/bin/bash

# NJOBS          --> Number of jobs sent to the slurm system
# MAXJOBSINSLURM --> Maximum number of jobs in the slurm system
# JOBSEND        --> Number of jobs finished in the slurm system
# TOTALJOBS      --> Jobs to be sent to the slurm system
# jobs.txt       --> Info of the jobs sent or finished in the slurm system

MAXJOBSINSLURM=50

NJOBS=`squeue -h |wc -ll`

COM=(`ls *.com`)
LENGTH=${#COM[@]}

if [[ ! -e ./jobs.txt ]]; then
    echo -n >./jobs.txt
fi

index=0
while [ ${index} -lt ${LENGTH} ]; do

    current=${COM[$index]}

    if [[ $NJOBS -lt $MAXJOBSINSLURM ]]; then
        base="${COM[$index]%.*}"
        sbatch ${base}.sh  1 > tmp.txt
        jobid=`awk '{print $NF}' tmp.txt`
        echo "${jobid} ${base} ${base}.log" >>./jobs.txt
        rm tmp.txt
        index=`echo "$index+1" | bc -l`
        echo "NEW `date` --> JOBSEND: ${index}, TOTALJOBS: ${TOTALJOBS}, ${base}"
    else
        # Each 60 seconds checks the jobs
        sleep 60
        echo  "WAIT `date` --> JOBSEND: ${JOBSEND}, TOTALJOBS: ${TOTALJOBS}"
    fi

    NJOBS=`squeue -h |wc -ll`
    TOTALJOBS=`ls -ld ./*_[0-9]*com |wc -ll`
done

echo 'Jobs Done!!!!'
