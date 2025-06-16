#!/bin/bash
#SBATCH --job-name=nccl_pair
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/home/ubuntu/oci_active_nccl/slurm-%j.out
#SBATCH --error=/home/ubuntu/oci_active_nccl/slurm-%j.out
#SBATCH --exclusive

perf_required=$1
rerun=false
if [[ $# -eq 4 ]]; then
    rerun=true
    retesting_node=$2
    drain_bad_node=$3
    drain_low_node=$4
fi

hist_file="/home/ubuntu/oci_active_nccl/good_nodes_${SLURM_JOB_PARTITION}.log"
low_nodes_file="/home/ubuntu/oci_active_nccl/low_nodes_${SLURM_JOB_PARTITION}.log"
bad_nodes_file="/home/ubuntu/oci_active_nccl/bad_nodes_${SLURM_JOB_PARTITION}.log"

declare -A hist_dict
read_hist () {
    #hist_file=$1
    if [[ -f "$hist_file" ]]; then
        while read -r key value; do
            # Remove leading/trailing whitespace from key and value
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            # Add the key-value pair to the dictionary
            hist_dict["$key"]="$value"
        done < "$hist_file"

        # Print the contents of the dictionary
        #echo "read good nodes history:"
        #for key in "${!hist_dict[@]}"; do
        #  echo "$key  ${hist_dict[$key]}"
        #done
    fi
}

update_hist () {
    node1=`scontrol show hostname $SLURM_NODELIST | head -1`
    node2=`scontrol show hostname $SLURM_NODELIST | tail -1`
    hist_dict["$node1"]="$current_time"
    hist_dict["$node2"]="$current_time"
}

write_hist () {
    #hist_file=$1
    if [[ -f $hist_file ]]; then mv -f $hist_file ${hist_file}.bak${current_time}; fi
    touch $hist_file
    for key in "${!hist_dict[@]}"; do
        printf "%s  %s\n" "$key" "${hist_dict[$key]}" >> "$hist_file"
    done

    # Print the contents of the dictionary
    echo "write good nodes history:"
    for key in "${!hist_dict[@]}"; do
        echo "$key  ${hist_dict[$key]}"
    done
}

# construct hostfile for running mpi jobs
hostfile=/home/ubuntu/oci_active_nccl/tmphostfile.${SLURM_JOB_ID}
scontrol show hostname $SLURM_NODELIST > $hostfile

# Run nccl job as usual, adding timeout
timeout --preserve-status --foreground 60s /home/ubuntu/oci_active_nccl/nccl.sh 1 $hostfile 2>&1 | tee LOG_${SLURM_JOB_ID}
echo "nccl.sh rc = $?"
# take timestamp
current_time=`date +%FT%T`

# To help make sure job_perf is correct, we make this a two-step thing: get the line, and then parse the line
perf_line=`grep "Avg bus b" LOG_${SLURM_JOB_ID} 2> /dev/null`
perf_line_rc=$?
echo "perf_line_rc = $perf_line_rc"
if [[ $perf_line_rc -eq 0 ]]; then
    # We have a complete run
    job_perf=`echo $perf_line | awk '{print $6}'`
    if (( `echo "${job_perf} > $perf_required" | bc -l` )); then
        # good pair
	echo "save good nodes with timestamps"
	read_hist
	update_hist
	write_hist
    else
        # low performing pair - recommend not to drain (by setting drain_low_node = 0)
        if [[ $rerun == "true" ]]; then
            if [[ ! -f $low_nodes_file ]]; then touch $low_nodes_file; fi
            echo "$retesting_node  $current_time  $job_perf" >> $low_nodes_file
            if [[ $drain_low_node -eq 1 ]]; then
                echo "draining node $retesting_node for nccl_low"
                sudo scontrol update state=drain nodename=$retesting_node reason=nccl_low
            fi
        fi
    fi
else
    # job timeout
    if [[ $rerun == "true" ]]; then
        if [[ ! -f $bad_nodes_file ]]; then touch $bad_nodes_file; fi
        echo "$retesting_node  $current_time    0" >> $bad_nodes_file
        if [[ $drain_bad_node -eq 1 ]]; then
            echo "draining node $retesting_node for nccl_timeout"
            sudo scontrol update state=drain nodename=$retesting_node reason=nccl_failed
        fi
    fi
fi

# Do not keep temporary log file and hostfile
rm -f LOG_${SLURM_JOB_ID} $hostfile
