#!/bin/bash

mypartition=${1:-compute}    # partition name of the cluster
gpus_per_node=${2:-8}        # 8 for BM H100 and A100, 4/2/1 for A10 BM, VM2 and VM1
perf_required=${3:-100}      # for nccl allreduce bw in GB/s, adjust accordingly
hist_hours=${4:-24}          # good nodes tested within the past 24 hours will not be picked
((max_hist=hist_hours*3600))
drain_bad_node=${5:-0}       # 1 - drain bad node, any other integer - do not drain bad node
drain_low_node=${6:-0}       # 1 - drain low node, any other integer - do not drain low node

hist_file="/home/ubuntu/oci_active_nccl/good_nodes_${mypartition}.log"

declare -A hist_dict   # for good nodes and their latest timestamps
declare -a host_list   # (host1 host2 ...)

read_hist () {
    hist_file=$1
    if [[ -f "$hist_file" ]]; then
        while read -r key value; do
            # Remove leading/trailing whitespace from key and value
            key=$(echo "$key" | xargs)
            value=$(echo "$value" | xargs)
            # Add the key-value pair to the dictionary
            hist_dict["$key"]="$value"
        done < "$hist_file"

        # Print the contents of the dictionary
        #echo "The good nodes history:"
        #for key in "${!hist_dict[@]}"; do
        #  echo "$key  ${hist_dict[$key]}"
        #done
    fi
}

remove_recent () {
    current_time=`date +%s`
    (( cutoff_time = current_time - max_hist ))
    for ((i = ${#host_list[@]} - 1; i >= 0; i--)); do
        host=${host_list[i]}
        # only compare the nodes which have non-empty timestamps
        if [[ "s${hist_dict[$host]}" != "s" ]]; then
            host_mark_time=`date -d "${hist_dict[$host]}" +%s`
            if [[ "$host_mark_time" > "$cutoff_time" ]]; then
                unset "host_list[$i]"
            fi
        fi
    done
    # Re-index the array
    host_list=("${host_list[@]}")
}

#-----------------------
# get idle node
sinfo_line=`sinfo -p $mypartition -t idle 2>&1 | grep " idle "`
sinfo_rc=$?
echo "sinfo_line = $sinfo_line"
# make sure grep gets (at least) one matching line
if [[ $sinfo_rc -ne 0 ]]; then
    echo "sinfo failed to get idle info"
    exit 1
fi

nnodes_idle=`echo $sinfo_line|cut -d ' ' -f 4`
echo "nnodes_idle = $nnodes_idle"
if [[ $nnodes_idle -lt 2 ]]; then
    echo "less than 2 idle nodes"
    exit 2
fi

slurmhosts=`echo $sinfo_line|cut -d ' ' -f 6`

# Convert slurmhosts to array host_list
for host in `scontrol show hostname $slurmhosts`; do
    host_list+=("$host")
done

# Remove hosts which had been tested good recently
read_hist $hist_file
remove_recent

nnodes=${#host_list[@]}
host_candidates=$(IFS=, ; echo "${host_list[*]}")
echo "host_candidates = $host_candidates"

if [[ $nnodes -lt 2 ]]; then
    if [[ $nnodes -eq 1 && $nnodes_idle -gt 1 ]]; then
        # we have one untested idle node and one or more tested idle nodes (good nodes). so we pair the untested node with any good ones and exit afterwards
        sbatch --time=1:10 --no-requeue -w $host_candidates -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition /home/ubuntu/oci_active_nccl/nccl_pair.sbatch $perf_required $host_candidates $drain_bad_node $drain_low_node
        exit 0
    else
        echo "all idle nodes are tested recently"
        exit 3
    fi
fi

# Submit a job to test one pair from host_candidates. Can remove time limit, but need --wait
# 99.99% will get nodes, but what if we do not get? do not want to put the job in the wait queue

sbatch_info=`sbatch --time=1:10 --no-requeue -w $host_candidates -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition --wait /home/ubuntu/oci_active_nccl/nccl_pair.sbatch $perf_required`
sbatch_rc=$?
echo "sbatch_info = $sbatch_info"
job_id=`echo $sbatch_info | cut -d ' ' -f 4`
# Sample sbatch_info:
# Submitted batch job 2002

sacct_line=`sacct -no JobID,State,ElapsedRaw,ExitCode,NodeList%60 -j ${job_id}.0`
echo "sacct_line = $sacct_line"
job_state=`echo $sacct_line | cut -d ' ' -f 2`
job_nodelist=`echo $sacct_line | cut -d ' ' -f 5`
node1=`scontrol show hostname $job_nodelist | head -1`
node2=`scontrol show hostname $job_nodelist | tail -1`
if [[ "$job_state" == "COMPLETED" ]]; then
    job_perf=`grep "Avg bus b" slurm-${job_id}.out | awk '{print $6}'`
    echo "job_perf = $job_perf  perf_required = $perf_required"
    if (( `echo "${job_perf} < $perf_required" | bc -l` )); then
        # low performing, further test the pair. Note the arguments are different
	echo "low perf - further test the bad pair ($node1 , $node2)"
        sbatch --time=1:10 --no-requeue -w $node1 -x $node2 -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition /home/ubuntu/oci_active_nccl/nccl_pair.sbatch $perf_required $node1 $drain_bad_node $drain_low_node
        sbatch --time=1:10 --no-requeue -w $node2 -x $node1 -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition /home/ubuntu/oci_active_nccl/nccl_pair.sbatch $perf_required $node2 $drain_bad_node $drain_low_node
    else
        # both nodes are good, do nothing
        exit 0
    fi
else
    # FAILED, CANCELLED, or TIMEOUT, further test the pair. Note the arguments are different
    echo "FAILED, CANCELLED, or TIMEOUT, - further test the bad pairs ($node1 , $node2)"
    sbatch --time=1:10 --no-requeue -w $node1 -x $node2 -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition /home/ubuntu/oci_active_nccl/nccl_pair.sbatch $perf_required $node1 $drain_bad_node $drain_low_node
    sbatch --time=1:10 --no-requeue -w $node2 -x $node1 -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition /home/ubuntu/oci_active_nccl/nccl_pair.sbatch $perf_required $node2 $drain_bad_node $drain_low_node
fi
