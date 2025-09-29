#!/bin/bash

mypartition=${1:-compute}    # partition name of the cluster
gpus_per_node=${2:-8}        # 8 for BM H100 and A100, 4/2/1 for A10 BM, VM2 and VM1
perf_required=${3:-100}      # for nccl allreduce bw in GB/s, adjust accordingly
hist_hours=${4:-24}          # good nodes tested within the past 24 hours will not be picked
((max_hist=hist_hours*3600))
drain_bad_node=${5:-0}       # 1 - drain bad node, any other integer - do not drain bad node
drain_low_node=${6:-0}       # 1 - drain low node, any other integer - do not drain low node

mydir="/home/ubuntu/oci_active_nccl"
cd "$mydir" || exit 10

hist_file="${mydir}/good_nodes_${mypartition}.log"

declare -A hist_dict       # for good nodes and their latest timestamps
declare -a host_list       # (host1 host2 ...), idle and not recently tested hosts
declare -a idle_list       # (host1 host2 ...), idle hosts, used for a work-around for Slurm bug on -w -x and -N
declare -a all_host_list   # (host1 host2 ...), all non-drain hosts, used for a work-around for Slurm bug on -w -x and -N
TL="1:10"

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

get_one_idle_v1 () {
    # get one idle node to pair with node_a
    node_a=$1
    for idle_node in "${idle_list[@]}"; do
        if [[ "$idle_node" != "$node_a" ]]; then
            node_b="$idle_node"
            break
        fi
    done
}

get_one_idle () {
    # get one idle node which is neither $1 (mandadory) nor $2 (optional)
    local node1=$1
    local node2=${2:-}
    node_b=""
    for idle_node in "${idle_list[@]}"; do
        if [[ "$idle_node" != "$node1" && "$idle_node" != "$node2" ]]; then
            node_b="$idle_node"
            break
        fi
    done
}

get_any_node () {
    # get one node which is neither $1 (mandadory) nor $2 (optional)
    local node1=$1
    local node2=${2:-}
    node_b=""
    for any_node in "${all_host_list[@]}"; do
        if [[ "$any_node" != "$node1" && "$any_node" != "$node2" ]]; then
            node_b="$any_node"
            break
        fi
    done
}

wait_for_file () {
    file=$1
    while true; do
        if [ -e "$file" ]; then
            sleep 1
            break
        fi
        sleep $2
    done
}

#-----------------------
# get all non-drain nodes
for host in `sinfo -Neh -p $mypartition -t idle,mix,alloc | cut -d ' ' -f 1`; do all_host_list+=("$host"); done

# get idle node
sinfo_line=`sinfo --noheader -p $mypartition -t idle`
sinfo_rc=$?
#echo "sinfo_line = $sinfo_line"
# make sure sinfo works
if [[ $sinfo_rc -ne 0 ]]; then
    echo "`date +%FT%T`  Partition: $mypartition, sinfo failed to get idle info"
    exit 1
fi

nnodes_idle=`echo $sinfo_line|cut -d ' ' -f 4`
#echo "nnodes_idle = $nnodes_idle"
if [[ $nnodes_idle -lt 2 ]]; then
    echo "`date +%FT%T`  Partition: $mypartition, less than 2 idle nodes"
    exit 2
fi

slurmhosts=`echo $sinfo_line|cut -d ' ' -f 6`

# Convert slurmhosts to array host_list
for host in `scontrol show hostname $slurmhosts`; do
    host_list+=("$host")
    idle_list+=("$host")
done

# Remove hosts which had been tested good recently
read_hist $hist_file
remove_recent

nnodes=${#host_list[@]}
host_candidates=$(IFS=, ; echo "${host_list[*]}")
echo "`date +%FT%T`  Partition: $mypartition, Total Available: ${#all_host_list[@]}, Idle: $nnodes_idle, Idle Not Tested: $nnodes, host_candidates: $host_candidates"

if [[ $nnodes -lt 2 ]]; then
    if [[ $nnodes -eq 1 && $nnodes_idle -gt 1 ]]; then
        # we have one untested idle node and one or more tested idle nodes (good nodes). so we pair the untested node with any good ones and exit afterwards
        # take one node from idle_list
        get_one_idle $host_candidates
        sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $host_candidates,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $host_candidates $drain_bad_node $drain_low_node
        #sbatch --time=$TL --no-requeue -w $host_candidates -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $host_candidates $drain_bad_node $drain_low_node
        exit 0
    else
        #echo "all idle nodes are tested recently"
        exit 3
    fi
fi

# nnodes >= 2
# Submit a job to test one pair from host_candidates. Can remove time limit, but need --wait
# 99.99% will get nodes, but what if we do not get? do not want to put the job in the wait queue

sbatch_info=`sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $host_candidates -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition --wait ${mydir}/nccl_pair.sbatch $perf_required`
sbatch_rc=$?
#echo "sbatch_rc = $sbatch_rc"
echo "sbatch_info = $sbatch_info"
if [[ $sbatch_rc -ne 0 ]]; then exit 4; fi
job_id=`echo $sbatch_info | cut -d ' ' -f 4`
# Sample sbatch_info:
# Submitted batch job 2002

# We want the job state of the job step-0, but the job_nodelist of the whole job
sacct_line=`sacct -no JobID,State,ElapsedRaw,ExitCode,NodeList%60 -X -j ${job_id}`
sacct_line0=`sacct -no JobID,State,ElapsedRaw,ExitCode,NodeList%60 -j ${job_id}.0`
echo "sacct_line  = $sacct_line"
echo "sacct_line0 = $sacct_line0"
job_state=`echo $sacct_line0 | cut -d ' ' -f 2`
job_nodelist=`echo $sacct_line | cut -d ' ' -f 5`
node1=`scontrol show hostname $job_nodelist | head -1`
node2=`scontrol show hostname $job_nodelist | tail -1`

# get an idle node in case the pair just tested is bad and we need to further test them out
if [[ $nnodes_idle -gt 2 ]]; then
    get_one_idle $node1 $node2
else
    # nnodes_idle=2 case, cannot be 1 or 0. Trying to queue on a running node
    get_any_node $node1 $node2
fi
#Double check the job is finished and has a valid performance number
if [[ "$job_state" == "COMPLETED" && $sbatch_rc -eq 0 ]]; then
    # Sometimes there is a lag for slurm log file
    wait_for_file ${mydir}/slurm-${job_id}.out 2
    job_perf=`grep "Avg bus b" ${mydir}/slurm-${job_id}.out | awk '{print $6}'`
    echo "node pair ($node1, $node2) job_perf = $job_perf  perf_required = $perf_required"
    if (( `echo "${job_perf} < $perf_required" | bc -l` )); then
        # low performing, further test the pair. Note the arguments are different
        echo "low perf - further test the bad pair ($node1 , $node2)"
        #TODO check Slurm release update or find a node to pair with node1 and node2
        # A work-around is to find a partner node and use it. 
        # For simplicity, we assume the idle list does not change. 
        # If it does change, then the job will be queued - not a big deal.
        sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $node1,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $node1 $drain_bad_node $drain_low_node
        sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $node2,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $node2 $drain_bad_node $drain_low_node
    else
        # both nodes are good, do nothing
        exit 0
    fi
elif [[ $job_state == CANCELLED* ]]; then
    echo "node pair ($node1, $node2) CANCELLED"
    # In one particular case, the job got cancelled but one node will stay in CG state. We need to single this case out
    sleep 5
    squeue_line=`squeue -p $mypartition --noheader -o "%.18i %.2t %R" -j ${job_id}`
    squeue_state=`echo $squeue_line | cut -d ' ' -f 2`
    squeue_node=`echo $squeue_line | cut -d ' ' -f 3`
    echo "squeue_line = $squeue_line"
    if [[ "$squeue_state" == "CG" ]]; then
        echo "node $squeue_node staying in $squeue_state state, restarting slurmd to clear it..."
        pdsh -R ssh -w $squeue_node 'sudo systemctl restart slurmd'
        echo "done clearing $squeue_state state. please take a look at this node: $squeue_node"
        echo "further testing the other node"
        #TODO check Slurm release update or find a node to pair with node1 and node2
        if [[ "$squeue_node" == "$node1" ]]; then
            sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $node2,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $node1 $drain_bad_node $drain_low_node
        elif [[ "$squeue_node" == "$node2" ]]; then
            sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $node1,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $node1 $drain_bad_node $drain_low_node
        else
            echo "both nodes are in CG - rare case, exiting"
            exit 5
        fi
    else
        # CANCELLED+ but not CG
        #TODO check Slurm release update or find a node to pair with node1 and node2
        echo "further testing the bad pairs ($node1 , $node2)"
        sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $node1,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $node1 $drain_bad_node $drain_low_node
        sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $node2,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $node2 $drain_bad_node $drain_low_node
    fi
else
    echo "node pair ($node1, $node2) FAILED or TIMEOUT"
    echo "further testing the bad pairs ($node1 , $node2)"
    #TODO check Slurm release update or find a node to pair with node1 and node2
    sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $node1,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $node1 $drain_bad_node $drain_low_node
    sbatch --time=$TL --deadline=now+2minutes --no-requeue -w $node2,$node_b -N 2 --gpus-per-node=$gpus_per_node --ntasks-per-node=$gpus_per_node -p $mypartition ${mydir}/nccl_pair.sbatch $perf_required $node2 $drain_bad_node $drain_low_node
fi
