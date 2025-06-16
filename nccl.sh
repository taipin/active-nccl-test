#!/bin/bash
set -e

# number of times to run the nccl test to stress the GPUs and RDMA network. This is different from -n iterations parameter of nccl allreduce which is set below using $iter
max=$1

# This assume, the hostfile  passed is already ordered based on their rackId
if [ -n "$2" ]; then
  hostfile=$2
else
  #hostfile="/home/opc/hostfile.tcp"
  #hostfile="/etc/opt/oci-hpc/hostfile.tcp"
  hostfile="/tmp/ordered_hostfile_system_name"
fi

#ORDEREDMACHINEFILE="ordered_hostfile_system_name"
#ORDEREDRANKMACHINEFILE="rankfile_system_name"
echo INPUTFILE
cat $hostfile

# will generate rack-aware ordered host file
#source /etc/os-release
#if [ $ID == "ol" ] || [ $ID == "centos" ] ; then
#    python3 /home/opc/node_ordering_by_rack.py --input_file $hostfile > /dev/null
#elif [ $ID == "debian" ] || [ $ID == "ubuntu" ] ; then
#    python3 /home/ubuntu/node_ordering_by_rack.py --input_file $hostfile > /dev/null
#fi
#
#hostfile=$ORDEREDMACHINEFILE
#rankfile=$ORDEREDRANKMACHINEFILE

#echo ORDEREDMACHINEFILE
#cat $ORDEREDMACHINEFILE
#echo ORDEREDRANKMACHINEFILE
#cat $ORDEREDRANKMACHINEFILE

# The number of GPUs to use for the test.  Has to be multiplier of 8.  If not passed, all GPUs will be used. 
if [ -n "$3" ]; then
  np=$3
else
  np=$((`less $hostfile | wc -l` * 8 ))
fi

#logfile="nccl_run_allreduce.sh.log"

for x in $(seq 1 1 $max)
do

  echo $x
#  echo $x >> $logfile
#  date >> $logfile

#  rankfile=$rankfile; np=$np ; iter=20;

  mpivars_path=`ls /usr/mpi/gcc/openmpi-*/bin/mpivars.sh`
  source $mpivars_path

  if [[ "$mpivars_path" == "" ]]; then echo "Could not find MPIPATH"; exit; fi

  first_node=`head $hostfile -n 1`
  shape=`ssh $first_node 'curl -sH "Authorization: Bearer Oracle" -L http://169.254.169.254/opc/v2/instance/shape'`
  case $shape in
    BM.GPU.H100.8)
      var_UCX_NET_DEVICES=mlx5_2:1
      var_NCCL_IB_HCA="=mlx5_0,mlx5_1,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_12,mlx5_13,mlx5_14,mlx5_15,mlx5_16,mlx5_17"
      ;;
    BM.GPU.H200.8)
      var_UCX_NET_DEVICES=eth0
      var_NCCL_IB_HCA="=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11"
      ;;
    BM.GPU4.8)
      var_UCX_NET_DEVICES=mlx5_4:1
      var_NCCL_IB_HCA="=mlx5_0,mlx5_2,mlx5_6,mlx5_8,mlx5_10,mlx5_12,mlx5_14,mlx5_16,mlx5_1,mlx5_3,mlx5_7,mlx5_9,mlx5_11,mlx5_13,mlx5_15,mlx5_17"
      ;;
    BM.GPU.B4.8 | BM.GPU.A100-v2.8)
      var_UCX_NET_DEVICES=mlx5_0:1
      var_NCCL_IB_HCA="=mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_9,mlx5_10,mlx5_11,mlx5_12"
      ;;
    BM.GPU.A10.4 | VM.GPU.A10.?)
      var_UCX_NET_DEVICES=ens3
      var_NCCL_IB_HCA=""
      export NCCL_NET=Socket
      export NCCL_SOCKET_IFNAME=ens3
      ;;
    *)
      echo "Shape $shape not enabled yet - add entry in $0 and re-run"
      exit 100
      ;;
  esac

  export UCX_TLS=ud,self,sm,tcp
  export UCX_NET_DEVICES="${var_UCX_NET_DEVICES},eth0"
  export NCCL_IB_HCA="${var_NCCL_IB_HCA}"
  export NCCL_IGNORE_CPU_AFFINITY=1
  export NCCL_DEBUG=WARN
  export OMPI_MCA_coll="^hcoll"
  export OMPI_MCA_pml=ucx
  export NCCL_IB_SL=0
  export NCCL_IB_TC=41
  export NCCL_IB_GID_INDEX=3

  case $shape in
    ?M.GPU.A10.?)
      export UCX_TLS=self,sm,tcp
      export UCX_NET_DEVICES="${var_UCX_NET_DEVICES}"
      srun -K --time=0:40 --mpi=pmix /opt/oci-hpc/nccl-test/build/all_reduce_perf -b4M -e256M -i$((     1024*1024*16)) -n 10
      echo "srun_rc = $?"
      ;;
    *)
      srun -K --time=0:40 --mpi=pmix /opt/oci-hpc/nccl-test/build/all_reduce_perf -b1G -e10G -i$((1024*1024*1024*9)) -n 10
      echo "srun_rc = $?"
      ;;
  esac  

done
