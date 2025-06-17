# active-nccl-test

Scripts to run NCCL tests actively on a production Slurm cluster

## How to Run
**Get files in place**\
Make sure dir /home/ubuntu is shared among all compute nodes and the controller node.\
`mkdir /home/ubuntu/oci_active_nccl`\
`cd /home/ubuntu/oci_active_nccl`\
`wget https://raw.githubusercontent.com/taipin/active-nccl-test/refs/heads/main/nccl_controller.sh ./`\
`wget https://raw.githubusercontent.com/taipin/active-nccl-test/refs/heads/main/nccl_pair.sbatch ./`\
`wget https://raw.githubusercontent.com/taipin/active-nccl-test/refs/heads/main/nccl.sh ./`\
`chmod +x nccl_controller.sh nccl.sh`\
**Run the scripts**\
To run as a standalone job\
`/home/ubuntu/oci_active_nccl/nccl_controller.sh <partition name> <gpus per node> <min perf> <hist hours> <drain bad node> <drain low node>`\
The default values of the prameters are:\
`<partition name> = compute`\
`<gpus per node> = 8`\
`<min perf> = 100`\
`<hist hours> = 24`\
`<drain bad node> = 0 (do not drain, set to 1 to drain)`\
`<drain low node> = 0 (do not drain, set to 1 to drain)`\
To run as a crontab job\
`crontab -e`\
then paste the following (it will launch the script every 5 minutes) to the end of and save it. One can customize the time interval, but recommendation is not to make the interval less than 2 minutes.\
`*/5 * * * * /home/ubuntu/oci_active_nccl/nccl_controller.sh <partition name> <gpus per node> <min perf> <hist hours> <drain bad node> <drain low node> >> /home/ubuntu/oci_active_nccl/LOG 2>&1` \
**Example 1** Running every 5 minutes with all default paramaters:\
`*/5 * * * * /home/ubuntu/oci_active_nccl/nccl_controller.sh >> /home/ubuntu/oci_active_nccl/LOG 2>&1` \
**Example 2** Running every 5 minutes and drain the bad nodes (see Documentation below for difference between a bad node and a low performing node):\
`*/5 * * * * /home/ubuntu/oci_active_nccl/nccl_controller.sh a10 1 1 24 1 0 >> /home/ubuntu/oci_active_nccl/LOG 2>&1`

## Documentation

The scripts will find idle nodes in the partition. If all nodes have been tested with good results (Allreduce BW > \<min perf\>) in the past \<hist hours\>, then it will exit. Otherwise it will test a random pair of nodes from the idle list. If this test is good, it will update the good node list with node names and timestamps. If the test is bad for this initial pair of nodes, then it will further test each of the nodes from this pair - if this further test is good, it will update the good node list; if the test is bad, it will update the bad node list (or low node list, depending on the performance number obtained) with node names, timestamps and performance number. A bad node is a node which fails to get a valid NCCL Allreduce BW due to various job errors.

## Help



## Contributing


## Security

Please consult the [security guide](./SECURITY.md) for our responsible security vulnerability disclosure process

