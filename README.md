
# Running things at the potsdam university cluster
This is better explained in my blog:
https://xarxax.xyz/training-ai-models-potsdam-university/


To run with gpu an apptainer container you should 

Build the image. 

```apptainer build img.sif recipe.def```

Run the image with --nv flag

```apptainer run --nv img.sif```

This last step is done in `slurm.job` so if you just `sbatch slurm.job` while in your cluster you should be fine. 

If you want to see how your task is doing, as [Uni Potsdam](https://www.uni-potsdam.de/en/zim/angebote-loesungen/hpc) says you can check the [Grafana](https://monitor.hpc.uni-potsdam.de/).

Comfy setup aliases for working in your machine but running things on the cluster (you can add this to your .bashrc):

```
#VARIABLES
export YOUR_CLUSTER_USERNAME="yourusername"
export project="/example_apptainer"#the shortcuts only work if the project is in your home folder
export CLUSTER_LOGIN="$YOUR_CLUSTER_USERNAME@login1.hpc.uni-potsdam.de"
export PATH_IN_CLUSTER="/work/$YOUR_CLUSTER_USERNAME/"

#SCRIPTS
alias ssh_uni="ssh -X $CLUSTER_LOGIN"
alias update_example_apptainer="rsync -av -e ssh --exclude='*.pyc' --exclude='.git' --exclude='*/generated_models/*' $HOME/$project $CLUSTER_LOGIN:$PATH_IN_CLUSTER "
alias reverse_update_example_apptainer="rsync -av -e ssh --exclude='*.pyc' --exclude='.git*' --exclude='*generate_model.py' --exclude='*.sif' --exclude='*.bin' --exclude='*.pt'  $CLUSTER_LOGIN:$PATH_IN_CLUSTER/$project $HOME  "
```
