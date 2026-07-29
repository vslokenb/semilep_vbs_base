#!/usr/bin/env bash

apptainer shell -B /work/users/vsloken-fnal/ -B /cvmfs -B /cvmfs/cms-griddata.cern.ch -B /cvmfs/cms.cern.ch       -B /tmp   -B /etc/sysconfig/     /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest
source /usr/local/etc/profile.d/conda.sh
conda activate /work/users/vsloken-fnal/envs/my-env
export PYTHONPATH='/work/users/vsloken-fnal/nonp_poco/PocketCoffea'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python -c "
from dask_gateway import Gateway
g = Gateway()
for cluster in g.list_clusters():
    g.connect(cluster.name).shutdown()
    print(f'Stopped {cluster.name}')
"

pocket-coffea run --cfg config_semilep.py -o SR_2018_july22 -e dask@purdue-af  --custom-run-options custom_full_opt.yaml --overwrite --process-separately --limit-files 20

python -c "
from dask_gateway import Gateway
g = Gateway()
for cluster in g.list_clusters():
    g.connect(cluster.name).shutdown()
    print(f'Stopped {cluster.name}')
"
