#!/usr/bin/env bash

WORKDIR=/work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace
IMG=/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-analysis/general/pocketcoffea:lxplus-el9-latest
BINDS=(-B /work/users/vsloken-fnal/ -B /cvmfs -B /cvmfs/cms-griddata.cern.ch -B /cvmfs/cms.cern.ch -B /tmp -B /etc/sysconfig/)

stop_clusters() {
python -c "
from dask_gateway import Gateway
g = Gateway()
for cluster in g.list_clusters():
    g.connect(cluster.name).shutdown()
    print(f'Stopped {cluster.name}')
"
}

stop_clusters

apptainer exec "${BINDS[@]}" "$IMG" bash "$WORKDIR/run_inside_validation.sh"
stop_clusters
stop_clusters

apptainer exec "${BINDS[@]}" "$IMG" bash "$WORKDIR/run_inside_inverted.sh"

stop_clusters
stop_clusters
