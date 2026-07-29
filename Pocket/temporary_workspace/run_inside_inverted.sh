#!/usr/bin/env bash
source /usr/local/etc/profile.d/conda.sh
conda activate /work/users/vsloken-fnal/envs/my-env
export PYTHONPATH='/work/users/vsloken-fnal/nonp_poco/PocketCoffea'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace
pocket-coffea run --cfg config_inverted.py -o debugging_nonp_chain_v18 -e dask@purdue-af --custom-run-options custom_nonprompt_opt.yaml --overwrite --process-separately
