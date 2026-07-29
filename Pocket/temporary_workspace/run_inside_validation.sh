#!/usr/bin/env bash
source /usr/local/etc/profile.d/conda.sh
conda activate /work/users/vsloken-fnal/envs/my-env
export PYTHONPATH='/work/users/vsloken-fnal/nonp_poco/PocketCoffea'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
cd /work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace
pocket-coffea run --cfg config_validation.py -o debugging_cr_chain_v18 -e dask@purdue-af --custom-run-options custom_full_opt.yaml --overwrite --process-separately --limit-files 50
