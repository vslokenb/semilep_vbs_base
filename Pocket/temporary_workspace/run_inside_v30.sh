#!/usr/bin/env bash
# $1=config  $2=outdir  $3=dataset keys (csv)  $4=opts yaml  $5=scaleout
# $6=limit-files ("" = none)  $7=year (default 2018)
source /usr/local/etc/profile.d/conda.sh
conda activate /work/users/vsloken-fnal/envs/my-env
# the updated checkout with the patched JES
export PYTHONPATH='/work/users/vsloken-fnal/latest_poco/PocketCoffea'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd /work/users/vsloken-fnal/semilep_vbs_base/Pocket/temporary_workspace
# $5 = scaleout, passed on the CLI (-s overrides run_options["scaleout"])
pocket-coffea run --cfg "$1" -o "$2" -e dask@purdue-af \
  --custom-run-options "$4" --process-separately \
  --scaleout "${5:-50}" ${6:+--limit-files "$6"} \
  --filter-years "${7:-2018}" --filter-datasets "$3"
