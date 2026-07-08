import os
import sys
import glob
from coffea.util import load, save
from coffea import processor

# Directory containing .coffea files — pass as CLI arg or fall back to hardcoded default
input_dir = sys.argv[1] if len(sys.argv) > 1 else "debugging_cr_chain_v12"
output_file = os.path.join(input_dir, f"output_merged_{input_dir}.coffea")

coffea_files = glob.glob(os.path.join(input_dir, "*.coffea"))

loaded_outputs = [load(f) for f in coffea_files]
merged_output = processor.accumulate(loaded_outputs)
save(merged_output, output_file)

print(f"Merged {len(coffea_files)} files into {output_file}")