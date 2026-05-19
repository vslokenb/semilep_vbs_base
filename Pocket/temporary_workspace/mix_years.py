# from coffea.util import load, save
# from coffea import processor

# # Load files
# out_2017 = load("data_control/output_all.coffea")
# out_2022 = load("wjet_test_LO/output_all.coffea")

# # Merge outputs (assuming they have the same structure)
# merged_out = processor.accumulate([out_2017, out_2022])

# # Save merged output
# save(merged_out, "output_merged_mt_LO.coffea")




import os
import glob
from coffea.util import load, save
from coffea import processor

# Directory containing .coffea files
input_dir = "2018_cr"  # <-- Change this to your directory
output_file = input_dir+"/output_merged_"+input_dir+".coffea"

# Find all .coffea files in the directory
coffea_files = glob.glob(os.path.join(input_dir, "*.coffea"))

# Load all files
loaded_outputs = [load(f) for f in coffea_files]

# Merge outputs
merged_output = processor.accumulate(loaded_outputs)

# Save merged output
save(merged_output, output_file)

print(f"Merged {len(coffea_files)} files into {output_file}")
