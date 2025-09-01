from coffea.util import load, save
from coffea import processor

# Load files
out_2017 = load("data_control/output_all.coffea")
out_2022 = load("wjet_test_LO/output_all.coffea")

# Merge outputs (assuming they have the same structure)
merged_out = processor.accumulate([out_2017, out_2022])

# Save merged output
save(merged_out, "output_merged_mt_LO.coffea")