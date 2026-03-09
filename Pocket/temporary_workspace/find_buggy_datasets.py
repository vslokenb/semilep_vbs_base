import uproot
import numpy as np
import os

# === User settings ===
directory = "/eos/cms/store/data/Run2017B/SingleElectron/NANOAOD/UL2017_MiniAODv2_NanoAODv9-v1/70000/"   # Directory containing ROOT files
tree_name = "Events"         # Name of the TTree inside each ROOT file
branch_name = "Electron_pt"       # Example variable
threshold = 35.0             # Example threshold (GeV)

# === Counters ===
total_events = 0
failed_events = 0

# === Loop over ROOT files ===
for filename in os.listdir(directory):
    if not filename.endswith(".root"):
        continue

    filepath = os.path.join(directory, filename)
    print(f"Processing {filepath}...")

    with uproot.open(filepath) as file:
        tree = file[tree_name]

        # jet_pt is jagged (one array per event)
        jet_pt = tree[branch_name].array(library="np")

        # Count how many jets per event pass the cut
        passes = [np.any(jets > threshold) for jets in jet_pt]

        n_events = len(passes)
        n_failed = np.sum(~np.array(passes))

        total_events += n_events
        failed_events += n_failed

print("\n=== Summary ===")
print(f"Total events: {total_events}")
print(f"Events failing (no entry passes cut): {failed_events}")
print(f"Fraction failed: {failed_events / total_events:.3f}")
