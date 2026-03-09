import coffea.util
import os
import glob
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from coffea.util import load
import numpy as np

def extract_yields_from_file(filename, variable='fj_W_vs_QCD', category='boosted_jet_in_window_mu', variation='nominal'):
    """
    Reads a Coffea histogram file and extracts per-bin yields for all processes
    for a given variable, category, and variation.
    
    Parameters
    ----------
    filename : str
        Path to the .coffea file
    variable : str
        Variable name inside histos['variables'] (e.g. 'HT_check')
    category : str, optional
        Category name (default: 'baseline')
    variation : str, optional
        Variation name (default: 'nominal')
    
    Returns
    -------
    dict
        { process_name: {'bin_edges': [...], 'yields': [...], 'errors': [...]} }
    """
    
    # --- Load the Coffea file
    histos = load(filename)
    output = {}
    
    # Access the variable level (e.g. histos['variables']['HT_check'])
    var_dict = histos['variables'][variable]
    
    # Loop over all processes (e.g. WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8)
    for proc_name, proc_dict in var_dict.items():
        for sample_name, hist in proc_dict.items():
            
            # Slice histogram
            try:
                h_sel = hist[category, variation, :]
            except Exception as e:
                print(f"⚠️ Skipping {sample_name}: {e}")
                continue

            # Extract data
            values = h_sel.values()
            variances = h_sel.variances()
            edges = h_sel.axes['candidate_boost.particleNet_WvsQCD'].edges

            # Store results
            output[sample_name] = {
                "bin_edges": edges.tolist(),
                "yields": values.tolist(),
                "errors": np.sqrt(variances).tolist() if variances is not None else None,
            }

    return output

def calculate_signal_strength(full_dict, n_bins=0):
    background = 0
    signal = 0
    for i in full_dict:
        events = np.sum(full_dict[i]['yields'][-n_bins:])
        if "EWK" in i:
            signal += events
            print("signal found! is ", i)
        else:
            background += events
    print("signal: ", signal)
    print("background: ",background)
    return signal/np.sqrt(background)
    
group_of_samples= extract_yields_from_file("outputs/stable_UL_v0/output_merged_stable_UL_v0.coffea")

strength=[]

for i in range(32):
    calculate=calculate_signal_strength(group_of_samples, i)
    print("S/sqrt(B) = ", calculate)
    strength.append(calculate)

bin_edges = np.linspace(1.1, 0, 33)
discrim_score = 0.5 * (bin_edges[:-1] + bin_edges[1:])
max_idx = np.nanargmax(strength)
x_max = discrim_score[max_idx]
y_max = strength[max_idx]

# Create plot
plt.figure(figsize=(7,5))
plt.scatter(discrim_score, strength, color='royalblue', s=50, label='Signal strength')
#plt.plot(discrim_score, strength, color='royalblue', alpha=0.5)
print(len(discrim_score))
# Highlight the maximum point
plt.axvline(x_max, color='red', linestyle='--', alpha=0.7)
plt.axhline(y_max, color='red', linestyle='--', alpha=0.7)
plt.scatter(x_max, y_max, color='darkred', s=80, zorder=5)

# Annotate the max point

plt.text(
    x_max + 0.02, y_max,
    f"max = ({x_max:.2f}, {y_max:.2f})",
    color='darkred',
    fontsize=10,
    verticalalignment='bottom'
)

# Formatting
plt.xlim(0, 1)
plt.ylim(0, 1.05 * max(strength))
plt.xlabel("pNet W vs QCD discrim score", fontsize=12)
plt.ylabel("Signal strength", fontsize=12)
plt.title("Cut score optimization WW", fontsize=14)
plt.grid(True, linestyle='--', alpha=0.4)

# Nice tick marks
plt.xticks(np.linspace(0, 1, 10))
plt.yticks(np.linspace(0, round(max(strength), 2), 6))

plt.tight_layout()
#plt.show()
plt.savefig("optimize_W_vs_QCD.png")
