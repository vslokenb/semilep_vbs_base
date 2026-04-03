import matplotlib.pyplot as plt
import coffea.util
import numpy as np

# Load file
file = coffea.util.load("/eos/user/v/vslokenb/vbs_semilep/outputs/check_signal_yield/output_merged_check_signal_yield.coffea")
histos = file['sumw']

# Define groups
groups = {
    "WJets_NLO": ["WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8"],
    "QCD": [
        "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "ZTo2LZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    ],
    "Ttbar": [
        "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
        "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
        "ttZJets_TuneCP5_13TeV_madgraphMLM_pythia8",
        "ttWJets_TuneCP5_13TeV_madgraphMLM_pythia8",
    ],
    "DY": [
        "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8_17",
        "DYJetsToLL_M-50_HT-70to100_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",                                                                                                           
        "DYJetsToLL_M-50_HT-100to200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",                                                                                                          
        "DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",                                                                                                          
        "DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",                                                                                                          
        "DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",                                                                                                          
        "DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",                                                                                                         
        "DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8",                                                                                                        
        "DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8", 
        "DYJetsToLL_M-4to50_HT-100to200_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-4to50_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-4to50_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8",
        "DYJetsToLL_M-4to50_HT-600toInf_TuneCP5_13TeV-madgraphMLM-pythia8",
    ],
    "triboson and gluglu": [
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "WWZ_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "GluGluWWToLNuQQ_TuneCP5_13TeV_madgraph-pythia8",
    ],
    "Vgamma": [
        "WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8",
        "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
    ],
    "top": [
        "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
        "ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
        "ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8",
        "ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
        "ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8",
    ],
    "WV_VBS": [
        "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusTo2JZTo2LJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "ZTo2LZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    ],
    "WJets_LO_HT": [
        "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
        "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
    ]
}

# Map sample to group
sample_to_group = {sample: group for group, samples in groups.items() for sample in samples}

# All categories present in the file
categories = list(histos.keys())

# Data structure: group -> category -> sumw
group_category_sumw = {}

# Loop over all samples
# Just to get sample list
for sample in histos[categories[0]]:
    group = next((g for s, g in sample_to_group.items() if s in sample), "Other")
    print(sample, "==========", group)

    if group not in group_category_sumw:
        group_category_sumw[group] = {cat: 0.0 for cat in categories}

    for cat in categories:
        # histos['sumw'][category][sample]
        sample_dict = histos[cat][sample]

        # one base-sample level
        base_sample = next(iter(sample_dict))

        # nominal systematic
        val = sample_dict[base_sample].get('nominal', 0.0)

        group_category_sumw[group][cat] += val



# Only keep non-empty groups
plot_groups = sorted([
    group for group, cat_vals in group_category_sumw.items()
    if any(val > 0 for val in cat_vals.values())
])

# Prepare plotting data
category_colors = plt.cm.tab10.colors  # Or choose a different colormap
category_to_color = {cat: category_colors[i % len(category_colors)] for i, cat in enumerate(categories)}

indices = np.arange(len(plot_groups))
width = 0.6 / len(categories)  # adjust width based on number of categories

fig, ax = plt.subplots(figsize=(12, 6))

for i, cat in enumerate(categories):
    vals = [group_category_sumw[group][cat] for group in plot_groups]
    ax.bar(indices + i * width, vals, width=width, label=cat, color=category_to_color[cat])

# Axis settings
ax.set_xticks(indices + (len(categories) - 1) * width / 2)
ax.set_xticklabels(plot_groups, rotation=45, ha='right')
ax.set_ylabel("Yield")
ax.set_yscale("log")
ax.set_title("Cutflow (Dynamic Categories)")
ax.yaxis.grid(True, linestyle="--", alpha=0.5)

ax.legend()
plt.tight_layout()
plt.savefig("cutflow_play_clean_v4.png")


# Print all yields (grouped by category and group)
print("\n=== YIELDS PER CATEGORY AND GROUP ===\n")
header = f"{'Group':<20}" + "".join([f"{cat:<25}" for cat in categories])
print(header)
print("-" * len(header))

for group in plot_groups:
    row = f"{group:<20}"
    for cat in categories:
        val = group_category_sumw[group][cat]
        row += f"{val:<25.2f}"
    print(row)

print("\nNote: Values are raw 'sumw' yields from histos.\n")
