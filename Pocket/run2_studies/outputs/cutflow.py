import matplotlib.pyplot as plt
import coffea.util
file = coffea.util.load("match_AN/output_all.coffea")
histos=file['sumw']
samples = list(histos['baseline'].keys())

import matplotlib.pyplot as plt

# Define your groups
groups = {
    "WJets": [
        "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8"
    ],
    "QCD": [
        "WplusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV",
        "WplusToLNuWplusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusTo2JZTo2LJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuWminusTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuZTo2JJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusTo2JWminusToLNuJJ_QCD_LO_SM_MJJ100PTJ10_TuneCP5_13TeV",
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
        "DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",
    ],
    "Multiboson": [
        "WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8",
        "WZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "ZZZ_TuneCP5_13TeV-amcatnlo-pythia8",
        "WGToLNuG_TuneCP5_13TeV-madgraphMLM-pythia8",
        "ZGToLLG_01J_5f_TuneCP5_13TeV-amcatnloFXFX-pythia8",
        "WZTo3LNu_mllmin01_NNPDF31_TuneCP5_13TeV_powheg_pythia8",
    ],
    "top": [
        "ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8",
        "ST_t-channel_top_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8",
        "ST_t-channel_antitop_4f_inclusiveDecays_TuneCP5_13TeV-powhegV2-madspin-pythia8",
    ],
    "WV_VBSx100": [
        "WplusTo2JWminusToLNuJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuWminusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuWplusTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WminusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
        "WplusToLNuZTo2JJJ_dipoleRecoil_EWK_LO_SM_MJJ100PTJ10_TuneCP5_13TeV-madgraph-pythia8",
    ],
}

# Reverse map for faster lookup: sample_name -> group_name
sample_to_group = {}
for group, sample_list in groups.items():
    for sample in sample_list:
        sample_to_group[sample] = group

# Initialize sums
group_baseline_sumw = {group: 0.0 for group in groups}
group_whad_sumw = {group: 0.0 for group in groups}

# Sum values in each group for baseline and whad_peak
for sample in histos['baseline']:
    # find group this sample belongs to
    # sample keys might not exactly match, so let's check which key contains the group sample string
    group = None
    for known_sample, g in sample_to_group.items():
        if known_sample in sample:
            group = g
            break
    if group is None:
        group = "Other"  # If you want to keep unknown samples separated
    
    # add to sums if baseline & whad_peak keys exist
    baseline_val = histos['baseline'][sample][list(histos['baseline'][sample].keys())[0]]
    whad_val = histos['whad_peak'][sample][list(histos['whad_peak'][sample].keys())[0]]
    
    if group not in group_baseline_sumw:
        group_baseline_sumw[group] = 0.0
        group_whad_sumw[group] = 0.0
    
    group_baseline_sumw[group] += baseline_val
    group_whad_sumw[group] += whad_val

# Sort groups for consistent plot order
plot_groups = sorted(group_baseline_sumw.keys())

# Prepare plot data
baseline_vals = [group_baseline_sumw[g] for g in plot_groups]
whad_vals = [group_whad_sumw[g] for g in plot_groups]

import numpy as np

fig, ax = plt.subplots(figsize=(10,6))

width = 0.4
indices = np.arange(len(plot_groups))

ax.bar(indices, baseline_vals, width=width, label='baseline')
ax.bar(indices + width, whad_vals, width=width, label='whad_peak')

ax.set_xticks(indices + width / 2)
ax.set_xticklabels(plot_groups, rotation=45, ha='right')
ax.set_yscale('log')
ax.set_ylabel('Sumw')
ax.set_title('Sumw by Group: baseline vs whad_peak')
ax.legend()

plt.tight_layout()
#fig.subplots_adjust(bottom=0.3)  # increase bottom margin for group labels

#plt.show()


#plt.show()

plt.savefig("cutflow.png")
