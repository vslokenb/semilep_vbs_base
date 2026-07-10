from coffea.util import load, save
import hist

def fix_axis_label(histo):
    new_axes = []
    for ax in histo.axes:
        if hasattr(ax, 'label') and "_{miss}" in str(ax.label):
            new_axes.append(hist.axis.Regular(
                20, 0, 100,
                name=ax.name,
                label=r"$p_{T}^{\mathrm{miss}}$ [GeV]"
            ))
        else:
            new_axes.append(ax)
    h_fixed = hist.Hist(*new_axes, storage=histo.storage_type())
    h_fixed.view(flow=True)[:] = histo.view(flow=True)
    return h_fixed


output = load("output_merged_nonprompt_FR_mT20.coffea")

h_dict = output["variables"]["pt_miss"]
for sample_key in h_dict:
    print(sample_key)
    for key in h_dict[sample_key]:
        print(key)
        h_dict[sample_key][key] = fix_axis_label(h_dict[sample_key][key])

output["variables"]["pt_miss"] = h_dict
save(output, "output_merged_nonprompt_FR_mT20_fixed.coffea")
print("Done")