import coffea.util
import uproot
import numpy as np
from boost_histogram import Histogram

def convert_boost_hist_to_root(h, name):
    axes_types = [type(axis).__name__ for axis in h.axes]
    cat_names = list(h.axes[0])

    # Detect whether axis[1] is a StrCategory (variation) or a physics axis
    has_variation_axis = (len(h.axes) > 1 and type(h.axes[1]).__name__ == 'StrCategory')

    # Count physics (Variable/Regular) axes only
    physics_axes = [t for t in axes_types if t in ('Variable', 'Regular')]
    ndim = len(physics_axes)

    hist_dict = {}
    ncat = len(cat_names)

    if has_variation_axis:
        unc_names = list(h.axes[1])
        nunc = len(unc_names)
        for icat in range(ncat):
            for iunc in range(nunc):
                key = f"{name}_{cat_names[icat]}_{unc_names[iunc]}"
                if ndim == 1:
                    hist_dict[key] = h[icat, iunc, :]
                elif ndim == 2:
                    hist_dict[key] = h[icat, iunc, :, :]
                elif ndim == 3:
                    hist_dict[key] = h[icat, iunc, :, :, :]
                else:
                    raise NotImplementedError(f"Unsupported ndim={ndim}")
    else:
        for icat in range(ncat):
            key = f"{name}_{cat_names[icat]}"
            if ndim == 1:
                hist_dict[key] = h[icat, :]
            elif ndim == 2:
                hist_dict[key] = h[icat, :, :]
            elif ndim == 3:
                hist_dict[key] = h[icat, :, :, :]
            else:
                raise NotImplementedError(f"Unsupported ndim={ndim}")

    return hist_dict


def iter_histograms(obj, prefix=""):
    """
    Recursively walk a coffea accumulator and yield (name, Histogram) pairs.

    Handles both the flat top-level layout ({hname: Histogram}) and the
    nested 'variables' layout ({hname: {sample: {dataset_era: Histogram}}}).
    Container keys are joined with '_' to build a unique name; the literal
    'variables' key is dropped from the prefix to keep names clean.
    """
    if isinstance(obj, Histogram):
        yield prefix, obj
        return
    if isinstance(obj, dict):
        for key, val in obj.items():
            if prefix == "" and key == "variables":
                child_prefix = ""          # don't pollute names with the container name
            elif prefix == "":
                child_prefix = str(key)
            else:
                child_prefix = f"{prefix}_{key}"
            yield from iter_histograms(val, child_prefix)


def coffea_to_root(infile, outfile):
    # Load coffea file
    hists = coffea.util.load(infile)

    # Create ROOT file
    with uproot.recreate(outfile) as rootfile:

        found = False
        for name, hist in iter_histograms(hists):
            found = True
            # boost_histogram object → ROOT TH1/TH2/TH3
            root_hists = convert_boost_hist_to_root(hist, name)
            for hist_name, root_hist in root_hists.items():
                rootfile[hist_name] = root_hist

        if not found:
            print("No boost_histogram objects found in", infile)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert coffea file to ROOT histograms")
    parser.add_argument("--input", help=".coffea file to convert")
    parser.add_argument("--output", help="Output ROOT file")
    args = parser.parse_args()

    coffea_to_root(args.input, args.output)