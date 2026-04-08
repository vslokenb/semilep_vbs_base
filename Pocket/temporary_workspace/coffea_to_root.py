import coffea.util
import uproot
import numpy as np
from boost_histogram import Histogram

def convert_boost_hist_to_root(h, name):
    """
    Convert a boost_histogram Histogram into a dict for uproot writing
    """

    axes_types = [type(axis).__name__ for axis in h.axes]
    ndim = axes_types.count('Variable')
    
    hist_dict = {}
    cat_names = list(h.axes[0])
    if "Muon" in name or "EGamma" in name:
        print(cat_names)
        for icat in range(len(h.axes[0].edges)-1):
            if ndim == 1:
                print()
                hist_dict[name+"_"+cat_names[icat]] = h[icat,:]
            elif ndim == 2:
                hist_dict[name+"_"+cat_names[icat]] = h[icat,:,:]
            else:
                raise NotImplementedError("Only 1D and 2D histograms are supported.")
    else:
        ndim = sum(1 for t in axes_types if t in ('Variable', 'Regular'))
        unc_names = list(h.axes[1])
        print(cat_names,unc_names)
        for icat in range(len(h.axes[0].edges)-1):
            for iunc in range(len(h.axes[1].edges)-1):
                if ndim == 1:
                    hist_dict[name+"_"+cat_names[icat]+"_"+unc_names[iunc]] = h[icat,iunc,:] 
                elif ndim == 2:
                    hist_dict[name+"_"+cat_names[icat]+"_"+unc_names[iunc]] = h[icat,iunc,:,:]
                else:
                    raise NotImplementedError("Only 1D and 2D histograms are supported.")
    return hist_dict


def coffea_to_root(infile, outfile):
    # Load coffea file
    hists = coffea.util.load(infile)

    # Create ROOT file
    with uproot.recreate(outfile) as rootfile:

        for name, hist in hists.items():
            if isinstance(hist, Histogram):
                # boost_histogram object → ROOT TH1/TH2
                root_hists = convert_boost_hist_to_root(hist, name)
                for hist_name,root_hist in root_hists.items():
                    rootfile[hist_name] = root_hist

            else:
                print(f"Skipping {name}: not a histogram")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert coffea file to ROOT histograms")
    parser.add_argument("--input", help=".coffea file to convert")
    parser.add_argument("--output", help="Output ROOT file")
    args = parser.parse_args()

    coffea_to_root(args.input, args.output)

