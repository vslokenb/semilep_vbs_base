import coffea.util
import os
import glob
import argparse
from collections import defaultdict

def compute_total_uncertainty_per_category(hist, process_name=None, summary_accumulator=None):
    variations = list(hist.axes["variation"])
    categories = list(hist.axes["cat"])

    print(f"\n=== Process: {process_name} ===")
    print(f"{'Category':30} {'Yield':>12} {'Stat unc':>10} {'Syst Up':>10} {'Syst Down':>12} {'Total Up':>10} {'Total Down':>12}")
    print("-" * 100)

    for cat in categories:
        nominal_hist = hist[{"variation": "nominal", "cat": cat}].sum(flow=True)
        nominal_yield = nominal_hist.value
        nominal_stat_unc = nominal_hist.variance ** 0.5

        syst_up_sq = 0.0
        syst_down_sq = 0.0

        for var in variations:
            if var == "nominal":
                continue

            var_hist = hist[{"variation": var, "cat": cat}].sum(flow=True)
            var_unc = var_hist.variance ** 0.5

            if var.endswith("Up"):
                syst_up_sq += var_unc ** 2
            elif var.endswith("Down"):
                syst_down_sq += var_unc ** 2
            else:
                # Treat unknown or symmetric as both
                syst_up_sq += var_unc ** 2
                syst_down_sq += var_unc ** 2

        syst_up = syst_up_sq ** 0.5
        syst_down = syst_down_sq ** 0.5
        total_up = (nominal_stat_unc ** 2 + syst_up ** 2) ** 0.5
        total_down = (nominal_stat_unc ** 2 + syst_down ** 2) ** 0.5

        print(f"{cat:30} {nominal_yield:12.2f} {nominal_stat_unc:10.2f} {syst_up:10.2f} {syst_down:12.2f} {total_up:10.2f} {total_down:12.2f}")

        # Accumulate squares of uncertainties per category
        if summary_accumulator is not None:
            summary_accumulator[cat]["stat_sq"] += nominal_stat_unc ** 2
            summary_accumulator[cat]["syst_up_sq"] += syst_up ** 2
            summary_accumulator[cat]["syst_down_sq"] += syst_down ** 2

def print_combined_uncertainties(summary_accumulator):
    print(f"\n=== Combined Uncertainties Across All Processes ===")
    print(f"{'Category':30} {'Stat unc':>10} {'Syst Up':>10} {'Syst Down':>12} {'Total Up':>10} {'Total Down':>12}")
    print("-" * 90)

    for cat, acc in summary_accumulator.items():
        stat = acc["stat_sq"] ** 0.5
        syst_up = acc["syst_up_sq"] ** 0.5
        syst_down = acc["syst_down_sq"] ** 0.5
        total_up = (acc["stat_sq"] + acc["syst_up_sq"]) ** 0.5
        total_down = (acc["stat_sq"] + acc["syst_down_sq"]) ** 0.5

        print(f"{cat:30} {stat:10.2f} {syst_up:10.2f} {syst_down:12.2f} {total_up:10.2f} {total_down:12.2f}")

def main():
    parser = argparse.ArgumentParser(description="Compute combined category uncertainties across matched .coffea files")
    parser.add_argument("directory", help="Directory containing output_{process}.coffea files")
    parser.add_argument("variable_key", help="Variable key under ['variables'], e.g. 'HT_check'")
    parser.add_argument("wildcard", help="Wildcard for process names, e.g. 'WJetsToLNu_HT-*'")
    args = parser.parse_args()

    pattern = os.path.join(args.directory, f"output_{args.wildcard}.coffea")
    matched_files = glob.glob(pattern)

    if not matched_files:
        print(f"No files matched pattern: {pattern}")
        return

    # Accumulator: category → dict of squared uncertainties
    summary_accumulator = defaultdict(lambda: {
        "stat_sq": 0.0,
        "syst_up_sq": 0.0,
        "syst_down_sq": 0.0,
    })

    for filepath in matched_files:
        filename = os.path.basename(filepath)
        if not filename.startswith("output_") or not filename.endswith(".coffea"):
            continue

        process = filename[len("output_"):-len("_2017.coffea")]

        try:
            file = coffea.util.load(filepath)
            hist = file["variables"][args.variable_key][process][process + "_2017"]
        except KeyError as e:
            print(f"❌ KeyError accessing histogram in {filename}: {e}")
            continue

        compute_total_uncertainty_per_category(hist, process_name=process, summary_accumulator=summary_accumulator)

    # Print final combined summary
    print_combined_uncertainties(summary_accumulator)

if __name__ == "__main__":
    main()
