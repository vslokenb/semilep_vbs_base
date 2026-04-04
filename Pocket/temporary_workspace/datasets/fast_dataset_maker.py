import json
import os
import copy

SAMPLES = [
    "TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8",
    "TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8",
    "WJetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8",
    "WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8",
    "DYJetsToLL_M-10to50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
    "DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8",
]

# Deduplicate while preserving order
SAMPLES = list(dict.fromkeys(SAMPLES))

XSEC_SCALE = 0.0344
TARGET_YEAR = "2016_PostVFP"
TARGET_ERA = "F"

INPUT_DIR = "."        # Directory containing the original .json files
OUTPUT_DIR = "./filtered"  # Directory to write edited copies into

os.makedirs(OUTPUT_DIR, exist_ok=True)

for sample in SAMPLES:
    input_path = os.path.join(INPUT_DIR, f"{sample}.json")

    if not os.path.exists(input_path):
        print(f"[WARN] File not found, skipping: {input_path}")
        continue

    with open(input_path) as f:
        data = json.load(f)

    filtered = {}
    for key, value in data.items():
        meta = value.get("metadata", {})
        year = meta.get("year", "")
        era  = meta.get("era", "")   # may or may not exist

        # Keep if year matches AND
        #   - 'era' field == TARGET_ERA, OR
        #   - key ends with _F (e.g. Sample_2016_PostVFP_F), OR
        #   - no era field at all but year matches (fallback — remove if too broad)
        year_match = (year == TARGET_YEAR)
        era_match  = (era == TARGET_ERA) or key.endswith(f"_{TARGET_ERA}")

        # If no era metadata and no era suffix, still keep if year matches
        # (covers files that don't encode era separately — adjust as needed)
        has_era_info = bool(era) or any(key.endswith(f"_{e}") for e in "BCDEFGH")
        if year_match and (era_match or not has_era_info):
            entry = copy.deepcopy(value)
            # Scale xsec
            original_xsec = float(entry["metadata"]["xsec"])
            new_xsec = original_xsec * XSEC_SCALE
            entry["metadata"]["xsec"] = str(new_xsec)
            entry["metadata"]["xsec_note"] = (
                f"original={original_xsec}, scaled by {XSEC_SCALE}"
            )
            filtered[key] = entry

    if not filtered:
        print(f"[WARN] No matching sections found in: {sample}.json")
        continue

    output_path = os.path.join(OUTPUT_DIR, f"{sample}_fast.json")
    with open(output_path, "w") as f:
        json.dump(filtered, f, indent=4)

    for key in filtered:
        orig = float(filtered[key]["metadata"]["xsec_note"].split("original=")[1].split(",")[0])
        new  = float(filtered[key]["metadata"]["xsec"])
        print(f"[OK] {sample}  |  key: {key}  |  xsec: {orig} → {new:.6g}")