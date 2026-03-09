import json

def overwrite_xsec(source_json, target_json, output_json):
    with open(source_json) as f:
        source = json.load(f)

    with open(target_json) as f:
        target = json.load(f)

    common_samples = set(source.keys()) & set(target.keys())

    for sample in common_samples:
        try:
            # Take xsec from first file in source
            source_xsec = source[sample]["files"][0]["metadata"]["xsec"]

            # --- propagate to sample level (optional but recommended)
            # target[sample]["xsec"] = source_xsec

            # --- propagate to ALL file entries
            for file_entry in target[sample].get("files", []):
                file_entry.setdefault("metadata", {})
                file_entry["metadata"]["xsec"] = source_xsec

        except (KeyError, IndexError, TypeError):
            print(f"⚠️  Skipping {sample}: xsec path not found")

    with open(output_json, "w") as f:
        json.dump(target, f, indent=2)

    print(f"✅ Wrote updated JSON to {output_json}")



# example usage
overwrite_xsec(
    source_json="rare_2018.json_with_xsec.json",
    target_json="all_2016_bkg_with_xsec.json",
    output_json="all_2016_bkg_with_xsec.json"
)
