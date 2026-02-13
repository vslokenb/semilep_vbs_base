import json

def overwrite_xsec(source_json, target_json, output_json):
    with open(source_json) as f:
        source = json.load(f)

    with open(target_json) as f:
        target = json.load(f)

    common_samples = set(source.keys()) & set(target.keys())

    for sample in common_samples:
        try:
            source_xsec = source[sample]["files"][0]["metadata"]["xsec"]
            target[sample]["files"][0]["metadata"]["xsec"] = source_xsec
        except (KeyError, IndexError, TypeError):
            print(f"⚠️  Skipping {sample}: xsec path not found")

    with open(output_json, "w") as f:
        json.dump(target, f, indent=2)


# example usage
overwrite_xsec(
    source_json="v15_soup.json",
    target_json="2018_bkg.json",
    output_json="2018_bkg_with_xsec.json"
)
