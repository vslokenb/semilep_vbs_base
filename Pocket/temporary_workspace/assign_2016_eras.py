import json

with open("all_2016_bkg_with_xsec.json") as f:
    data = json.load(f)

for sample_name, sample_block in data.items():
    for file_block in sample_block.get("files", []):
        das_list = file_block.get("das_names", [])

        # Join in case there are multiple entries
        das_string = " ".join(das_list)

        if "preVFP" in das_string:
            year_value = "2016_PreVFP"
        else:
            year_value = "2016_PostVFP"

        file_block["metadata"]["year"] = year_value

with open("all_2016_erabkg_with_xsec.json", "w") as f:
    json.dump(data, f, indent=2)
