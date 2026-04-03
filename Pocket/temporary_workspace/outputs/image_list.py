import os
import json

def generate_image_list(directory="."):
    exts = (".png", ".jpg", ".jpeg", ".svg")
    images = sorted(
        [f for f in os.listdir(directory) if f.lower().endswith(exts)]
    )

    outpath = os.path.join(directory, "images.json")
    with open(outpath, "w") as f:
        json.dump(images, f, indent=2)
    print(f"✅ Found {len(images)} images, wrote {outpath}")

if __name__ == "__main__":
    generate_image_list("/afs/cern.ch/user/v/vslokenb/vbs_semilep/semilep_vbs_base/Pocket/temporary_workspace/outputs/data_mc_ht_v4_full/plots/baseline")  # Change to your plot directory
