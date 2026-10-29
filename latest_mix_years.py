import os, glob
from coffea.util import load, save
from coffea import processor

input_dir = "latest"
tmp_dir = f"{input_dir}/_merge_tmp"
os.makedirs(tmp_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(input_dir, "*.coffea")))

round_id = 0
while len(files) > 1:
    new_files = []
    print(f"\n=== Merge round {round_id} ({len(files)} files) ===")

    for i in range(0, len(files), 2):
        f1 = files[i]
        if i + 1 < len(files):
            f2 = files[i + 1]

            print(f"Merging:\n  {f1}\n  {f2}")
            a = load(f1)
            b = load(f2)
            merged = processor.accumulate([a, b])

            out = f"{tmp_dir}/round{round_id}_{i//2}.coffea"
            save(merged, out)

            del a, b, merged
            new_files.append(out)
        else:
            # odd one out, carry forward
            new_files.append(f1)

    files = new_files
    round_id += 1

final_output = f"{input_dir}/output_merged_{input_dir}.coffea"
os.rename(files[0], final_output)

print(f"\nFinal merged file: {final_output}")
