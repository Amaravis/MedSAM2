import os
import re

src_dir = "valdo-gt"

for filename in os.listdir(src_dir):
    if filename.endswith("_space-T2S_CMB.nii.gz"):
        # Extract sub_num
        match = re.match(r"(sub-\d+)_space-T2S_CMB\.nii\.gz", filename)
        if match:
            sub_num = match.group(1)
            new_filename = f"{sub_num}_space-T2S_desc-masked_T2S.nii.gz"

            old_path = os.path.join(src_dir, filename)
            new_path = os.path.join(src_dir, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed: {filename} -> {new_filename}")