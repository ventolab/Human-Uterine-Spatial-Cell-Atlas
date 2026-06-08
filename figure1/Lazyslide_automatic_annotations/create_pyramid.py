#!/usr/bin/env python3

import pyvips

# The path to your original, flat TIFF file
input_tiff_path = "/nfs/team292/vl6/Endometriosis/Xenium/A66-RPT-8-FO-1-S40/A66_crop.tif"

# The path where the new pyramidal TIFF will be saved
output_tiff_path = "/nfs/team292/vl6/Endometriosis/histology/A66_xenium/A66-RPT-8-FO-1-S40_pyramidal.tiff"

# Open the image and save it as a new pyramidal TIFF
image = pyvips.Image.new_from_file(input_tiff_path)

# Save the image with options to create a pyramid
# - tile=True: Enables tiling, which is essential for WSI formats
# - pyramid=True: Tells pyvips to generate the downsampled layers
# - compression="jpeg": Uses JPEG compression (common for WSI)
# - bigtiff=True: Allows the output file to be larger than 4GB
print(f"Converting {input_tiff_path} to pyramidal TIFF...")
image.tiffsave(
    output_tiff_path,
    tile=True,
    pyramid=True,
    compression="jpeg",
    bigtiff=True
)
print(f"Successfully saved pyramidal TIFF to {output_tiff_path}")
