# convert_snap_img_to_tif_gdal.py
import os
import sys

from osgeo import gdal

def convert_img_folder_to_tif(in_folder: str, out_folder: str, compress: str = "LZW") -> None:
    if not os.path.isdir(in_folder):
        raise FileNotFoundError(in_folder)

    os.makedirs(out_folder, exist_ok=True)

    # Be explicit about driver registration
    gdal.AllRegister()

    imgs = sorted([f for f in os.listdir(in_folder) if f.lower().endswith(".img")])
    print(f"Found {len(imgs)} .img files")

    for fn in imgs:
        # ✅ SKIP elevation raster
        if "elevation" in fn.lower():
             continue

        src = os.path.join(in_folder, fn)
        base = os.path.splitext(fn)[0]
        dst = os.path.join(out_folder, base + ".tif")

        if os.path.exists(dst):
            print(f"Skip (exists): {dst}")
            continue

        print(f"Converting: {fn}")

        ds = gdal.Open(src, gdal.GA_ReadOnly)
        if ds is None:
            print(f"  ERROR: GDAL cannot open: {src}")
            continue

        # GeoTIFF creation options
        # - COMPRESS=LZW is safe
        # - TILED=YES helps performance
        # - BIGTIFF=IF_SAFER avoids >4GB failure
        creation_opts = [
            f"COMPRESS={compress}",
            "TILED=YES",
            "BIGTIFF=IF_SAFER",
        ]

        # Translate (convert) to GeoTIFF
        out_ds = gdal.Translate(
            destName=dst,
            srcDS=ds,
            format="GTiff",
            creationOptions=creation_opts
        )

        # Close datasets to release file handles
        out_ds = None
        ds = None

        if os.path.exists(dst):
            print(f"  -> OK: {dst}")
        else:
            print(f"  ERROR: output not created: {dst}")



if __name__ == "__main__":
    dd=r"20231030"
    in_folder = "D:\\7_Sentinel\\North2023_Re\\" + dd + "\\S1A_" + dd + "_TC.data"
    zonal_folder = os.path.join(r"D:\7_Sentinel\North2023_Re\\" + dd + "\\", "Zonal")
    if not os.path.isdir(zonal_folder):
        os.makedirs(zonal_folder)
    out_folder = zonal_folder

    try:
        convert_img_folder_to_tif(in_folder, out_folder)
        print("\nAll conversions finished.")
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
