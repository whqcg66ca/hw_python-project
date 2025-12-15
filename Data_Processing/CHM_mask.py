from osgeo import gdal
import numpy as np
import os

# -------------------------------
# INPUT / OUTPUT
# -------------------------------
chm_path = r"N:\UAV Data_RSPLab Projects 2024\UAV Zenmuse P1 (RGB)_Processed\P1RGB_20240821_ForageSainfoin (2nd Cut)\CHM\CHM.tif"   # <-- change to your CHM path

out_path = os.path.join(
    os.path.dirname(chm_path),
    "CHM_masked.tif"
)

# -------------------------------
# READ CHM
# -------------------------------
ds = gdal.Open(chm_path, gdal.GA_ReadOnly)
if ds is None:
    raise RuntimeError("Cannot open CHM file")

band = ds.GetRasterBand(1)
chm = band.ReadAsArray().astype(np.float32)

nodata = band.GetNoDataValue()
if nodata is None:
    nodata = np.nan

# -------------------------------
# MASK VALUES
# -------------------------------
chm_masked = chm.copy()

# Mask invalid canopy heights
chm_masked[chm_masked < 0.01] = np.nan
chm_masked[chm_masked > 0.7]  = np.nan

# Replace NaN with NoData value for saving
out_nodata = -9999
chm_masked = np.where(np.isnan(chm_masked), out_nodata, chm_masked)

# -------------------------------
# WRITE OUTPUT
# -------------------------------
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(
    out_path,
    ds.RasterXSize,
    ds.RasterYSize,
    1,
    gdal.GDT_Float32,
    options=["COMPRESS=LZW"]
)

out_ds.SetGeoTransform(ds.GetGeoTransform())
out_ds.SetProjection(ds.GetProjection())

out_band = out_ds.GetRasterBand(1)
out_band.WriteArray(chm_masked)
out_band.SetNoDataValue(out_nodata)
out_band.FlushCache()

# Close datasets
ds = None
out_ds = None

print("Masked CHM saved to:")
print(out_path)
