import os
import numpy as np
from osgeo import gdal

gdal.UseExceptions()

def downscale_dsm_to_dtm_grid(dsm_path, dtm_path, dsm_matched_path,
                             resample_alg="average"):
    """
    Resample (downscale) DSM to match DTM grid exactly (extent, resolution, size, projection).
    resample_alg: "average" (recommended for downscaling), or "bilinear", "cubic", etc.
    """
    dtm_ds = gdal.Open(dtm_path, gdal.GA_ReadOnly)
    if dtm_ds is None:
        raise RuntimeError(f"Cannot open DTM: {dtm_path}")

    gt = dtm_ds.GetGeoTransform()
    proj = dtm_ds.GetProjection()
    xsize = dtm_ds.RasterXSize
    ysize = dtm_ds.RasterYSize

    # DTM grid properties
    xres = gt[1]
    yres = abs(gt[5])

    # DTM bounds
    xmin = gt[0]
    ymax = gt[3]
    xmax = xmin + xsize * xres
    ymin = ymax - ysize * yres

    # Map algorithm name to GDAL constant
    alg_map = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "cubic": gdal.GRA_Cubic,
        "cubicspline": gdal.GRA_CubicSpline,
        "lanczos": gdal.GRA_Lanczos,
        "average": gdal.GRA_Average,
        "mode": gdal.GRA_Mode,
        "max": gdal.GRA_Max,
        "min": gdal.GRA_Min,
        "med": gdal.GRA_Med,
        "q1": gdal.GRA_Q1,
        "q3": gdal.GRA_Q3,
    }
    if resample_alg.lower() not in alg_map:
        raise ValueError(f"Unsupported resample_alg: {resample_alg}")

    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=proj,
        outputBounds=(xmin, ymin, xmax, ymax),
        width=xsize,
        height=ysize,
        resampleAlg=alg_map[resample_alg.lower()],
        multithread=True,
        creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    )

    # Warp DSM into DTM grid
    out_ds = gdal.Warp(dsm_matched_path, dsm_path, options=warp_options)
    if out_ds is None:
        raise RuntimeError("gdal.Warp failed.")
    out_ds = None
    dtm_ds = None


def compute_chm(dsm_matched_path, dtm_path, chm_path,
                set_negative_to_zero=True, out_nodata=-9999):
    """
    CHM = DSM_resampled - DTM
    """
    dsm_ds = gdal.Open(dsm_matched_path, gdal.GA_ReadOnly)
    dtm_ds = gdal.Open(dtm_path, gdal.GA_ReadOnly)
    if dsm_ds is None or dtm_ds is None:
        raise RuntimeError("Cannot open DSM_matched or DTM.")

    if (dsm_ds.RasterXSize != dtm_ds.RasterXSize) or (dsm_ds.RasterYSize != dtm_ds.RasterYSize):
        raise RuntimeError("DSM_matched and DTM sizes do not match. Resampling step failed?")

    dsm_band = dsm_ds.GetRasterBand(1)
    dtm_band = dtm_ds.GetRasterBand(1)

    dsm_nodata = dsm_band.GetNoDataValue()
    dtm_nodata = dtm_band.GetNoDataValue()

    dsm = dsm_band.ReadAsArray().astype(np.float32)
    dtm = dtm_band.ReadAsArray().astype(np.float32)

    valid = np.ones(dsm.shape, dtype=bool)
    if dsm_nodata is not None:
        valid &= (dsm != dsm_nodata)
    if dtm_nodata is not None:
        valid &= (dtm != dtm_nodata)

    chm = np.full(dsm.shape, out_nodata, dtype=np.float32)
    chm[valid] = dsm[valid] - dtm[valid]

    if set_negative_to_zero:
        chm[valid] = np.maximum(chm[valid], 0.0)

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        chm_path,
        dsm_ds.RasterXSize,
        dsm_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"]
    )
    out_ds.SetGeoTransform(dsm_ds.GetGeoTransform())
    out_ds.SetProjection(dsm_ds.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(out_nodata)
    out_band.WriteArray(chm)
    out_band.FlushCache()

    out_ds = None
    dsm_ds = None
    dtm_ds = None


if __name__ == "__main__":
    # ---- EDIT THESE PATHS ----
    dsm_path = r"N:\UAV Data_RSPLab Projects 2024\UAV Zenmuse P1 (RGB)_Processed\P1RGB_20240821_ForageSainfoin (2nd Cut)\CHM\Forage20240821_dsm.tif"
    dtm_path = r"N:\UAV Data_RSPLab Projects 2024\UAV Zenmuse P1 (RGB)_Processed\P1RGB_20240821_ForageSainfoin (2nd Cut)\CHM\Forage20240821_dtm.tif"
    dsm_matched_path = r"N:\UAV Data_RSPLab Projects 2024\UAV Zenmuse P1 (RGB)_Processed\P1RGB_20240821_ForageSainfoin (2nd Cut)\CHM\DSM_downscaled_to_DTM.tif"
    chm_path = r"N:\UAV Data_RSPLab Projects 2024\UAV Zenmuse P1 (RGB)_Processed\P1RGB_20240821_ForageSainfoin (2nd Cut)\CHM\CHM.tif"
    # -------------------------

    # 1) Downscale DSM to match DTM grid
    downscale_dsm_to_dtm_grid(dsm_path, dtm_path, dsm_matched_path,
                              resample_alg="average")  # good for downscaling

    # 2) CHM = DSM_matched - DTM
    compute_chm(dsm_matched_path, dtm_path, chm_path,
                set_negative_to_zero=True, out_nodata=-9999)

    print("Done.")
    print("DSM matched:", dsm_matched_path)
    print("CHM:", chm_path)
