import os
import sys

# Optional but sometimes helps VS Code find the SNAP Python module if import fails
# sys.path.append(r"C:\Users\HONGQUAN\miniconda3\envs\SAR\Lib\site-packages")

from esa_snappy import ProductIO, GPF, HashMap

# SNAP operators are registered lazily; this loads them
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()

in_prod  = r"H:\7_Sentinel\South2023_zip\S1A_IW_SLC__1SDV_20230503T230909_20230503T230936_048376_05D190_AD33.zip"    # or .SAFE folder, or .zip
out_prod = r"H:\7_Sentinel\South2023\test\S1A_20230503_TC.dim"

print("Reading product...")
p = ProductIO.readProduct(in_prod)

# Example 1: Apply Orbit File
orbit_params = HashMap()
orbit_params.put("Apply-Orbit-File", True)  # (SNAP sometimes ignores this key; safe to omit)
p_orbit = GPF.createProduct("Apply-Orbit-File", orbit_params, p)

# Example 2: Calibration (Sigma0)
cal_params = HashMap()
cal_params.put("outputSigmaBand", True)
cal_params.put("sourceBands", "Intensity_VV")   # adjust if your band name differs
p_cal = GPF.createProduct("Calibration", cal_params, p_orbit)

# Example 3: Terrain Correction
tc_params = HashMap()
tc_params.put("demName", "SRTM 3Sec")
tc_params.put("pixelSpacingInMeter", 10.0)
tc_params.put("mapProjection", "AUTO:42001")  # auto UTM
p_tc = GPF.createProduct("Terrain-Correction", tc_params, p_cal)

print("Writing product...")
ProductIO.writeProduct(p_tc, out_prod, "BEAM-DIMAP")  # outputs .dim + .data

print("Done:", out_prod)
