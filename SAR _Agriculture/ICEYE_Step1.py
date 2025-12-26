import os
from esa_snappy import ProductIO, GPF, HashMap, jpy

# --- IMPORTANT: load SNAP operators ---
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()


def _list_bands(product, header="Available bands"):
    band_names = [product.getBandAt(i).getName() for i in range(product.getNumBands())]
    print(f"   {header} ({len(band_names)}): {band_names}")
    return band_names


def calibrate_sigma0_grd(product, pol="VV"):
    """
    ICEYE GRD: Radiometric calibration -> Sigma0_<pol> (linear)
    """
    band_names = _list_bands(product, header="Input bands")

    # Try to find a reasonable "source band" for calibration.
    # ICEYE/SAR readers vary; common keywords: intensity, amplitude, sigma0, gamma0.
    src_candidates = [b for b in band_names if "intensity" in b.lower()]
    if not src_candidates:
        src_candidates = [b for b in band_names if "amplitude" in b.lower()]
    if not src_candidates:
        # If product already has sigma/gamma, calibration may still work, but we try anyway.
        src_candidates = [b for b in band_names if ("sigma0" in b.lower() or "gamma0" in b.lower())]

    if not src_candidates:
        raise RuntimeError(
            "No suitable source band found for Calibration. "
            "Please check the printed band names and adjust the source band selection."
        )

    params = HashMap()
    params.put("outputSigmaBand", True)
    params.put("outputBetaBand", False)
    params.put("outputGammaBand", False)
    params.put("selectedPolarisations", pol)  # "VV"
    params.put("sourceBands", ",".join(src_candidates))

    print(f"   Calibration sourceBands = {src_candidates}")
    p_cal = GPF.createProduct("Calibration", params, product)

    _list_bands(p_cal, header="Bands after Calibration")
    return p_cal


def sigma0_to_db(product, pol="VV"):
    """
    Convert Sigma0_<pol> (linear) to dB using SNAP operator LinearToFromdB.
    """
    band_names = _list_bands(product, header="Bands before dB conversion")

    # Prefer calibrated sigma0 band(s)
    sigma_candidates = [b for b in band_names if ("sigma0" in b.lower() and pol.lower() in b.lower())]
    if not sigma_candidates:
        # fallback: any sigma0
        sigma_candidates = [b for b in band_names if "sigma0" in b.lower()]

    if not sigma_candidates:
        raise RuntimeError(
            "Could not find Sigma0 band(s) to convert to dB. "
            "Check the bands printed above; you may need to adjust the band name filter."
        )

    params = HashMap()
    params.put("sourceBands", ",".join(sigma_candidates))  # convert these bands
    # Default behavior: linear -> dB
    p_db = GPF.createProduct("LinearToFromdB", params, product)

    _list_bands(p_db, header="Bands after dB conversion")
    return p_db


def speckle_filter_5x5(product, filter_name="Lee"):
    """
    Same speckle filter as before (5x5).
    """
    params = HashMap()
    params.put("filter", filter_name)

    JInt = jpy.get_type("java.lang.Integer")
    params.put("Size X", JInt(5))
    params.put("Size Y", JInt(5))

    return GPF.createProduct("Speckle-Filter", params, product)


def terrain_correction(product, pixel_spacing=10.0, dem="SRTM 1Sec HGT"):
    """
    Range-Doppler geometric correction in SNAP is done via 'Terrain-Correction'
    (Range-Doppler TC for SAR).
    """
    params = HashMap()
    params.put("demName", dem)
    params.put("pixelSpacingInMeter", float(pixel_spacing))
    params.put("mapProjection", "AUTO:42001")  # Auto UTM
    params.put("nodataValueAtSea", False)

    # Keep output(s)
    params.put("saveSelectedSourceBand", True)

    # Optional extra layers (keep if you want the same outputs as your S1 pipeline)
    params.put("saveDEM", True)
    params.put("saveLocalIncidenceAngle", True)

    return GPF.createProduct("Terrain-Correction", params, product)


def main_iceye_grd(in_path, out_path, pol="VV", pixel_spacing=10.0, dem="SRTM 1Sec HGT"):
    print("Reading ICEYE GRD:", in_path)
    p = ProductIO.readProduct(in_path)
    if p is None:
        raise RuntimeError(
            "SNAP could not read this ICEYE file. "
            "Make sure the ICEYE reader is installed/enabled in SNAP, and that in_path points to a supported file."
        )

    print("\n1) Radiometric Calibration -> Sigma0 (linear)")
    p_cal = calibrate_sigma0_grd(p, pol=pol)

    print("\n1b) Convert Sigma0 -> dB")
    p_db = sigma0_to_db(p_cal, pol=pol)

    print("\n2) Speckle Filter 5x5 (same as before)")
    p_flt = speckle_filter_5x5(p_db, filter_name="Lee")
    _list_bands(p_flt, header="Bands after Speckle Filter")

    print("\n3) Range-Doppler Terrain Correction (Geometric Correction)")
    p_tc = terrain_correction(p_flt, pixel_spacing=pixel_spacing, dem=dem)
    _list_bands(p_tc, header="Bands after Terrain Correction")

    # Output
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("\nWriting:", out_path)
    ProductIO.writeProduct(p_tc, out_path, "BEAM-DIMAP")
    print("Done.")


if __name__ == "__main__":
    # ✅ Set these to your ICEYE GRD input and desired output
    # NOTE: in_path must be something SNAP can read via its ICEYE reader (varies by ICEYE delivery format).
    in_path  = r"H:\10_ICEYE_Lethbridge_Project\ICEYE\20250520\IX_CS-20241_SAL000119_COLN10_SM_4755148_370978_1_GRD\ICEYE_X35_GRD_SM_4755148_20250520T182358.xml"
    out_path = r"H:\10_ICEYE_Lethbridge_Project\ICEYE\20250520\ICEYE_GRD_TC.dim"

    main_iceye_grd(
        in_path=in_path,
        out_path=out_path,
        pol="VV",
        pixel_spacing=10.0,
        dem="SRTM 1Sec HGT",
    )
