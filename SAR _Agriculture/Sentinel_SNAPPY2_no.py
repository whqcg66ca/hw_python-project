import os
from esa_snappy import ProductIO, GPF, HashMap
print('modal is ok')
# --- IMPORTANT: load SNAP operators ---
GPF.getDefaultInstance().getOperatorSpiRegistry().loadOperatorSpis()


def apply_orbit(product):
    params = HashMap()
    params.put("orbitType", "Sentinel Precise (Auto Download)")
    params.put("continueOnFail", True)
    return GPF.createProduct("Apply-Orbit-File", params, product)


def topsar_split(product, subswath="IW1", pols="VV,VH"):
    """
    TOPSAR-Split is normally required before Deburst for IW SLC.
    This keeps all bursts by default (no explicit burst selection).
    """
    params = HashMap()
    params.put("subswath", subswath)              # IW1 / IW2 / IW3
    params.put("selectedPolarisations", pols)     # "VV,VH" or "VV" etc.
    # Optionally restrict bursts:
    # params.put("firstBurstIndex", 1)
    # params.put("lastBurstIndex", 999)
    return GPF.createProduct("TOPSAR-Split", params, product)


def deburst(product):
    params = HashMap()
    return GPF.createProduct("TOPSAR-Deburst", params, product)


def calibrate_sigma0(product):
    # List available bands
    band_names = [product.getBandAt(i).getName() for i in range(product.getNumBands())]
    print("   Available bands:", band_names)

    # Pick intensity bands automatically
    intensity_bands = [b for b in band_names if b.lower().startswith("intensity")]

    # Fallbacks for some S1 products/operators
    if not intensity_bands:
        intensity_bands = [b for b in band_names if "amplitude" in b.lower()]
    if not intensity_bands:
        intensity_bands = [b for b in band_names if "sigma0" in b.lower() or "gamma0" in b.lower()]

    if not intensity_bands:
        raise RuntimeError("No suitable source bands found for Calibration. Check band names printed above.")

    params = HashMap()
    params.put("outputSigmaBand", True)
    params.put("outputBetaBand", False)
    params.put("outputGammaBand", False)
    params.put("sourceBands", ",".join(intensity_bands))  # <-- valid bands only

    return GPF.createProduct("Calibration", params, product)



def speckle_filter_5x5(product, filter_name="Lee"):
    params = HashMap()
    params.put("filter", filter_name)   # e.g., "Lee", "Refined Lee", "Gamma Map"
    params.put("filterSizeX", 5)
    params.put("filterSizeY", 5)
    return GPF.createProduct("Speckle-Filter", params, product)


def terrain_correction(product, pixel_spacing=10.0, dem="SRTM 1Sec HGT"):
    params = HashMap()
    params.put("demName", dem)                       # "SRTM 1Sec HGT" or "SRTM 3Sec"
    params.put("pixelSpacingInMeter", float(pixel_spacing))
    params.put("mapProjection", "AUTO:42001")        # Auto UTM
    params.put("nodataValueAtSea", False)
    params.put("saveSelectedSourceBand", True)
    return GPF.createProduct("Terrain-Correction", params, product)


def topsar_merge(products_list):
    """Merge IW1/IW2/IW3 back into one product (after deburst)."""
    params = HashMap()
    return GPF.createProduct("TOPSAR-Merge", params, products_list)


def main(in_path, out_path, pols="VV,VH", do_merge=True):
    print("Reading:", in_path)
    p = ProductIO.readProduct(in_path)

    print("1) Apply Orbit")
    p_orb = apply_orbit(p)

    # Process each subswath
    subswaths = ["IW1", "IW2", "IW3"]
    processed = []

    for sw in subswaths:
        print(f"\n2) TOPSAR Split: {sw}")
        p_split = topsar_split(p_orb, subswath=sw, pols=pols)

        print("   Deburst")
        p_deb = deburst(p_split)

        print("3) Radiometric Calibration (Sigma0)")
        # Convert "VV,VH" -> ("VV","VH")
        pol_tuple = tuple([x.strip() for x in pols.split(",") if x.strip()])
        p_cal = calibrate_sigma0(p_deb)

        print("4) Speckle Filter 5x5")
        p_flt = speckle_filter_5x5(p_cal, filter_name="Lee")

        processed.append(p_flt)

    if do_merge and len(processed) > 1:
        print("\nMerging IW1/IW2/IW3 ...")
        p_m = topsar_merge(processed)
        p_for_tc = p_m
    else:
        p_for_tc = processed[0]

    print("5) Terrain Correction (TC)")
    p_tc = terrain_correction(p_for_tc, pixel_spacing=10.0, dem="SRTM 1Sec HGT")

    # Output
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print("Writing:", out_path)
    ProductIO.writeProduct(p_tc, out_path, "BEAM-DIMAP")
    print("Done.")


if __name__ == "__main__":
    # Example inputs:
    # in_path  can be .SAFE folder or .zip (SLC zip) or .dim
    in_path = r"H:\7_Sentinel\South2023_zip\S1A_IW_SLC__1SDV_20230503T230909_20230503T230936_048376_05D190_AD33.zip" 
    out_path = r"H:\7_Sentinel\South2023\test\S1A_20230503_TC.dim"

    main(in_path, out_path, pols="VV,VH", do_merge=True)
