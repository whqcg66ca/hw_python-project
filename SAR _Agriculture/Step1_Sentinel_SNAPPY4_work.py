import os
from esa_snappy import ProductIO, GPF, HashMap, jpy

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


def calibrate_sigma0(product, pols="VV,VH"):
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
    params.put("selectedPolarisations", pols)
    params.put("sourceBands", ",".join(intensity_bands))  # keep your original approach

    return GPF.createProduct("Calibration", params, product)

from esa_snappy import jpy  # make sure this import exists at the top

def speckle_filter_5x5(product, filter_name="Lee"):
    params = HashMap()

    # exact SNAP param names in some builds:
    params.put("filter", filter_name)

    # force Java Integer (not Python int)
    JInt = jpy.get_type('java.lang.Integer')
    params.put("Size X", JInt(5))
    params.put("Size Y", JInt(5))

    return GPF.createProduct("Speckle-Filter", params, product)



def terrain_correction(product, pixel_spacing=10.0, dem="SRTM 1Sec HGT"):
    params = HashMap()
    params.put("demName", dem)                       # "SRTM 1Sec HGT" or "SRTM 3Sec"
    params.put("pixelSpacingInMeter", float(pixel_spacing))
    params.put("mapProjection", "AUTO:42001")        # Auto UTM
    params.put("nodataValueAtSea", False)

    # Keep your main band(s)
    params.put("saveSelectedSourceBand", True)

    # ✅ Add these to output extra layers
    params.put("saveDEM", True)                                  # output DEM raster
    params.put("saveLocalIncidenceAngle", True)                   # output local incidence angle
    #params.put("saveProjectedLocalIncidenceAngle", True)         # output local incidence angle (projected)

    # (Optional extras you may also like)
    # params.put("saveLocalIncidenceAngle", True)                # non-projected local incidence angle (if supported)
    # params.put("saveIncidenceAngleFromEllipsoid", True)        # incidence angle from ellipsoid

    return GPF.createProduct("Terrain-Correction", params, product)



def topsar_merge(products_list):
    """Merge IW1/IW2/IW3 back into one product (after deburst)."""
    params = HashMap()
    # Many SNAP builds require Java Product[] rather than a Python list
    arr = jpy.array('org.esa.snap.core.datamodel.Product', len(products_list))
    for i, p in enumerate(products_list):
        arr[i] = p
    return GPF.createProduct("TOPSAR-Merge", params, arr)


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

        # ✅ FIX: Calibration MUST be applied BEFORE Deburst for your SLC TOPS product
        print("3) Radiometric Calibration (Sigma0)  [BEFORE Deburst]")
        p_cal = calibrate_sigma0(p_split, pols=pols)

        print("   Deburst")
        p_deb = deburst(p_cal)

        print("4) Speckle Filter 5x5")
        p_flt = speckle_filter_5x5(p_deb, filter_name="Lee")

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
    in_path = r"D:\7_Sentinel\North2023_zip\ZIP\S1A_IW_SLC__1SDV_20230503T230934_20230503T231001_048376_05D190_B660.zip"
    out_path = r"D:\7_Sentinel\North2023_Re\20230503\S1A_20230515_TC.dim"

    main(in_path, out_path, pols="VV,VH", do_merge=True)
