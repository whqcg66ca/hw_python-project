# -*- coding: utf-8 -*-
import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

# --- MODIFY THESE PATHS ---
in_folder   = r"D:\7_Sentinel\North2023_Re\20230503\S1A_20230503_TC.data"
out_folder  = r"D:\7_Sentinel\North2023_Re\20230503\Zonal"
in_polygons = r"D:\2_2019-2023_NewLiskeard_data_organized\Shapfile_merged2023\All_Polygon.shp"
zone_field  = "Field"
# ----------------------------

if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

arcpy.env.scratchWorkspace = out_folder
arcpy.env.workspace = in_folder

# ------------------------------------------------------------
# List SNAP rasters from .data folder (filesystem-safe)
# ------------------------------------------------------------
rasters = []
for fn in os.listdir(in_folder):
    low = fn.lower()
    if low.endswith(".tif") or low.endswith(".tiff") or low.endswith(".img"):
        rasters.append(fn)

rasters.sort()
print("Found rasters: {0}".format(rasters))

# ============================================================
# Main loop
# ============================================================
for raster_name in rasters:
    print("\nProcessing: {0}".format(raster_name))

    # --------------------------------------------------------
    # ✅ SKIP elevation raster (ONLY CHANGE REQUESTED)
    # --------------------------------------------------------
    if "elevation" in raster_name.lower():
        print("  Skipping elevation raster.")
        continue
    # --------------------------------------------------------

    in_raster = os.path.join(in_folder, raster_name)
    base = os.path.splitext(raster_name)[0]

    out_table_full = os.path.join(out_folder, base + "_zonal_full.dbf")

    db_raster_path = None
    null_raster_path = None

    r = r_null = r_db = None

    try:
        name_lc = raster_name.lower()

        # ----------------------------------------------------
        # Decide whether to do dB conversion
        # ----------------------------------------------------
        if "projectedlocalincidenceangle" in name_lc or "localincidenceangle" in name_lc:
            print("  Detected Local Incidence Angle → no dB conversion.")
            raster_for_stats = in_raster

        elif ("_db" in name_lc) or ("db_" in name_lc) or ("db" in name_lc and "sigma" in name_lc):
            print("  Detected dB product → no dB conversion.")
            raster_for_stats = in_raster

        else:
            print("  Creating intermediates (SetNull → dB)...")

            null_raster_path = os.path.join(out_folder, base + "_tmp_null.tif")
            db_raster_path   = os.path.join(out_folder, base + "_tmp_dB.tif")

            r = Raster(in_raster)
            r_null = SetNull(r <= 0, r)
            r_null.save(null_raster_path)

            r_db = 10 * Log10(Raster(null_raster_path))
            r_db.save(db_raster_path)

            raster_for_stats = db_raster_path

        # ----------------------------------------------------
        # Zonal statistics
        # ----------------------------------------------------
        print("  Running ZonalStatisticsAsTable...")
        ZonalStatisticsAsTable(
            in_polygons,
            zone_field,
            raster_for_stats,
            out_table_full,
            "DATA",
            "ALL"
        )

        # ----------------------------------------------------
        # Clean intermediate rasters
        # ----------------------------------------------------
        for p in [db_raster_path, null_raster_path]:
            if p and arcpy.Exists(p):
                arcpy.management.Delete(p)
                print("  Deleted intermediate: {0}".format(p))

        # ----------------------------------------------------
        # Keep only zone_field, MEAN, STD
        # ----------------------------------------------------
        print("  Creating filtered table (MEAN, STD)...")

        fields = arcpy.ListFields(out_table_full)
        existing = {}
        for f in fields:
            existing[f.name.upper()] = f.name

        fm = arcpy.FieldMappings()

        def add_field(fname):
            real = existing.get(fname.upper(), None)
            if real is None:
                return False
            fmap = arcpy.FieldMap()
            fmap.addInputField(out_table_full, real)
            fm.addFieldMap(fmap)
            return True

        ok_zone = add_field(zone_field)
        ok_mean = add_field("MEAN")
        ok_std  = add_field("STD")

        if not ok_zone:
            print("  WARNING: zone field not found.")
        if not ok_mean:
            print("  WARNING: MEAN field not found.")
        if not ok_std:
            print("  WARNING: STD field not found.")

        filtered_name = base + "_zonal_mean_std.dbf"
        filtered_dbf = os.path.join(out_folder, filtered_name)

        arcpy.conversion.TableToTable(
            out_table_full,
            out_folder,
            filtered_name,
            field_mapping=fm
        )

        # ----------------------------------------------------
        # Export to Excel (ArcMap-safe .xls)
        # ----------------------------------------------------
        out_excel = os.path.join(out_folder, base + "_zonal_mean_std.xls")
        print("  Exporting to Excel: {0}".format(out_excel))

        arcpy.conversion.TableToExcel(filtered_dbf, out_excel)

        print("Finished: {0}".format(raster_name))

    except Exception as e:
        print("ERROR on: {0}".format(raster_name))
        print("Exception: {0}".format(e))
        print(arcpy.GetMessages(2))

        for p in [db_raster_path, null_raster_path]:
            if p and arcpy.Exists(p):
                try:
                    arcpy.management.Delete(p)
                except:
                    pass

    finally:
        for v in ("r", "r_null", "r_db"):
            try:
                del globals()[v]
            except:
                pass

arcpy.CheckInExtension("Spatial")
print("\nAll done.")
