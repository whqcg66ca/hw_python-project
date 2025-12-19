# -*- coding: utf-8 -*-
import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

# ArcMap has no addOutputsToMap; this env exists in Pro only.
# So we simply don't set it in ArcMap.
# arcpy.env.addOutputsToMap = False

# --- MODIFY THESE PATHS ---
in_folder   = r"D:\7_Sentinel\North2023_Re\20230503\S1A_20230503_TC.data"
out_folder  = r"D:\7_Sentinel\North2023_Re\20230503\Zonal"
in_polygons = r"D:\2_2019-2023_NewLiskeard_data_organized\Shapfile_merged2023\All_Polygon.shp"
zone_field  = "Field"
# ----------------------------

if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

# In ArcMap, scratchWorkspace should be a folder OR a file geodatabase.
# Folder is OK for rasters/dbf outputs.
arcpy.env.scratchWorkspace = out_folder

# Workspace
arcpy.env.workspace = in_folder

# -------------------------------------------------------------------
# SNAP *.data folders usually contain GeoTIFFs (often .tif) or .img
# ArcMap may NOT recognize "IMAGINE Image" raster type in ListRasters,
# so we'll do a filesystem scan for files we care about.
# -------------------------------------------------------------------
rasters = []
for fn in os.listdir(in_folder):
    low = fn.lower()
    if low.endswith(".tif") or low.endswith(".tiff") or low.endswith(".img"):
        rasters.append(fn)

rasters.sort()
print("Found rasters: {0}".format(rasters))

for raster_name in rasters:
    print("\nProcessing: {0}".format(raster_name))

    in_raster = os.path.join(in_folder, raster_name)
    base = os.path.splitext(raster_name)[0]

    out_table_full = os.path.join(out_folder, base + "_zonal_full.dbf")

    db_raster_path = None
    null_raster_path = None

    r = None
    r_null = None
    r_db = None

    try:
        name_lc = raster_name.lower()

        # ----------------------------------------------------
        # Skip conversion for LIA (or any non-backscatter layer)
        # (case-insensitive contains check)
        # ----------------------------------------------------
        if "projectedlocalincidenceangle" in name_lc or "localincidenceangle" in name_lc:
            print("  Detected Local Incidence Angle → using original values (no dB conversion).")
            raster_for_stats = in_raster

        # If already dB (by naming convention)
        elif ("_db" in name_lc) or ("db_" in name_lc) or ("db" in name_lc and "sigma" in name_lc) or ("db" in name_lc):
            print("  Detected likely dB product → using original values (no dB conversion).")
            raster_for_stats = in_raster

        else:
            print("  Creating intermediates on disk (SetNull, then dB)...")

            null_raster_path = os.path.join(out_folder, base + "_tmp_null.tif")
            db_raster_path   = os.path.join(out_folder, base + "_tmp_dB.tif")

            r = Raster(in_raster)
            r_null = SetNull(r <= 0, r)
            r_null.save(null_raster_path)

            r_db = 10 * Log10(Raster(null_raster_path))
            r_db.save(db_raster_path)

            raster_for_stats = db_raster_path

        # ----------------------------------------------------
        # ZonalStatisticsAsTable (ArcMap-friendly positional args)
        # ZonalStatisticsAsTable(in_zone_data, zone_field, in_value_raster, out_table, ignore_nodata, statistics_type)
        # ----------------------------------------------------
        print("  Running ZonalStatisticsAsTable (ALL stats)...")
        ZonalStatisticsAsTable(in_polygons, zone_field, raster_for_stats, out_table_full, "DATA", "ALL")

        # Delete intermediates
        for p in [db_raster_path, null_raster_path]:
            if p and arcpy.Exists(p):
                arcpy.management.Delete(p)
                print("  Deleted intermediate: {0}".format(p))

        # ----------------------------------------------------
        # Filter fields to: zone_field, MEAN, STD
        # ----------------------------------------------------
        print("  Creating table with only MEAN and STD...")

        fields = arcpy.ListFields(out_table_full)
        existing = {}
        for f in fields:
            existing[f.name.upper()] = f.name

        fm = arcpy.FieldMappings()

        def add_field(field_name):
            real = None
            # exact match
            for f in fields:
                if f.name == field_name:
                    real = f.name
                    break
            # case-insensitive match
            if real is None:
                real = existing.get(field_name.upper(), None)

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
            print("  WARNING: zone field '{0}' not found in {1}".format(zone_field, out_table_full))
        if not ok_mean:
            print("  WARNING: field 'MEAN' not found in zonal table.")
        if not ok_std:
            print("  WARNING: field 'STD' not found in zonal table.")

        filtered_dbf_name = base + "_zonal_mean_std.dbf"
        filtered_dbf = os.path.join(out_folder, filtered_dbf_name)

        arcpy.conversion.TableToTable(out_table_full, out_folder, filtered_dbf_name, field_mapping=fm)

        # ----------------------------------------------------
        # ArcMap TableToExcel often fails / depends on install.
        # If you have it, use .xls; if you prefer .xlsx, do it in Pro.
        # Safer default for ArcMap is .xls.
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
        try:
            del r
        except:
            pass
        try:
            del r_null
        except:
            pass
        try:
            del r_db
        except:
            pass

arcpy.CheckInExtension("Spatial")
print("\nAll done.")
