import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

# NOTE: ArcMap doesn't use addOutputsToMap; keep it but it may be ignored
try:
    arcpy.env.addOutputsToMap = False
except:
    pass

# --- MODIFY THESE PATHS ---
in_folder   = r"D:\7_Sentinel\North2023_Re\20230503\S1A_20230503_TC.data"  # SNAP .data folder
out_folder  = r"D:\7_Sentinel\North2023_Re\20230503\Zonal"
in_polygons = r"D:\2_2019-2023_NewLiskeard_data_organized\Shapfile_merged2023\All_Polygon.shp"
zone_field  = "Field"
# ----------------------------

if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

# ------------------------------------------------------------
# IMPORTANT for ArcMap stability:
# Do NOT set scratchWorkspace to a plain folder here.
# (ArcMap can freeze / lock when GP tools try to write temp tables)
# If you want, you can set it to a FileGDB instead.
# ------------------------------------------------------------
# arcpy.env.scratchWorkspace = out_folder

# ------------------------------------------------------------
# Raster listing:
# - If in_folder is a SNAP *.data folder, ArcMap ListRasters() may hang or return nothing.
# - Use filesystem scan for .tif/.img inside the folder (fast, safe).
# ------------------------------------------------------------
rasters = []
for fn in os.listdir(in_folder):
    low = fn.lower()
    if low.endswith(".tif") or low.endswith(".tiff") or low.endswith(".img"):
        rasters.append(fn)

rasters.sort()
print(rasters)

for raster_name in rasters:
    print("Processing: " + raster_name)

    # ----------------------------------------------------
    # SKIP elevation
    # ----------------------------------------------------
    if "elevation" in raster_name.lower():
        print("  Skipping elevation raster.")
        continue

    in_raster = os.path.join(in_folder, raster_name)
    base = os.path.splitext(raster_name)[0]

    out_table_full = os.path.join(out_folder, base + "_zonal_full.dbf")

    db_raster_path = None
    null_raster_path = None

    try:
        # ----------------------------------------------------
        # CONDITION: If filename contains "Theta" or LIA → skip dB conversion
        # ----------------------------------------------------
        name_lc = raster_name.lower()
        if ("theta" in name_lc) or ("projectedlocalincidenceangle" in name_lc) or ("localincidenceangle" in name_lc):
            print("  Detected Theta/LIA → using original values, no dB conversion.")
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
        # Zonal stats
        # ----------------------------------------------------
        print("  Running ZonalStatisticsAsTable (ALL stats)...")
        ZonalStatisticsAsTable(
            in_polygons,
            zone_field,
            raster_for_stats,
            out_table_full,
            "DATA",
            "ALL"
        )

        # ----------------------------------------------------
        # Delete intermediate rasters immediately after zonal statistics
        # ----------------------------------------------------
        if db_raster_path and arcpy.Exists(db_raster_path):
            arcpy.management.Delete(db_raster_path)
            print("  Deleted intermediate: " + db_raster_path)

        if null_raster_path and arcpy.Exists(null_raster_path):
            arcpy.management.Delete(null_raster_path)
            print("  Deleted intermediate: " + null_raster_path)

        # ----------------------------------------------------
        # Build a table with only: zone_field, MEAN, STD
        # ----------------------------------------------------
        print("  Creating table with only MEAN and STD...")

        fm = arcpy.FieldMappings()
        fields = arcpy.ListFields(out_table_full)

        def add_field_to_mappings(field_name_logical):
            for fld in fields:
                if fld.name.upper() == field_name_logical.upper():
                    fmap = arcpy.FieldMap()
                    fmap.addInputField(out_table_full, fld.name)
                    fm.addFieldMap(fmap)
                    return True
            return False

        added_zone = add_field_to_mappings(zone_field)
        added_mean = add_field_to_mappings("MEAN")
        added_std  = add_field_to_mappings("STD")

        if not (added_zone and added_mean and added_std):
            print("  Warning: One or more fields (zone, MEAN, STD) not found in " + out_table_full)

        filtered_dbf_name = base + "_zonal_mean_std.dbf"
        filtered_dbf = os.path.join(out_folder, filtered_dbf_name)

        arcpy.TableToTable_conversion(
            in_rows=out_table_full,
            out_path=out_folder,
            out_name=filtered_dbf_name,
            field_mapping=fm
        )

        # ----------------------------------------------------
        # Export DBF to Excel
        # ArcMap is happiest with .xls (not .xlsx).
        # ----------------------------------------------------
        out_excel = os.path.join(out_folder, base + "_zonal_mean_std.xls")
        print("  Exporting to Excel: " + out_excel)

        arcpy.conversion.TableToExcel(
            Input_Table=filtered_dbf,
            Output_Excel_File=out_excel
        )

        print("Finished: " + raster_name)

    except Exception as e:
        print("ERROR on: " + raster_name)
        print(e)
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
