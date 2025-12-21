import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False  # helps avoid Pro UI overhead

# --- MODIFY THESE PATHS ---
in_folder   = r"D:\7_Sentinel\North2023_Re\20230714\Zonal"
out_folder  = in_folder
in_polygons = r"D:\2_2019-2023_NewLiskeard_data_organized\Shapfile_merged2023\All_Polygon.shp"
zone_field  = "Field"
# ----------------------------

# Make sure output folder exists
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)

# Set workspace and list rasters
arcpy.env.workspace = in_folder
rasters = arcpy.ListRasters()
print(rasters)

for raster_name in rasters:
    print("Processing: " + raster_name)
    
    # ✅ SKIP elevation raster
    if "elevation" in raster_name.lower():
        continue

    in_raster = os.path.join(in_folder, raster_name)
    base = os.path.splitext(raster_name)[0]

    out_table_full = os.path.join(out_folder, base + "_zonal_full.dbf")

    # Track temp outputs to delete each iteration
    db_raster_path = None
    null_raster_path = None

    try:
        # ----------------------------------------------------
        # CONDITION: If filename contains "Theta" → skip dB conversion
        # ----------------------------------------------------
        if "theta" in raster_name.lower():
            print("  Detected Theta image → using original values, no dB conversion.")
            raster_for_stats = in_raster  # pass path (lowest memory)

        else:
            # ----------------------------------------------------
            # Save intermediate rasters to disk to reduce memory
            # 1) SetNull result
            # 2) dB result
            # ----------------------------------------------------
            print("  Creating intermediates on disk (SetNull, then dB)...")

            # (Python 2/3 compatible: no f-strings)
            null_raster_path = os.path.join(out_folder, "{}_tmp_null.tif".format(base))
            db_raster_path   = os.path.join(out_folder, "{}_tmp_dB.tif".format(base))

            # Build Null mask and save (disk)
            r = Raster(in_raster)
            r_null = SetNull(r <= 0, r)
            r_null.save(null_raster_path)

            # Compute dB directly from the in-memory raster object (avoid re-reading temp)
            r_db = 10 * Log10(r_null)
            r_db.save(db_raster_path)

            # Release locks (important before deleting)
            del r_db, r_null, r
            try:
                arcpy.management.ClearWorkspaceCache()
            except Exception:
                pass

            # ✅ Delete the original linear raster AFTER dB exists
            try:
                if arcpy.Exists(in_raster):
                    arcpy.management.Delete(in_raster)
                    print("  Deleted linear raster: {}".format(in_raster))
            except Exception as e:
                print("  WARNING: could not delete linear raster ({}): {}".format(in_raster, e))

            raster_for_stats = db_raster_path  # use saved dB raster path

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
            print("  Deleted intermediate:", db_raster_path)

        if null_raster_path and arcpy.Exists(null_raster_path):
            arcpy.management.Delete(null_raster_path)
            print("  Deleted intermediate:", null_raster_path)

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
            print("  Warning: One or more fields (zone, MEAN, STD) not found in {}".format(out_table_full))

        filtered_dbf_name = base + "_zonal_mean_std.dbf"
        filtered_dbf = os.path.join(out_folder, filtered_dbf_name)

        arcpy.TableToTable_conversion(
            in_rows=out_table_full,
            out_path=out_folder,
            out_name=filtered_dbf_name,
            field_mapping=fm
        )

        # ----------------------------------------------------
        # Export filtered DBF to Excel (.xlsx)
        # ----------------------------------------------------
        out_excel = os.path.join(out_folder, base + "_zonal_mean_std.xls")
        print("  Exporting to Excel: {}".format(out_excel))

        arcpy.conversion.TableToExcel(
            Input_Table=filtered_dbf,
            Output_Excel_File=out_excel
        )

        print("Finished: " + raster_name)

    except Exception as e:
        print("ERROR on:", raster_name)
        print(e)
        print(arcpy.GetMessages(2))

        # If an error happened, still try to delete intermediates
        for p in [db_raster_path, null_raster_path]:
            if p and arcpy.Exists(p):
                try:
                    arcpy.management.Delete(p)
                except:
                    pass

    finally:
        # Release ArcPy object references to reduce locks/memory
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
