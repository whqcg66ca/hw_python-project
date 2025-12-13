import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

# --- ROOT FOLDER WITH ALL DATE SUBFOLDERS ---
base_folder = r"D:\7_Sentinel\North2023"

# Polygon and zone field are the same for all dates
in_polygons = r"D:\2_2019-2023_NewLiskeard_data_organized\Shapfile_merged2023\All_Polygon.shp"
zone_field  = "Field"

# --------------------------------------------------------------------
# Get all date folders under base_folder
# (assumes folders like 20230503, 20230515, etc.)
# --------------------------------------------------------------------
date_folders = [
    d for d in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, d))
    and d.isdigit() and len(d) == 8
]

date_folders.sort()
arcpy.AddMessage("Date folders found: {}".format(date_folders))

# --------------------------------------------------------------------
# Loop over each date folder and run your existing code
# --------------------------------------------------------------------
for date_name in date_folders:

    # Skip 20230503 (already processed)
    if date_name == "20230503":
        arcpy.AddMessage("\nSkipping date folder (already processed): {}".format(date_name))
        continue

    arcpy.AddMessage("\n====================================")
    arcpy.AddMessage("Processing date folder: {}".format(date_name))
    arcpy.AddMessage("====================================")

    # --- SET PATHS FOR THIS DATE ---
    in_folder  = os.path.join(base_folder, date_name)
    out_folder = os.path.join(in_folder, "Zonal")
    # --------------------------------

    # Make sure output folder exists
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
        arcpy.AddMessage("  Created Zonal folder: {}".format(out_folder))
    else:
        arcpy.AddMessage("  Zonal folder already exists: {}".format(out_folder))

    # Set workspace and list rasters
    arcpy.env.workspace = in_folder
    rasters = arcpy.ListRasters()

    arcpy.AddMessage("  Rasters found: {}".format(rasters))

    for raster_name in rasters:
        arcpy.AddMessage("  Processing: {}".format(raster_name))

        in_raster = os.path.join(in_folder, raster_name)

        base = os.path.splitext(raster_name)[0]
        # full zonal stats table (ALL stats)
        out_table_full = os.path.join(out_folder, base + "_zonal_full.dbf")

        # ----------------------------------------------------
        # CONDITION: If filename contains "Theta" → skip dB conversion
        # ----------------------------------------------------
        if "theta" in raster_name.lower():
            arcpy.AddMessage("    Detected Theta image → using original values, no dB conversion.")
            raster_for_stats = Raster(in_raster)
        else:
            # Convert linear to dB (in memory, no save)
            arcpy.AddMessage("    Converting to dB...")
            raster_linear = Raster(in_raster)
            raster_linear = SetNull(raster_linear <= 0, raster_linear)
            raster_for_stats = 10 * Log10(raster_linear)

        # ----------------------------------------------------
        # Zonal stats (uses original raster OR dB raster)
        # ----------------------------------------------------
        arcpy.AddMessage("    Running ZonalStatisticsAsTable (ALL stats)...")
        ZonalStatisticsAsTable(
            in_polygons,
            zone_field,
            raster_for_stats,
            out_table_full,
            "DATA",
            "ALL"
        )

        # ----------------------------------------------------
        # Build a table with only: zone_field, MEAN, STD
        # ----------------------------------------------------
        arcpy.AddMessage("    Creating table with only MEAN and STD...")

        # Prepare field mappings
        fm = arcpy.FieldMappings()
        fields = arcpy.ListFields(out_table_full)

        # Helper function to add a field to field mappings if it exists
        def add_field_to_mappings(field_name_logical):
            """Add a field to field mappings by (case-insensitive) name."""
            for fld in fields:
                if fld.name.upper() == field_name_logical.upper():
                    fmap = arcpy.FieldMap()
                    fmap.addInputField(out_table_full, fld.name)
                    fm.addFieldMap(fmap)
                    return True
            return False

        # Add zone field, MEAN, STD
        added_zone = add_field_to_mappings(zone_field)
        added_mean = add_field_to_mappings("MEAN")
        added_std  = add_field_to_mappings("STD")

        if not (added_zone and added_mean and added_std):
            arcpy.AddMessage(
                "    Warning: One or more fields (zone, MEAN, STD) not found in {}".format(out_table_full)
            )

        # Output filtered DBF on disk
        filtered_dbf_name = base + "_zonal_mean_std.dbf"
        filtered_dbf = os.path.join(out_folder, filtered_dbf_name)

        arcpy.TableToTable_conversion(
            in_rows=out_table_full,
            out_path=out_folder,
            out_name=filtered_dbf_name,
            field_mapping=fm
        )

        # ----------------------------------------------------
        # Export filtered DBF to Excel
        # ----------------------------------------------------
        out_excel = os.path.join(out_folder, base + "_zonal_mean_std.xls")
        arcpy.AddMessage("    Exporting to Excel: {}".format(out_excel))

        arcpy.TableToExcel_conversion(
            Input_Table=filtered_dbf,
            Output_Excel_File=out_excel
        )

        arcpy.AddMessage("  Finished: {}".format(raster_name))

arcpy.AddMessage("\nAll date folders processed.")