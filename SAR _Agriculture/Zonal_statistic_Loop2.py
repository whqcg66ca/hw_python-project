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
# --------------------------------------------------------------------
date_folders = [
    d for d in os.listdir(base_folder)
    if os.path.isdir(os.path.join(base_folder, d))
    and d.isdigit() and len(d) == 8
]

date_folders.sort()
print("Date folders found:", date_folders)

# --------------------------------------------------------------------
# Loop over each date folder and run your existing code
# --------------------------------------------------------------------
for date_name in date_folders:

    # ================================================================
    # SKIP 20230503 BECAUSE IT IS ALREADY PROCESSED
    # ================================================================
    if date_name == "20230503":
        print("\nSkipping date folder (already processed):", date_name)
        continue

    # ================================================================

    print("\n====================================")
    print("Processing date folder:", date_name)
    print("====================================")

    # --- MODIFY THESE PATHS PER DATE ---
    in_folder  = os.path.join(base_folder, date_name)
    out_folder = os.path.join(in_folder, "Zonal")
    # -----------------------------------

    # Make sure output folder exists
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
        print("  Created Zonal folder:", out_folder)
    else:
        print("  Zonal folder already exists:", out_folder)

    # Set workspace
    arcpy.env.workspace = in_folder
    rasters = arcpy.ListRasters()

    print("  Rasters found:", rasters)

    for raster_name in rasters:
        print("  Processing:", raster_name)

        in_raster = os.path.join(in_folder, raster_name)
        base = os.path.splitext(raster_name)[0]

        # Output table
        out_table_full = os.path.join(out_folder, base + "_zonal_full.dbf")

        # CONDITION: Theta → skip dB conversion
        if "theta" in raster_name.lower():
            print("    Theta image → no dB conversion.")
            raster_for_stats = Raster(in_raster)
        else:
            print("    Converting to dB...")
            raster_linear = Raster(in_raster)
            raster_linear = SetNull(raster_linear <= 0, raster_linear)
            raster_for_stats = 10 * Log10(raster_linear)

        # Zonal stats
        print("    Running ZonalStatisticsAsTable...")
        ZonalStatisticsAsTable(
            in_polygons,
            zone_field,
            raster_for_stats,
            out_table_full,
            "DATA",
            "ALL"
        )

        # Build reduced table: Field, MEAN, STD
        print("    Creating reduced table...")

        fm = arcpy.FieldMappings()
        fields = arcpy.ListFields(out_table_full)

        def add_field(field_name):
            for fld in fields:
                if fld.name.upper() == field_name.upper():
                    fmap = arcpy.FieldMap()
                    fmap.addInputField(out_table_full, fld.name)
                    fm.addFieldMap(fmap)

        add_field(zone_field)
        add_field("MEAN")
        add_field("STD")

        filtered_dbf = os.path.join(out_folder, base + "_zonal_mean_std.dbf")
        arcpy.TableToTable_conversion(
            out_table_full,
            out_folder,
            base + "_zonal_mean_std.dbf",
            field_mapping=fm
        )

        # Export to Excel
        out_excel = os.path.join(out_folder, base + "_zonal_mean_std.xls")
        print("    Exporting to Excel:", out_excel)

        arcpy.TableToExcel_conversion(filtered_dbf, out_excel)

        print("  Finished:", raster_name)

print("\nAll date folders processed.")
