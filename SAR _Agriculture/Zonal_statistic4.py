import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

# --- MODIFY THESE PATHS ---
in_folder   = r"H:\7_Sentinel\North2023\20230812"
out_folder  = r"H:\7_Sentinel\North2023\20230812\Zonal"
in_polygons = r"H:\2_2019-2023_NewLiskeard_data_organized\Shapfile_merged2023\All_Polygon.shp"
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

    in_raster = os.path.join(in_folder, raster_name)

    base = os.path.splitext(raster_name)[0]
    # full zonal stats table (ALL stats)
    out_table_full = os.path.join(out_folder, base + "_zonal_full.dbf")

    # ----------------------------------------------------
    # CONDITION: If filename contains "Theta" → skip dB conversion
    # ----------------------------------------------------
    db_raster_path = None  # will be used only for non-theta rasters

    if "theta" in raster_name.lower():
        print("  Detected Theta image → using original values, no dB conversion.")
        raster_for_stats = Raster(in_raster)
    else:
        # Convert linear to dB (SAVE to disk temporarily)
        print("  Converting to dB (saved temporarily)...")
        raster_linear = Raster(in_raster)
        raster_linear = SetNull(raster_linear <= 0, raster_linear)
        raster_db = 10 * Log10(raster_linear)

        # Save dB raster to Zonal folder
        db_raster_name = base + "_dB.tif"
        db_raster_path = os.path.join(out_folder, db_raster_name)
        raster_db.save(db_raster_path)

        # Use saved dB raster for zonal statistics
        raster_for_stats = Raster(db_raster_path)

    # ----------------------------------------------------
    # Zonal stats (uses original raster OR dB raster)
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

    # Delete temporary dB raster right after zonal statistics
    if db_raster_path is not None and arcpy.Exists(db_raster_path):
        arcpy.management.Delete(db_raster_path)
        print("  Temporary dB raster deleted: {}".format(db_raster_path))

    # ----------------------------------------------------
    # Build a table with only: zone_field, MEAN, STD
    # ----------------------------------------------------
    print("  Creating table with only MEAN and STD...")

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
        print("  Warning: One or more fields (zone, MEAN, STD) not found in {}".format(out_table_full))

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
    print("  Exporting to Excel: {}".format(out_excel))

    arcpy.TableToExcel_conversion(
        Input_Table=filtered_dbf,
        Output_Excel_File=out_excel
    )

    print("Finished: " + raster_name)
