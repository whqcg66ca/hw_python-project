import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True
arcpy.env.addOutputsToMap = False  # reduce UI overhead

# =========================
# CONFIG: common folder that contains date subfolders
# =========================
base_dir    = r"D:\10_ICEYE_Lethbridge_Project\ICEYE"   # <-- common folder
zonal_name  = "Zonal"                                   # each date has a Zonal subfolder
in_polygons = r"D:\10_ICEYE_Lethbridge_Project\Ground_data\Shp\Field_Leth_Final.shp"
zone_field  = "Field_N"

# =========================
# Loop over all date subfolders
# =========================
for dd in sorted(os.listdir(base_dir)):

    date_dir = os.path.join(base_dir, dd)
    if not os.path.isdir(date_dir):
        continue

    in_folder  = os.path.join(date_dir, zonal_name)  # rasters (.tif) live here
    out_folder = in_folder

    if not os.path.isdir(in_folder):
        print("⚠️ Skipping {} (no Zonal folder): {}".format(dd, in_folder))
        continue

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    print("\n==============================")
    print("📅 Processing date folder:", dd)
    print("Zonal folder:", in_folder)
    print("==============================")

    arcpy.env.workspace = in_folder
    rasters = arcpy.ListRasters()
    print(rasters)

    for raster_name in rasters:
        print("Processing:", raster_name)

        # Skip DEM / elevation layers
        if "elevation" in raster_name.lower():
            continue

        in_raster = os.path.join(in_folder, raster_name)
        base = os.path.splitext(raster_name)[0]

        out_table_full = os.path.join(out_folder, base + "_zonal_full.dbf")

        null_raster_path = None

        try:
            # ----------------------------------------------------
            # ICEYE data already in dB → optional masking only
            # ----------------------------------------------------
            print("  Using ICEYE dB raster directly (no conversion).")

            null_raster_path = os.path.join(out_folder, "{}_tmp_null.tif".format(base))

            r = Raster(in_raster)
            r_null = SetNull(IsNull(r), r)   # safest generic mask
            r_null.save(null_raster_path)

            del r, r_null
            try:
                arcpy.management.ClearWorkspaceCache()
            except Exception:
                pass

            raster_for_stats = null_raster_path

            # ----------------------------------------------------
            # Zonal statistics
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
            # Delete temp raster (only the _tmp_ raster)
            # ----------------------------------------------------
            if null_raster_path and arcpy.Exists(null_raster_path):
                arcpy.management.Delete(null_raster_path)

            # ----------------------------------------------------
            # Extract only zone_field, MEAN, STD
            # ----------------------------------------------------
            print("  Creating table with only MEAN and STD...")

            fm = arcpy.FieldMappings()
            fields = arcpy.ListFields(out_table_full)

            def add_field(name):
                for f in fields:
                    if f.name.upper() == name.upper():
                        fmap = arcpy.FieldMap()
                        fmap.addInputField(out_table_full, f.name)
                        fm.addFieldMap(fmap)
                        return True
                return False

            add_field(zone_field)
            add_field("MEAN")
            add_field("STD")

            filtered_dbf = os.path.join(out_folder, base + "_zonal_mean_std.dbf")

            arcpy.TableToTable_conversion(
                out_table_full,
                out_folder,
                os.path.basename(filtered_dbf),
                field_mapping=fm
            )

            # ----------------------------------------------------
            # Export to Excel
            # ----------------------------------------------------
            out_excel = os.path.join(out_folder, base + "_zonal_mean_std.xls")
            print("  Exporting to Excel:", out_excel)

            arcpy.conversion.TableToExcel(filtered_dbf, out_excel)

            print("Finished:", raster_name)

        except Exception as e:
            print("ERROR on:", raster_name)
            print(e)
            print(arcpy.GetMessages(2))

            if null_raster_path and arcpy.Exists(null_raster_path):
                try:
                    arcpy.management.Delete(null_raster_path)
                except:
                    pass
