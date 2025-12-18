import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")
arcpy.env.overwriteOutput = True

# --- MODIFY THESE PATHS ---
in_folder   = r"D:\7_Sentinel\North2023\20230503"
in_polygons = r"D:\2_2019-2023_NewLiskeard_data_organized\Shapfile_merged2023\All_Polygon.shp"
zone_field  = "Field"
out_folder  = r"D:\7_Sentinel\North2023\20230503\Zonal"
# ----------------------------

# Set workspace and list rasters
arcpy.env.workspace = in_folder
rasters = arcpy.ListRasters()

print(rasters)

for raster_name in rasters:
    print("Processing: " + raster_name)

    in_raster = os.path.join(in_folder, raster_name)

    base = os.path.splitext(raster_name)[0]
    out_db = os.path.join(out_folder, base + "_dB.tif")
    out_table = os.path.join(out_folder, base + "_zonal.dbf")

    # Convert to dB
    raster_linear = Raster(in_raster)
    raster_linear = SetNull(raster_linear <= 0, raster_linear)
    raster_db = 10 * Log10(raster_linear)
    raster_db.save(out_db)

    # Zonal stats
    ZonalStatisticsAsTable(
        in_polygons,
        zone_field,
        raster_db,
        out_table,
        "DATA",
        "ALL"
    )

    print("Finished: " + raster_name)