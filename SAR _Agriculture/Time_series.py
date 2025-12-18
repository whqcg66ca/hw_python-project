import os
import glob
import pandas as pd

# =========================
# USER SETTINGS
# =========================
root_folder = r"H:\7_Sentinel\North2023"     # contains date folders like 20230513
zonal_subfolder_name = "Zonal"              # subfolder under each date folder
pol_tag = "VH"                              # change to "HV" if your files are HV
zone_field = "Field"                        # field ID column in your Excel
stat_col = "MEAN"                           # column holding HV/VH statistic (e.g., MEAN)

# output
out_long_csv = os.path.join(root_folder, f"{pol_tag}_temporal_DOY_long.csv")
out_wide_csv = os.path.join(root_folder, f"{pol_tag}_temporal_DOY_wide.csv")
# =========================


def is_date_folder(name: str) -> bool:
    return name.isdigit() and len(name) == 8


def read_excel_safely(path: str) -> pd.DataFrame:
    """
    Read Excel reliably:
    - .xls  uses xlrd
    - .xlsx uses openpyxl
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xls":
        return pd.read_excel(path, engine="xlrd")
    elif ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    else:
        return pd.read_excel(path)


records = []
skipped_schema = []

# Loop over YYYYMMDD folders
for d in os.listdir(root_folder):
    date_dir = os.path.join(root_folder, d)
    if not os.path.isdir(date_dir) or not is_date_folder(d):
        continue

    date = pd.to_datetime(d, format="%Y%m%d")
    year = int(date.year)
    doy = int(date.dayofyear)

    zonal_dir = os.path.join(date_dir, zonal_subfolder_name)
    if not os.path.isdir(zonal_dir):
        continue

    # Find Excel tables under YYYYMMDD/Zonal/ (including subfolders)
    excel_files = (
        glob.glob(os.path.join(zonal_dir, "**", "*.xls"), recursive=True) +
        glob.glob(os.path.join(zonal_dir, "**", "*.xlsx"), recursive=True)
    )

    # Filter by polarization tag in filename (case-insensitive)
    excel_files = [
        f for f in excel_files
        if pol_tag.lower() in os.path.basename(f).lower()
        and not os.path.basename(f).startswith("~$")
    ]

    for xls in excel_files:
        try:
            df = read_excel_safely(xls)
        except Exception as e:
            skipped_schema.append((xls, [f"READ_ERROR: {e}"]))
            continue

        if zone_field not in df.columns or stat_col not in df.columns:
            skipped_schema.append((xls, list(df.columns)))
            continue

        tmp = df[[zone_field, stat_col]].copy()
        tmp.columns = ["field", "value"]

        tmp["field"] = tmp["field"].astype(str).str.strip()
        tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
        tmp = tmp.dropna(subset=["value"])

        tmp["year"] = year
        tmp["doy"] = doy
        tmp["date"] = date
        tmp["source_file"] = os.path.basename(xls)

        records.append(tmp)

if not records:
    raise RuntimeError(
        f"No valid Excel tables found. Check:\n"
        f" - root_folder={root_folder}\n"
        f" - zonal_subfolder_name={zonal_subfolder_name}\n"
        f" - pol_tag={pol_tag}\n"
        f" - zone_field={zone_field}\n"
        f" - stat_col={stat_col}\n"
    )

# Combine all rows
long_df = pd.concat(records, ignore_index=True)

# If multiple tables exist for same (year,doy,field), average them
long_df = (
    long_df
    .groupby(["year", "doy", "field"], as_index=False)
    .agg(value=("value", "mean"))
)

# Wide table: index=(year,doy), columns=field
wide_df = (
    long_df
    .pivot(index=["year", "doy"], columns="field", values="value")
    .sort_index()
)

# Save
long_df.to_csv(out_long_csv, index=False)
wide_df.to_csv(out_wide_csv)

print("✅ Done.")
print("Long (year,doy,field,value):", out_long_csv)
print("Wide ((year,doy) x fields):", out_wide_csv)
print("\nPreview wide:")
print(wide_df.head())

# Optional: show a few skipped files (schema mismatch)
if skipped_schema:
    print("\n⚠️ Skipped some Excel files because required columns were missing or read errors occurred.")
    for f, cols in skipped_schema[:10]:
        print(" -", f)
        print("   info/columns:", cols)
