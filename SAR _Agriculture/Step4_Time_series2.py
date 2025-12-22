import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
root_folder = r"D:\7_Sentinel\North2023_Re"     # contains date folders like 20230513
zonal_subfolder_name = "Zonal"              # subfolder under each date folder
pol_tag = "VV"                              # change to "HV" if your files are HV
zone_field = "Field"                        # field ID column in your Excel
stat_col = "MEAN"                           # column holding HV/VH statistic (e.g., MEAN)

# output
out_long_csv   = os.path.join(root_folder, f"{pol_tag}_temporal_DOY_long.csv")
out_wide_csv   = os.path.join(root_folder, f"{pol_tag}_temporal_DOY_wide.csv")
out_wide_xlsx  = os.path.join(root_folder, f"{pol_tag}_temporal_DOY_wide.xlsx")  # <-- NEW
plot_png       = os.path.join(root_folder, f"{pol_tag}_first5_fields_timeseries.png")  # optional save
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

# Save CSV
long_df.to_csv(out_long_csv, index=False)
wide_df.to_csv(out_wide_csv)

# =========================
# NEW: Save WIDE table to Excel
# =========================
# Put index into columns for Excel readability
wide_out = wide_df.reset_index()
with pd.ExcelWriter(out_wide_xlsx, engine="openpyxl") as writer:
    wide_out.to_excel(writer, index=False, sheet_name=f"{pol_tag}_DOY")

print("✅ Done.")
print("Long CSV:", out_long_csv)
print("Wide CSV:", out_wide_csv)
print("Wide Excel:", out_wide_xlsx)

# =========================
# NEW: Plot first 5 fields
# =========================
# Build a plotting x-axis: if multiple years exist, use "YYYY-DOY" labels; else just DOY
years = wide_df.index.get_level_values("year")
doys  = wide_df.index.get_level_values("doy")

if years.nunique() == 1:
    x = doys
    xlabel = "DOY"
else:
    x = [f"{y}-{d:03d}" for y, d in zip(years, doys)]
    xlabel = "Year-DOY"

# pick first 5 fields (columns)
first5 = list(wide_df.columns[:5])
if len(first5) == 0:
    raise RuntimeError("wide_df has no field columns to plot.")

plt.figure()
for f in first5:
    plt.plot(x, wide_df[f], marker="o", label=str(f))

plt.xlabel(xlabel)
plt.ylabel(f"{pol_tag} ({stat_col})")
plt.title(f"Temporal evolution ({pol_tag}) - first 5 fields")
plt.legend()
plt.grid(True)

# If x is Year-DOY strings, rotate for readability
if years.nunique() > 1:
    plt.xticks(rotation=45, ha="right")

plt.tight_layout()

# Save plot (optional) and show
plt.savefig(plot_png, dpi=300)
plt.show()

print("Plot saved:", plot_png)

# Optional: show a few skipped files (schema mismatch)
if skipped_schema:
    print("\n⚠️ Skipped some Excel files because required columns were missing or read errors occurred.")
    for f, cols in skipped_schema[:10]:
        print(" -", f)
        print("   info/columns:", cols)


#%% Visualization of the results

plt.figure()

# Ensure time order
wide_df_plot = wide_df.sort_index()

for f in first5:
    # Select valid (non-NaN) points for this field
    y = wide_df_plot[f].dropna()

    # Build corresponding x values
    if isinstance(wide_df_plot.index, pd.MultiIndex):
        # (year, doy) index
        x_valid = y.index.get_level_values("doy")
    else:
        x_valid = y.index

    plt.plot(
        x_valid,
        y,
        linestyle="-",     # connect all existing points
        marker="o",
        linewidth=2,
        label=f"Field {f}"
    )

plt.xlabel(xlabel)
plt.ylabel(f"{pol_tag} ({stat_col})")
plt.title(f"Temporal evolution ({pol_tag}) – first 5 fields")
plt.legend()
#plt.grid(True)
plt.tight_layout()
plt.show()