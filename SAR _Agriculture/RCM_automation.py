import os
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from scipy.ndimage import uniform_filter1d

# Function to load SAR data using GDAL
def load_sar_data(file_path):
    dataset = gdal.Open(file_path)  # Open the SAR data file
    band = dataset.GetRasterBand(1)  # Get the first band (e.g., HH or VV)
    data = band.ReadAsArray()  # Read the data as a numpy array
    return data

# Example directory containing RCM SAR data (change to your path)
sar_data_dir = "path/to/your/rcm/data/"  # Update this path with your data

# Load all SAR images (assuming they are in .tif format)
sar_images = []
for file_name in os.listdir(sar_data_dir):
    if file_name.endswith(".tif"):
        file_path = os.path.join(sar_data_dir, file_name)
        sar_images.append(load_sar_data(file_path))

# Stack the images into a 3D numpy array (time, height, width)
sar_stack = np.array(sar_images)

# 1. Time Series Analysis - Calculate the mean and standard deviation of backscatter
mean_backscatter = np.mean(sar_stack, axis=0)
std_backscatter = np.std(sar_stack, axis=0)

# Visualize the mean and standard deviation of backscatter
plt.figure(figsize=(12, 5))

# Plot the mean backscatter
plt.subplot(1, 2, 1)
plt.imshow(mean_backscatter, cmap='gray')
plt.title("Mean Backscatter Over Time")
plt.colorbar()

# Plot the standard deviation of backscatter
plt.subplot(1, 2, 2)
plt.imshow(std_backscatter, cmap='jet')
plt.title("Standard Deviation of Backscatter")
plt.colorbar()

plt.show()

# 2. Change Detection for Seeding and Harvest
# Assume first image is the seeding stage and the last image is the harvest stage
initial_image = sar_stack[0]  # First image (seeding)
final_image = sar_stack[-1]  # Last image (harvest)

# Calculate the difference in backscatter between the two images
difference = final_image - initial_image

# Visualize the difference to highlight the change
plt.imshow(difference, cmap='coolwarm')
plt.title("Change in Backscatter (Seeding to Harvest)")
plt.colorbar()
plt.show()

# Set a threshold to detect the change (e.g., significant decrease in backscatter for harvest)
threshold = np.percentile(difference, 90)  # Top 10% change
change_mask = difference > threshold

# Show the change detection mask
plt.imshow(change_mask, cmap='gray')
plt.title("Change Detection Mask (Harvest Detected)")
plt.show()

# 3. Seeding and Harvest Date Detection using Change Detection over Time
# Apply a moving average filter to the SAR stack (smoothing over time)
smoothed_stack = uniform_filter1d(sar_stack, size=5, axis=0)

# Calculate the change over time (difference between smoothed images)
change_over_time = np.diff(smoothed_stack, axis=0)

# Identify significant changes that could correspond to seeding and harvest dates
change_threshold = np.percentile(np.abs(change_over_time), 90)  # Detect top 10% changes
seeding_date = np.argmax(np.abs(change_over_time) > change_threshold)
harvest_date = len(sar_stack) - np.argmax(np.abs(change_over_time[::-1]) > change_threshold)

# Print the estimated seeding and harvest dates
print(f"Estimated Seeding Date: {seeding_date}")
print(f"Estimated Harvest Date: {harvest_date}")

# 4. Visualization of Change Over Time and Detection of Seeding and Harvest Dates
plt.plot(range(len(sar_stack)-1), np.abs(change_over_time).mean(axis=(1, 2)), label='Change Magnitude')
plt.axvline(x=seeding_date, color='g', linestyle='--', label='Seeding Date')
plt.axvline(x=harvest_date, color='r', linestyle='--', label='Harvest Date')
plt.xlabel("Time (Days)")
plt.ylabel("Change Magnitude")
plt.legend()
plt.title("Seeding and Harvest Date Detection")
plt.show()
