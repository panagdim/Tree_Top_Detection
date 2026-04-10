# %%
# Works better with Conifers 

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from rasterio.warp import reproject, Resampling
from tkinter import Tk, filedialog 

# 1. File selection | Select the DTM 

Tk().withdraw()  # hide Tkinter root window

print("Select DEM (DTM) file")
dem_path = filedialog.askopenfilename(
    title="Select DEM (DTM)",
    filetypes=[("GeoTIFF files", "*.tif *.tiff")]
)

# Select the DSM
print("Select DSM file")
dsm_path = filedialog.askopenfilename(
    title="Select DSM",
    filetypes=[("GeoTIFF files", "*.tif *.tiff")]
)

# 2. Load DEM and DSM

with rasterio.open(dem_path) as dem_src, rasterio.open(dsm_path) as dsm_src:
    dem = dem_src.read(1).astype(float)
    transform = dem_src.transform
    dem_crs = dem_src.crs

    dsm = dsm_src.read(1).astype(float)

    # Resample DSM to DEM grid
    dsm_resampled = np.empty_like(dem, dtype=float)
    reproject(
        source=dsm,
        destination=dsm_resampled,
        src_transform=dsm_src.transform,
        src_crs=dsm_src.crs,
        dst_transform=dem_src.transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear
    )

# 3. Compute CHM (Canopy Height Model)

chm = dsm_resampled - dem
chm[chm < 0] = 0  # remove negative heights


# 4. Smooth CHM for broadleaf crowns

sigma_smooth = 3  # adjust according to crown size
chm_smooth = gaussian_filter(chm, sigma=sigma_smooth)

# 5. Detect tree tops

min_distance_pixels = 3  # approx half average crown diameter
min_height = 2.5           # minimum tree height to detect (meters)

coordinates = peak_local_max(
    chm_smooth,
    min_distance=min_distance_pixels,
    threshold_abs=min_height
)

# 6. Extract tree data

tree_heights = chm_smooth[coordinates[:, 0], coordinates[:, 1]]

# 6a. Remove extreme tree heights

# Remove heights <=0 and > maximum expected (e.g., 25 m)
max_tree_height = 5
valid_idx = np.where((tree_heights > 0) & (tree_heights <= max_tree_height))[0]

coordinates_filtered = coordinates[valid_idx]
tree_heights_filtered = tree_heights[valid_idx]

# Optional: further filter using mean ± 2*std
mean_h = np.mean(tree_heights_filtered)
std_h = np.std(tree_heights_filtered)
valid_idx2 = np.where((tree_heights_filtered >= mean_h - 2*std_h) &
                      (tree_heights_filtered <= mean_h + 2*std_h))[0]

coordinates_filtered = coordinates_filtered[valid_idx2]
tree_heights_filtered = tree_heights_filtered[valid_idx2]


# 7. Save filtered tree data to CSV

df_filtered = pd.DataFrame({
    "Tree_ID": np.arange(1, len(coordinates_filtered) + 1),
    "X": coordinates_filtered[:, 0],
    "Y": coordinates_filtered[:, 1],
    "Height": tree_heights_filtered
})

output_csv = "C:/Users/Dimitris/Desktop/Tree Top Detection.csv"
df_filtered.to_csv(output_csv, index=False)

print(f"Filtered tree data saved to {output_csv}")

# Print the number of detected trees
num_trees = len(df_filtered)
print(f"Number of filtered trees detected: {num_trees}")


# 8. Visualization

plt.figure(figsize=(10, 8))
plt.imshow(df_filtered, cmap='viridis')
plt.scatter(
    coordinates[:, 1],
    coordinates[:, 0],
    c='red',
    s=8,
    label='Tree Tops'
)
plt.legend()

plt.title("Detected Tree Tops")
#plt.colorbar(label="Height (m)")

# Add North Arrow | Position in axes fraction (top-left corner)
ax = plt.gca()
ax.annotate('N', xy=(0.95, 0.15), xytext=(0.95, 0.05),
            arrowprops=dict(facecolor='black', width=3, headwidth=10),
            ha='center', va='center', fontsize=12, xycoords='axes fraction')

plt.savefig("tree_tops.png", dpi=300)
plt.show()


# %%
# ADAPTIVE UAV TREE TOP DETECTION 

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.ndimage import gaussian_filter
from scipy import ndimage as ndi
from rasterio.warp import reproject, Resampling

# 1. FILE INPUT

Tk().withdraw()

print("Select DTM file")
dem_path = filedialog.askopenfilename(
    title="Select DTM",
    filetypes=[("GeoTIFF files", "*.tif *.tiff")]
)

print("Select DSM file")
dsm_path = filedialog.askopenfilename(
    title="Select DSM",
    filetypes=[("GeoTIFF files", "*.tif *.tiff")]
)

# 2. LOAD DATA

with rasterio.open(dem_path) as dem_src, rasterio.open(dsm_path) as dsm_src:
    dem = dem_src.read(1).astype(float)
    dsm = dsm_src.read(1).astype(float)

    transform = dem_src.transform
    dem_crs = dem_src.crs

    dsm_resampled = np.empty_like(dem, dtype=float)

    reproject(
        source=dsm,
        destination=dsm_resampled,
        src_transform=dsm_src.transform,
        src_crs=dsm_src.crs,
        dst_transform=dem_src.transform,
        dst_crs=dem_crs,
        resampling=Resampling.bilinear
    )

# 3. CANOPY HEIGHT MODEL

chm = dsm_resampled - dem
chm[chm < 0] = 0


# 4. SMOOTHING

chm_smooth = gaussian_filter(chm, sigma=3)

# SET Parameters | Modify Accordingly 

min_height = 3

# 5. TREE DETECTION

distance = ndi.distance_transform_edt(chm_smooth > min_height)

coordinates = peak_local_max(
    distance,
    min_distance=3,
    threshold_abs=min_height
)

num_trees = len(coordinates)


# 6. EVALUATION OF FOREST METRICS (DECISION + COVER BASE)

pixel_size_x = abs(transform.a)
pixel_size_y = abs(transform.e)

area_per_pixel = pixel_size_x * pixel_size_y
image_area = chm.shape[0] * chm.shape[1] * area_per_pixel

density_per_ha = (num_trees / image_area) * 10000

if density_per_ha < 400:
    USE_WATERSHED = False
    print("\nOPEN CONIFER STAND → skipping watershed")
else:
    USE_WATERSHED = True
    print("\nMIXED / DENSE FOREST → applying watershed")

# 7. TREE PROCESSING

tree_data = []

if not USE_WATERSHED:

    for (row, col) in coordinates:
        height = chm_smooth[row, col]
        tree_data.append((row, col, height))

else:

    markers = np.zeros_like(chm_smooth, dtype=int)

    for i, (row, col) in enumerate(coordinates):
        markers[row, col] = i + 1

    labels = watershed(-chm_smooth, markers, mask=chm_smooth > min_height)

    for label_id in range(1, np.max(labels) + 1):
        mask = labels == label_id

        if not np.any(mask):
            continue

        crown_vals = chm_smooth[mask]

        max_height = np.max(crown_vals)

        crown_pixels = np.argwhere(mask)
        max_idx = np.argmax(crown_vals)

        row, col = crown_pixels[max_idx]

        tree_data.append((row, col, max_height))

tree_data = np.array(tree_data)

# 8. CONVERT TO COORDINATES

rows = tree_data[:, 0]
cols = tree_data[:, 1]
heights = tree_data[:, 2]

xs, ys = rasterio.transform.xy(transform, rows, cols)

xs = np.array(xs)
ys = np.array(ys)

# 9. FILTERING | MODIFY ACCORDINGLY

min_tree_height = 3
max_tree_height = 15

valid = (heights >= min_tree_height) & (heights <= max_tree_height)

xs = xs[valid]
ys = ys[valid]
heights = heights[valid]

if len(heights) > 5:
    mean_h = np.mean(heights)
    std_h = np.std(heights)

    valid2 = (heights >= mean_h - 2 * std_h) & (heights <= mean_h + 2 * std_h)

    xs = xs[valid2]
    ys = ys[valid2]
    heights = heights[valid2]

# 10. CANOPY COVER in (%)

pixel_area = abs(transform.a) * abs(transform.e)

if USE_WATERSHED:

    total_crown_pixels = 0

    for label_id in range(1, np.max(labels) + 1):
        mask = labels == label_id
        total_crown_pixels += np.sum(mask)

    crown_area_m2 = total_crown_pixels * pixel_area
    occupancy_percent = (crown_area_m2 / image_area) * 100

else:

    avg_crown_area = 4  # m² assumption
    crown_area_m2 = num_trees * avg_crown_area
    occupancy_percent = (crown_area_m2 / image_area) * 100


# FOREST COVER METRICS + CLASSIFICATION

print("\n=== FOREST COVER METRICS ===")
print(f"Estimated Canopy Cover: {occupancy_percent:.2f}%")

# Density classification
if occupancy_percent < 30:
    density_class = "LOW DENSITY 🌱"
elif occupancy_percent < 60:
    density_class = "MEDIUM DENSITY 🌿"
else:
    density_class = "HIGH DENSITY 🌳"

print(f"Forest density class: {density_class}")

# Visual bar
bar_length = 30
filled_length = int((occupancy_percent / 100) * bar_length)

bar = "█" * filled_length + "-" * (bar_length - filled_length)

print("\nCanopy Density Indicator:")
print(f"[{bar}] {occupancy_percent:.1f}%")

# 11. SAVE CSV

df = pd.DataFrame({
    "Tree_ID": np.arange(1, len(xs) + 1),
    "X": xs,
    "Y": ys,
    "Height": heights
})

output_csv = "C:/Users/Dimitris/Desktop/Adaptive_Tree_Detection.csv"
df.to_csv(output_csv, index=False)

print("\n=== OUTPUT ===")
print(f"Saved CSV: {output_csv}")
print(f"Final Tree Count: {len(df)}")

# 12. VISUALIZATION

plt.figure(figsize=(10, 8))

# DSM as background (resampled to match CHM grid)
plt.imshow(dsm_resampled, cmap='gray')

# Optional: overlay CHM transparency (better depth perception)
plt.imshow(chm_smooth, cmap='viridis', alpha=0.35)

# Tree tops
plt.scatter(
    cols[valid],
    rows[valid],
    c='red',
    s=10,
    label='Detected Trees'
)

plt.title("Adaptive UAV Tree Detection on DSM Background")

# North arrow
ax = plt.gca()
ax.annotate(
    'N',
    xy=(0.95, 0.15),
    xytext=(0.95, 0.05),
    arrowprops=dict(facecolor='black', width=3, headwidth=10),
    ha='center',
    va='center',
    fontsize=12,
    xycoords='axes fraction'
)

plt.legend()
plt.savefig("adaptive_tree_detection_dsm_overlay.png", dpi=300)
plt.show()


