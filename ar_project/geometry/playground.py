import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. LOAD POINT CLOUD
# -----------------------------
data = o3d.data.DemoColoredICPPointClouds()
pcd1 = o3d.io.read_point_cloud(data.paths[0])
pcd2 = o3d.io.read_point_cloud(data.paths[1])

print("Original point cloud:")
print(pcd1)

o3d.visualization.draw_geometries([pcd1], window_name="Original")

# -----------------------------
# 2. DOWNSAMPLE
# -----------------------------
pcd_down = pcd1.voxel_down_sample(voxel_size=0.02)

print("downsampled point cloud:")
print(pcd_down)

#o3d.visualization.draw_geometries([pcd_down], window_name="Downsampled")


# -----------------------------
# 3. ESTIMATE NORMALS
# -----------------------------
pcd_down.estimate_normals()
#o3d.visualization.draw_geometries([pcd_down], window_name="Normals", point_show_normal=True)

# -----------------------------
# 5. REMOVE NOISE
# -----------------------------
pcd_clean, ind = pcd_down.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)
#o3d.visualization.draw_geometries([pcd_clean], window_name="Noise Removed")


print("MIN BOUND: ", pcd1.get_min_bound())
print("MAX BOUND: ",pcd1.get_max_bound())

# -----------------------------
# 6. CROP
# -----------------------------
bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=pcd_clean.get_min_bound() + [0.1, 0.1, 0.1],
    max_bound=pcd_clean.get_max_bound() - [0.1, 0.1, 0.1]
)
pcd_crop = pcd_clean.crop(bbox)
o3d.visualization.draw_geometries([pcd_crop], window_name="Cropped")

# -----------------------------
# 7. TRANSFORM (ROTATE)
# -----------------------------
R = pcd_crop.get_rotation_matrix_from_xyz((0, np.pi/4, 0))
pcd_crop.rotate(R, center=(0, 0, 0))
o3d.visualization.draw_geometries([pcd_crop], window_name="Rotated")

# -----------------------------
# 8. BOUNDING BOX
# -----------------------------
bbox = pcd_crop.get_axis_aligned_bounding_box()
bbox.color = (0, 1, 0)

o3d.visualization.draw_geometries([pcd_crop, bbox], window_name="Bounding Box")

# -----------------------------
# 9. DISTANCE BETWEEN CLOUDS
# -----------------------------
dist = pcd1.compute_point_cloud_distance(pcd2)
print("Distance stats:")
print("Min:", np.min(dist))
print("Max:", np.max(dist))