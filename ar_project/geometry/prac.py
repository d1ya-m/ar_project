import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


pcd = o3d.io.read_point_cloud(
    "D:/Documents/COLLEGE/PROJECTS/AR VR/ar_project/datasets/ITC_groundfloor.ply")

pcd_center = pcd.get_center()
pcd.translate(-pcd_center)

o3d.visualization.draw_geometries([pcd])

#random sampling
retained_ratio = 0.2
sampled_pcd = pcd.random_down_sample(retained_ratio)
o3d.visualization.draw_geometries([sampled_pcd], window_name = "Random Sampling")

#removing outliers
nn = 16
std_multiplier = 10

filtered_pcd, filtered_idx = pcd.remove_statistical_outlier(nn, std_multiplier)

outliers = pcd.select_by_index(filtered_idx, invert=True)
outliers.paint_uniform_color([1, 0, 0])
o3d.visualization.draw_geometries([filtered_pcd, outliers])

#voxel downsample

voxel_size = 0.05
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)

#normals
nn_distance = 0.05
radius_normals=nn_distance*4
pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normals, max_nn=16), fast_normal_computation=True)

pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6])
o3d.visualization.draw_geometries([pcd_downsampled,outliers])

#ransac
nn_distance = np.mean(pcd.compute_nearest_neighbor_distance())

distance_threshold = 0.05
ransac_n = 3
num_iterations = 1000

plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,ransac_n=3,num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert=True)

#Paint the clouds
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])

#Visualize the inliers and outliers
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name = "RANSAC")

#multi order ransac!!
segment_models={}
segments={}

max_plane_idx=10

rest=pcd

for i in range(max_plane_idx):
    colors = plt.get_cmap("tab20")(i)
    segment_models[i], inliers = rest.segment_plane(
    distance_threshold=0.1,ransac_n=3,num_iterations=1000)
    segments[i]=rest.select_by_index(inliers)
    # add dbscan here 
    segments[i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers, invert=True)
    print("pass",i,"/",max_plane_idx,"done.")
o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest],window_name = "MULTIORDER RANSAC")

#eucledian clustering refinement 

o3d.visualization.draw_geometries([rest],window_name = "Rest")
labels = np.array(rest.cluster_dbscan(eps=0.15, min_points=10))
max_label = labels.max()
colors = plt.get_cmap("tab20")(labels / (max_label 
if max_label > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([rest])

#eucledian clustering to prevent red lines/planes cutting planar elements
#DBSCAN: density based spatial clustering of applications with noise 
# diff from KMEANS as no of clusters dont have to be predefined and clusters of any shape possible
# good for detecting shapes, doesnt keep all inliners only largest cluster in each plane and throws the rest

max_plane_idx = 5
distance_threshold = 0.02

epsilon = 0.1
min_cluster_points = 5

segments = {}
segment_models = {}
rest = pcd

for i in range(max_plane_idx):

    print(f"\n--- Pass {i} ---")

    plane_model, inliers = rest.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=1000)

    if len(inliers) < 100:
        print("Too few inliers, stopping")
        break
    segment_models[i] = plane_model
    segment = rest.select_by_index(inliers)
    labels = np.array(segment.cluster_dbscan(eps=epsilon, min_points=min_cluster_points))
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels >= 0]  
    if len(unique_labels) == 0:
        print("No clusters found, skipping")
        rest = rest.select_by_index(inliers, invert=True)
        continue
    best_label = max(unique_labels, key=lambda x: np.sum(labels == x))
    best_cluster = segment.select_by_index(np.where(labels == best_label)[0])
    other_clusters = segment.select_by_index(np.where(labels != best_label)[0])
    rest = rest.select_by_index(inliers, invert=True) + other_clusters
    segments[i] = best_cluster
    color = plt.get_cmap("tab20")(i)
    segments[i].paint_uniform_color(color[:3])

    print(f"Plane {i}: {len(best_cluster.points)} points")

o3d.visualization.draw_geometries(list(segments.values()) + [rest])

#voxelization and labeling

voxel_size=0.1

min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()

pcd_ransac=o3d.geometry.PointCloud()
for i in segments:
 pcd_ransac += segments[i]

pcd_ransac = pcd_ransac.voxel_down_sample(0.02)
rest = rest.voxel_down_sample(0.02)

voxel_grid_structural = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_ransac, voxel_size=voxel_size)
rest.paint_uniform_color([0.1, 0.1, 0.8])
voxel_grid_clutter = o3d.geometry.VoxelGrid.create_from_point_cloud(rest, voxel_size=voxel_size)
o3d.visualization.draw_geometries([voxel_grid_clutter,voxel_grid_structural])

# indoor spatial modeling 

def fit_voxel_grid(point_cloud, voxel_size, min_b=False, max_b=False):
# This is where we write what we want our function to do.
    # Determine the minimum and maximum coordinates of the point cloud
    if type(min_b) == bool or type(max_b) == bool:
        min_coords = np.min(point_cloud, axis=0)
        max_coords = np.max(point_cloud, axis=0)
    else:
        min_coords = min_b
        max_coords = max_b
    # Calculate the dimensions of the voxel grid
    grid_dims = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    # Create an empty voxel grid
    voxel_grid = np.zeros(grid_dims, dtype=bool)
    # Calculate the indices of the occupied voxels
    indices = ((point_cloud - min_coords) / voxel_size).astype(int)
    # Mark occupied voxels as True
    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    return voxel_grid, indices

voxel_size = 0.3

#get the bounds of the original point cloud
min_bound = pcd.get_min_bound()
max_bound = pcd.get_max_bound()

ransac_voxels, idx_ransac = fit_voxel_grid(pcd_ransac.points,voxel_size, min_bound, max_bound)
rest_voxels, idx_rest = fit_voxel_grid(np.asarray(rest.points), voxel_size, min_bound, max_bound)

#Gather the filled voxels from RANSAC Segmentation
filled_ransac = np.transpose(np.nonzero(ransac_voxels))

#Gather the filled remaining voxels (not belonging to any segments)
filled_rest = np.transpose(np.nonzero(rest_voxels))

#Compute and gather the remaining empty voxels
total = pcd_ransac + rest
total_voxels, idx_total = fit_voxel_grid(total.points, voxel_size, min_bound, max_bound)
empty_indices = np.transpose(np.nonzero(~total_voxels))

voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_ransac, voxel_size=voxel_size)

#Open3D VoxelGrid
o3d.visualization.draw_geometries([voxel_grid])


## voxels -> point cloud
def voxel_indices_to_points(indices, voxel_size, min_bound):
    return indices * voxel_size + min_bound


# convert voxel indices → 3D points
ransac_points = voxel_indices_to_points(filled_ransac, voxel_size, min_bound)
rest_points = voxel_indices_to_points(filled_rest, voxel_size, min_bound)

# create point clouds
pcd_ransac_vox = o3d.geometry.PointCloud()
pcd_ransac_vox.points = o3d.utility.Vector3dVector(ransac_points)
pcd_ransac_vox.paint_uniform_color([1, 0, 0])  # red

pcd_rest_vox = o3d.geometry.PointCloud()
pcd_rest_vox.points = o3d.utility.Vector3dVector(rest_points)
pcd_rest_vox.paint_uniform_color([0, 0, 1])  # blue

# visualize: red-planes, blue-objects
o3d.visualization.draw_geometries([pcd_ransac_vox, pcd_rest_vox])

empty_points = voxel_indices_to_points(empty_indices, voxel_size, min_bound)

pcd_empty = o3d.geometry.PointCloud()
pcd_empty.points = o3d.utility.Vector3dVector(empty_points)
pcd_empty.paint_uniform_color([0.7, 0.7, 0.7])

##visualize empty voxels
o3d.visualization.draw_geometries([pcd_empty])

#EXPORTING 3D DATASETS

xyz_segments=[]
for idx in segments:
 print(idx,segments[idx])
 a = np.asarray(segments[idx].points)
 N = len(a)
 b = idx*np.ones((N,3+1))
 b[:,:-1] = a
 xyz_segments.append(b)

 rest_w_segments=np.hstack((np.asarray(rest.points),(labels+max_plane_idx).reshape(-1, 1)))

 xyz_segments.append(rest_w_segments)

 #np.savetxt("../RESULTS/" + DATANAME.split(".")[0] + ".xyz", np.concatenate(xyz_segments), delimiter=";", fmt="%1.9f")

 #voxel model export

 def voxel_modelling(filename, indices, voxel_size):
    voxel_assembly=[]
    with open(filename, "a") as f:
        cpt = 0
        for idx in indices:
            voxel = cube(idx,voxel_size,cpt)
            f.write(f"o {idx}  \n")
            np.savetxt(f, voxel,fmt='%s')
            cpt += 1
            voxel_assembly.append(voxel)
    return voxel_assembly
 
vrsac = voxel_modelling("../RESULTS/ransac_vox.obj", filled_ransac, 1)
vrest = voxel_modelling("../RESULTS/rest_vox.obj", filled_rest, 1)
vempty = voxel_modelling("../RESULTS/empty_vox.obj", empty_indices, 1)