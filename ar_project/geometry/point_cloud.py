import open3d as o3d


# Load sample point cloud
pcd_data = o3d.data.PCDPointCloud() 
pcd = o3d.io.read_point_cloud(pcd_data.path)

print(pcd)

# Visualize
o3d.visualization.draw_geometries([pcd])


# Load better dataset (multiple objects)
#data = o3d.data.DemoColoredICPPointClouds()
#pcd1 = o3d.io.read_point_cloud(data.paths[0])
#pcd2 = o3d.io.read_point_cloud(data.paths[1])



# Visualize
#o3d.visualization.draw_geometries([pcd1,pcd2])

#combining
#pcd_combined = pcd1 + pcd2
#o3d.visualization.draw_geometries([pcd_combined])


#pcd_down = pcd.voxel_down_sample(0.02)