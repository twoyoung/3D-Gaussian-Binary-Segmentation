import open3d as o3d
import numpy as np
import colorsys



# Load the point cloud
# pcloud = o3d.io.read_point_cloud("../christmasTree_point_cloud_7000_for_open3d_no_p_color_integer.ply")



# o3d.visualization.draw_geometries([pcloud])


from plyfile import PlyData, PlyElement
import numpy as np
import open3d as o3d

# Load the PLY file
ply = PlyData.read('./christmasTree_point_cloud_30000_with_p_rgb_trained.ply')
vertex = ply['vertex']

# Extract position data
xyz = np.stack((vertex['x'], vertex['y'], vertex['z']), axis=-1).astype(np.float32)

# Extract and normalize color data
rgb = np.stack((vertex['red'], vertex['green'], vertex['blue']), axis=-1).astype(np.float32)

# Extract 'p' data
p = np.array(vertex['p']).astype(np.float32)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)
# pcd.p = o3d.utility.Vector3dVector(p)


# filter function to get the points of the tree
def pass_through_filter(bbox, pcloud):
    points = np.asarray(pcloud.points)
    colors = np.asarray(pcloud.colors)
    points[:,0]
    x_range = np.logical_and(points[:,0] >= bbox["x"][0] ,points[:,0] <= bbox["x"][1])
    y_range = np.logical_and(points[:,1] >= bbox["y"][0] ,points[:,1] <= bbox["y"][1])
    z_range = np.logical_and(points[:,2] >= bbox["z"][0] ,points[:,2] <= bbox["z"][1])

    enclosed = np.logical_and(x_range,np.logical_and(y_range,z_range))

    pcloud.points = o3d.utility.Vector3dVector(points[enclosed])
    pcloud.colors = o3d.utility.Vector3dVector(colors[enclosed])

    return pcloud

# Get the point cloud of the tree
bounding_box = {"x":[-2.2,1.5], "y":[-4.6,3.6], "z":[-1.5,3]}
pcd = pass_through_filter(bounding_box, pcd)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd], 
                                  zoom=0.56333333333333435,
                                  front=[ 0.025167187841641528, 0.51823250195133597, -0.85486939738032619 ],
                                  lookat=[ -0.34997761249542236, 0.60990989208221436, 0.75132882595062256 ],
                                  up=[ -0.001390132354422236, -0.85512128876625715, -0.51842612687901135 ])

####################### filter by color ###############################################
def rgb_to_hsv(rgb):
    # Normalize RGB values to [0, 1] range
    rgb_normalized = rgb / 255.0
    hsv_normalized = np.apply_along_axis(lambda x: colorsys.rgb_to_hsv(*x), 1, rgb_normalized)
    return hsv_normalized * np.array([360, 1, 1])  # Scale H to [0, 360] and S, V to [0, 1]

# Function to filter the points by color values
def filter_color_hsv(pcloud, min_hsv, max_hsv):
    # Extract RGB colors and convert to HSV
    colors = np.asarray(pcloud.colors) * 255  # Denormalize to [0, 255]
    hsv_colors = rgb_to_hsv(colors)

    # Apply the HSV color range filter
    hsv_filter = np.all(np.logical_and(min_hsv <= hsv_colors, hsv_colors <= max_hsv), axis=1)

    # Select points and colors that are inside the HSV color range
    filtered_points = np.asarray(pcloud.points)[hsv_filter]
    filtered_colors = colors[hsv_filter] / 255  # Renormalize to [0, 1]

    # Create a new point cloud for the filtered data
    filtered_pcloud = o3d.geometry.PointCloud()
    filtered_pcloud.points = o3d.utility.Vector3dVector(filtered_points)
    filtered_pcloud.colors = o3d.utility.Vector3dVector(filtered_colors)

    return filtered_pcloud

# Define the color range (normalized RGB values)
min_hsv = np.array([200, 0.2, 0.2])  # Minimum color (e.g., black)
max_hsv = np.array([260, 1.0, 1.0])  # Maximum color (e.g., dark gray)

# Filter the point cloud by color
filtered_pcloud = filter_color_hsv(pcd, min_hsv, max_hsv)

filtered_pcloud, _ = filtered_pcloud.remove_statistical_outlier(nb_neighbors=15, std_ratio=4.0)

o3d.visualization.draw_geometries([filtered_pcloud])


##################################### filter by p ###############################################


# Filter points where 'p' value is greater than 0.9
mask = p > 0.8
filtered_xyz = xyz[mask]
filtered_rgb = rgb[mask]

# Create new Open3D point cloud with filtered data
filtered_pcd = o3d.geometry.PointCloud()
filtered_pcd.points = o3d.utility.Vector3dVector(filtered_xyz)
filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_rgb)

bounding_box = {"x":[-2.2,1.5], "y":[-4.6,3.6], "z":[-1.5,3]}
filtered_pcd = pass_through_filter(bounding_box, filtered_pcd)

# Visualize the filtered point cloud
o3d.visualization.draw_geometries([filtered_pcd],
                                  zoom=1.0801666666666663,
                                  front=[ 0.025167187841641528, 0.51823250195133597, -0.85486939738032619 ],
                                  lookat=[ -0.34997761249542236, 0.60990989208221436, 0.75132882595062256 ],
                                  up=[ -0.001390132354422236, -0.85512128876625715, -0.51842612687901135 ])


# {
# 	"class_name" : "ViewTrajectory",
# 	"interval" : 29,
# 	"is_loop" : false,
# 	"trajectory" : 
# 	[
# 		{
# 			"boundingbox_max" : [ 0.66975677013397217, 3.4441173076629639, 1.5454154014587402 ],
# 			"boundingbox_min" : [ -2.0628492832183838, 0.21736598014831543, 0.10764311254024506 ],
# 			"field_of_view" : 60.0,
# 			"front" : [ 0.025167187841641528, 0.51823250195133597, -0.85486939738032619 ],
# 			"lookat" : [ -0.34997761249542236, 0.60990989208221436, 0.75132882595062256 ],
# 			"up" : [ -0.001390132354422236, -0.85512128876625715, -0.51842612687901135 ],
# 			"zoom" : 1.0801666666666663
# 		}
# 	],
# 	"version_major" : 1,
# 	"version_minor" : 0
# }