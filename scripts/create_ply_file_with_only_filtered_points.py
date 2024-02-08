import numpy as np
from plyfile import PlyData, PlyElement

# Load the PLY file
ply_data = PlyData.read('.\christmasTree_point_cloud_30000_with_p_rgb_trained.ply')

        

# Extract vertex data as a structured NumPy array
element = ply_data.elements[0]

# Convert structured array to regular numpy array for filtering
vertex_array = np.array([element[name] for name in element.data.dtype.names]).T

# Filter out points where p < 0.85
filtered_vertex_array = vertex_array[vertex_array[:, -4] >= 0.9]

# Remove the p, red, green, and blue attributes (assuming they are the last four columns)
filtered_vertex_array = filtered_vertex_array[:, :-4]

# Convert back to a structured array
dtype = element.data.dtype.descr[:-4]  # Remove last four columns from dtype
filtered_vertex_data = np.array([tuple(row) for row in filtered_vertex_array], dtype=dtype)

# Create a new PlyElement for the modified vertex data
filtered_vertex = PlyElement.describe(filtered_vertex_data, 'vertex')

# Write the new data to a PLY file
PlyData([filtered_vertex]).write('modified_file.ply')

print(f"Number of points in the new PLY file: {len(filtered_vertex_data)}")