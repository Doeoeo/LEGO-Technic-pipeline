import colorsys

import numpy as np
import open3d as o3d

def appendGridLinesToGrid(voxelGrid):
    # add x
    voxelGrid[0:15, 0, 0] = 1
    # add y
    voxelGrid[0, 0:10, 0] = 1
    # add z
    voxelGrid[0, 0, 0:5] = 1


def voxels_to_list(voxel_grid):
    """
    Convert a 3D boolean NumPy array voxel grid into a list of voxels with positions.

    Args:
        voxel_grid (numpy.ndarray): 3D boolean NumPy array representing a voxel grid.

    Returns:
        list: A list of voxel positions as tuples.
    """
    voxels = []
    dims = voxel_grid.shape
    
    for x in range(dims[0]):
        for y in range(dims[1]):
            for z in range(dims[2]):
                if voxel_grid[x, y, z]:
                    voxels.append(o3d.geometry.Voxel(np.array([x, y, z]), np.array([0, 0, 0])))
    
    return voxels


colors_dict = {
    0: [255, 0, 0],   # Red
    1: [0, 255, 0],   # Green
    2: [0, 0, 255],   # Blue
    3: [255, 255, 0], # Yellow
    4: [255, 0, 255], # Magenta
    5: [0, 255, 255], # Cyan
    6: [128, 128, 128] # Gray
}

def drawVoxels(voxels, voxel_dim, point_size = 12, testVox = None, i = [0, 0, 0], j = [0, 0, 0], colorVolume = None):
    ## Create a visualization window

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    colors = np.zeros((len(voxels), 3))
    voxel_centers = np.zeros((len(voxels), 3))
    tmp = set()
    for v in range(len(voxels)):
        colors[v][2] = abs(voxels[v].grid_index[2] / voxel_dim[2])
        colors[v][1] = abs(1 - voxels[v].grid_index[1] / voxel_dim[1])
        colors[v][0] = abs(1 - voxels[v].grid_index[0] / voxel_dim[0])
        idx = voxels[v].grid_index
        if not (testVox is None):
            colors[v][0] = colors_dict[(testVox[idx[0], idx[1], idx[2]]) %7][0]
            colors[v][1] = colors_dict[(testVox[idx[0], idx[1], idx[2]]) %7][1]
            colors[v][2] = colors_dict[(testVox[idx[0], idx[1], idx[2]]) %7][2]
        if not (colorVolume is None):
            color = colorVolume[idx[0], idx[1], idx[2]]
            # if color[1] == 0:
                # continue
                # colors[v][0] = 0
                # colors[v][1] = 1
                # colors[v][2] = 0
            # else:
            rgb = colorsys.hsv_to_rgb(color[0]/255, color[1] / 255, color[2] / 255)
            colors[v][0] = rgb[0]
            colors[v][1] = rgb[1]
            colors[v][2] = rgb[2]

        if np.array_equal(idx, i):
           colors[v] =  [255, 0, 0]
        if np.array_equal(idx, j):
           colors[v] =  [255, 255, 0]
        voxel_centers[v, :] = voxels[v].grid_index / 10


    # Create colored point cloud from voxel centers and colors
    colored_point_cloud = o3d.geometry.PointCloud()
    colored_point_cloud.points = o3d.utility.Vector3dVector(voxel_centers)
    colored_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Add the colored point cloud to the visualization
    vis.add_geometry(colored_point_cloud)

    # Set the rendering options
    render_options = vis.get_render_option()
    render_options.background_color = [0, 0, 0]  # Set the background color to black
    render_options.point_size = point_size

    # Run the visualization
    vis.run()
    


def drawSegment(voxelsZ, voxel_dim, point_size = 12, testVox = None, i = [0, 0, 0], j = [0, 0, 0], col = None):
    ## Create a visualization window
    voxels = voxels_to_list(voxelsZ != 0)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    colors = np.zeros((len(voxels), 3))
    voxel_centers = np.zeros((len(voxels), 3))
    for v in range(len(voxels)):
        colors[v][2] = abs(voxels[v].grid_index[2] / voxel_dim[2])
        colors[v][1] = abs(1 - voxels[v].grid_index[1] / voxel_dim[1])
        colors[v][0] = abs(1 - voxels[v].grid_index[0] / voxel_dim[0])
        idx = voxels[v].grid_index
        #colors[v][0] = colors_dict[testVox[idx[0], idx[1], idx[2]]][0]
        #colors[v][1] = colors_dict[testVox[idx[0], idx[1], idx[2]]][1]
        #colors[v][2] = colors_dict[testVox[idx[0], idx[1], idx[2]]][2]
        if voxelsZ[tuple(idx)] < 0:
            if voxelsZ[tuple(idx)] in col:
                colors[v] = col[voxelsZ[tuple(idx)]]
            else: colors[v] = (1,0,0)
        voxel_centers[v, :] = voxels[v].grid_index / 10


    # Create colored point cloud from voxel centers and colors
    colored_point_cloud = o3d.geometry.PointCloud()
    colored_point_cloud.points = o3d.utility.Vector3dVector(voxel_centers)
    colored_point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Add the colored point cloud to the visualization
    vis.add_geometry(colored_point_cloud)

    # Set the rendering options
    render_options = vis.get_render_option()
    render_options.background_color = [0, 0, 0]  # Set the background color to black
    render_options.point_size = point_size

    # Run the visualization
    vis.run()