from scipy.ndimage import binary_erosion
from itertools import combinations
from math import floor
import open3d as o3d
import numpy as np
import pyvista as pv
import plotly.graph_objects as go
from Util.Drawer import drawVoxels, voxels_to_list
from Util.Support import padToCube, solid_from_shell
import vtk
# import vtk.util
from vtkmodules.util import numpy_support
import cv2
# from vtk.util import numpy_support
from Skeletonization.thinner import reduce


#from mayavi import mlab

def voxToCloud(voxels, name="test"):
    with open(f"Objects/{name}.dat", 'w') as f:
        for vox in voxels:
            v = vox.grid_index
            f.write(f"{v[0]}, {v[1]}, {v[2]}\n")

def draw_voxel(bool_array, voxel_size=15):
    # Extract voxel positions where True
    x, y, z = np.where(bool_array)

    # Compute distances from a hypothetical light source (adjust for shading effect)
    distances = np.sqrt((x - 5)**2 + (y - 5)**2 + (z - 5)**2)

    # Normalize distances to [0, 1]
    distances = (distances - distances.min()) / (distances.max() - distances.min())

    # Map distances to opacity values in the range [0, 1]
    opacity = 1 - distances

    # Create a trace for voxels with adjusted transparency for shading effect
    voxel_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            color='blue',  # You can customize the color here
            size=voxel_size,  # Adjust the size of the markers as needed
            opacity=opacity.tolist()  # Convert opacity values to a list
        )
    )

    # Create a layout
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )

    # Create a figure
    fig = go.Figure(data=[voxel_trace], layout=layout)

    # Show the plot
    fig.show()

def loadObj():
    ## Load the .obj file using open3d
    mesh = o3d.io.read_triangle_mesh("C:/Users/Uporabnik/Desktop/Mag/Skeletonization/Objects/Cottage_03.obj")

    # Define voxel grid parameters
    voxel_size = 20  # Voxel size in meters (adjust according to your needs)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    ## Get the minimum and maximum bounds of the voxel grid
    min_bound, max_bound = voxel_grid.get_min_bound(), voxel_grid.get_max_bound()
    
    ## Calculate the size (dimensions) of the voxel grid
    voxel_dim = (voxel_grid.get_max_bound() - voxel_grid.get_min_bound()) // voxel_size
    voxel_dim = voxel_dim.astype(int) + 1
    
    ## Print the voxel grid size
    print("Voxel Grid Size:", voxel_dim)

    voxels = voxel_grid.get_voxels()
    testVox = np.zeros(tuple(voxel_dim + 2))
    for v in voxels:
        idx = v.grid_index
        testVox[idx[0] + 1, idx[1] + 1, idx[2] + 1] = 1
        
    return testVox

def createTest():
    # Define the dimensions of the voxel grid
    grid_size = 31 # Adjust the size as needed to accommodate the pyramid

    # Create an empty 3D NumPy array filled with zeros (representing empty space)
    voxel_grid = np.zeros((grid_size, grid_size * 3, grid_size), dtype=int)

    # Build the pyramid by setting voxels to True for each step
    for step in  range(floor(grid_size / 2)):
        step_size = 2 * step + 1  # Voxel size for the current step
        z_start = floor(grid_size / 2) - step  # Starting Z-coordinate for the current step
        z_end = z_start + step_size  # Ending Z-coordinate for the current step
    
        # Set the voxels in the current step to True
        voxel_grid[z_start:z_end, 3*step:3*step + 3, z_start:z_end] = 1
        
    #return np.flip(voxel_grid, 1)
    return voxel_grid

def createTest2():
    # Define the dimensions of the voxel grid
    grid_size = 5  # Adjust the size as needed to accommodate the pyramid

    # Create an empty 3D NumPy array filled with zeros (representing empty space)
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=int)


    voxel_grid[0:grid_size, 0:grid_size, 0:grid_size] = 1
        
    #return np.flip(voxel_grid, 1)
    return voxel_grid



def voxelType(voxel_grid, pos, boundary, inbound):
    neigbors = (voxel_grid[pos[0] - 1:pos[0] + 2, pos[1] - 1:pos[1] + 2, pos[2] - 1:pos[2] + 2] > 0)
    k = neigbors.sum()
    k = (neigbors[1, 1, :] > 0).sum() + (neigbors[1, :, 1] > 0).sum() + (neigbors[:, 1, 1] > 0).sum() - 3
    if k == 6:
        inbound.append(pos)
        #k = 6
    else:
        boundary.append(pos)
    voxel_grid[pos] = k

def voxelTypeAll(voxel_grid, pos, boundary, inbound):
    neigbors = (voxel_grid[pos[0] - 1:pos[0] + 2, pos[1] - 1:pos[1] + 2, pos[2] - 1:pos[2] + 2] > 0)
    k = neigbors.sum() - 1
    voxel_grid[pos] = k
    
def find_boundary(voxel_grid, typeFun):
    voxels = np.where(voxel_grid > 0)
    voxels = list(zip(voxels[0], voxels[1], voxels[2]))
    boundary = []
    inbound = []
    for i in voxels: typeFun(voxel_grid, i, boundary, inbound)    
    return (boundary, inbound)
   #return (list(reversed(sorted(boundary))), inbound)

def remove_Vertex(pos, voxel_grid, neighbors):
    n_faces = np.zeros((3,3,3), dtype = bool)
    n_faces[1, 1, (0, 2)] = True
    n_faces[1, (0, 2), 1] = True
    n_faces[(0, 2), 1, 1] = True
    a = np.zeros((3,3,3))
    a[n_faces] = neighbors[n_faces]
    bdry = np.where(a == 6)
    if len(bdry[0]) == 0: bdry = []
    else: bdry = np.array(list(zip(bdry[0], bdry[1], bdry[2]))) - 1 + pos
    neighbors[n_faces] -= 1
    neighbors[1, 1, 1]  = 0
    n = voxel_grid[pos[0] - 1:pos[0] + 2, pos[1] - 1:pos[1] + 2, pos[2] - 1:pos[2] + 2]
    #voxel_grid[pos] = 0
    
    return bdry

def rotate(neighbors):
     np.rot90(neighbors)
  
def test_k1(red, yellow, green):
    if np.any(red > 0): return False
    for y, g in zip(yellow, green):
        if g < 1 and y > 0: return False
    return True

faces = np.zeros((3,3,3), dtype = bool)
faces[1, 1, (0, 2)] = True
faces[1, (0, 2), 1] = True
faces[(0, 2), 1, 1] = True

def checkK1(neighbors, voxel_grid, v, neighbors_new):
    p = False
    if neighbors[0, 1, 1] >= 1: 
        p = p or test_k1(np.concatenate((neighbors[2, 1, (0, 2)], neighbors[2, (0, 2), 1])), np.concatenate((neighbors[2, 2, (0, 2)], neighbors[2, 0, (0, 2)])), np.concatenate((neighbors[1, 2, (0, 2)], neighbors[1, 0, (0, 2)])))  # W
    if neighbors[2, 1, 1] >= 1: 
        p = p or test_k1(np.concatenate((neighbors[0, 1, (0, 2)], neighbors[0, (0, 2), 1])), np.concatenate((neighbors[0, 2, (0, 2)], neighbors[0, 0, (0, 2)])), np.concatenate((neighbors[1, 2, (0, 2)], neighbors[1, 0, (0, 2)])))  # E
    if neighbors[1, 1, 0] >= 1: 
        p = p or test_k1(np.concatenate((neighbors[(0, 2), 1, 2], neighbors[1, (0, 2), 2])), np.concatenate((neighbors[2, (0, 2), 2], neighbors[0, (0, 2), 2])), np.concatenate((neighbors[2, (0, 2), 1], neighbors[0, (0, 2), 1])))  # S            
    if neighbors[1, 1, 2] >= 1: 
        p = p or test_k1(np.concatenate((neighbors[(0, 2), 1, 0], neighbors[1, (0, 2), 0])), np.concatenate((neighbors[2, (0, 2), 0], neighbors[0, (0, 2), 0])), np.concatenate((neighbors[2, (0, 2), 1], neighbors[0, (0, 2), 1])))  # N         
    if neighbors[1, 0, 1] >= 1: 
        p = p or test_k1(np.concatenate((neighbors[(0, 2), 2, 1], neighbors[1, 2, (0, 2)])), np.concatenate((neighbors[(0, 2), 2, 2], neighbors[(0, 2), 2, 0])), np.concatenate((neighbors[(0, 2), 1, 2], neighbors[(0, 2), 1, 0])))  # Pb
    elif neighbors[1, 2, 1] >= 1: 
        p = p or test_k1(np.concatenate((neighbors[(0, 2), 0, 1], neighbors[1, 0, (0, 2)])), np.concatenate((neighbors[(0, 2), 0, 2], neighbors[(0, 2), 0, 0])), np.concatenate((neighbors[(0, 2), 1, 2], neighbors[(0, 2), 1, 0]))) #Pt
            
    if p:
        #print("removed k = 1")
        return list(remove_Vertex(v, voxel_grid, neighbors_new))
    global faces
    if np.any(np.where(neighbors[faces] == 6)): return [v]
    else: return [v]

def checkK2(neighbors, voxel_grid, v, neighbors_new):
    coordinates = [2, 1, 0, 1]
    red = [[[1, 1, 0], [2, 0, 1]], [[2, 0, 1], [1, 1, 2]], [[1, 1, 2], [2, 0, 1]], [[2, 0, 1], [1, 1, 0]]]
    yellow = [[[0, 0], [2, 0]], [[0, 2], [2, 0]], [[2, 2], [2, 0]], [[0, 2], [0, 0]]]
    green = []
    opposite = [2, 1, 0, 1]
    for i in range(3):
        for j in (0, 2):
            pB = np.roll(np.array([j, 1, 1]), i)
            if neighbors[pB[0], pB[1], pB[2]] < 6: continue
            for x, x2, r, y  in zip(coordinates, reversed(coordinates), red, yellow):
                pT = np.roll(np.array([1, x, x2]), i)            
                if neighbors[pT[0], pT[1], pT[2]] < 1: continue
                tar_dim = 1 if i == 2 else 0
                r_idx = np.array([r[0], r[tar_dim], r[1]]).T
                y_idx = np.array([y[0], y[tar_dim], y[1]]).T
                g_idx = np.array([y[0], y[tar_dim], y[1]]).T
                r_idx[:, i] = 0 if j == 2 else 2
                y_idx[:, i] = 0 if j == 2 else 2
                g_idx[:, i] = 1
                
                if test_k1(neighbors[r_idx[:,0], r_idx[:,1], r_idx[:,2]], neighbors[y_idx[:,0], y_idx[:,1], y_idx[:,2]], neighbors[g_idx[:,0], g_idx[:,1], g_idx[:,2]]):
                    #print("removed k=2")
                    return list(remove_Vertex(v, voxel_grid, neighbors_new))
    global faces
    if np.any(np.where(neighbors[faces] == 6)): return [v]
    else: return [v]
   
# support function for CheckK3
def flip(x): 
    if x == 0: return 2
    if x == 2: return 0
    return x
flipNP = np.vectorize(flip)

def checkK3(neighbors, voxel_grid, v, neighbors_new):
    coordinates = [2, 1, 0, 1]   
    global flipNP
    for i in range(3):
        for j in (0, 2):
            pB = np.roll(np.array([j, 1, 1]), i)
            if neighbors[pB[0], pB[1], pB[2]] < 6: continue
            c1 = 0
            for x1, x2 in zip(coordinates, reversed(coordinates)):
                c1 += 1
                pT1 = np.roll(np.array([1, x1, x2]), i)            
                if neighbors[pT1[0], pT1[1], pT1[2]] < 1: continue
                c2 = c1
                for y1, y2 in zip(coordinates[c1:], reversed(coordinates[:-c1])):
                    c2 += 1
                    pT2 = np.roll(np.array([1, y1, y2]), i)            
                    if neighbors[pT2[0], pT2[1], pT2[2]] < 1: continue 
                    tar_dim = 1 if i == 0 else 0
                    r_idx = np.array([flipNP(pT1), flipNP(pT2)])
                    r_idx[:, i] =  0 if j == 2 else 2
                    y_coord = 3 - pT1[tar_dim] - pT2[tar_dim]
                    g_idx = [y_coord, y_coord, 3 - pT1[2] - pT2[2]]
                    y_idx = [y_coord, y_coord, 3 - pT1[2] - pT2[2]]
                    y_idx[i] = 0 if j == 2 else 2
                    g_idx[i] = 1
                    if sum((pT1 - pT2) == 0) == 2:
                        idx = [0, 1, 2]
                        idx.remove(i)
                        r_idx = np.array([pT1, pT2])
                        r_idx[:, idx] = r_idx[:, list(reversed(idx))]
                        if test_k1(neighbors[r_idx[:,0], r_idx[:,1], r_idx[:,2]], [], []):
                        #print("removed k=3")
                            return list(remove_Vertex(v, voxel_grid, neighbors_new))
                    elif test_k1(neighbors[r_idx[:,0], r_idx[:,1], r_idx[:,2]], [neighbors[y_idx[0], y_idx[1], y_idx[2]]], [neighbors[g_idx[0], g_idx[1], g_idx[2]]]):
                        #print("removed k=3")
                        return list(remove_Vertex(v, voxel_grid, neighbors_new)) 
    global faces
    if np.any(np.where(neighbors[faces] == 6)): return [v]
    else: return [v]

def checkK4(neighbors, voxel_grid, v, neighbors_new):
    coordinates = [2, 1, 0, 1]
    blue = [[1, 2, 1], [1, 1, 0], [1, 0, 1], [1, 1, 2]]
    for i in range(3):
        for j in (0, 2):
            pB = np.roll(np.array([j, 1, 1]), i)
            if neighbors[pB[0], pB[1], pB[2]] < 6: continue
            for x, y, z in combinations(list(map(lambda arr: np.roll(arr, i), blue)), 3):
                if neighbors[x[0], x[1], x[2]] < 1 or neighbors[y[0], y[1], y[2]] < 1 or neighbors[z[0], z[1], z[2]] < 1: continue
                rx_coord = 4 - x[0] - y[0] - z[0]
                ry_coord = 4 - x[1] - y[1] - z[1]
                rz_coord = 4 - x[2] - y[2] - z[2]
                r_idx = [rx_coord, ry_coord, rz_coord]
                r_idx[i] = 0 if j == 2 else 2
                if test_k1(neighbors[r_idx[0], r_idx[1], r_idx[2]], [], []):
                        a = list(remove_Vertex(v, voxel_grid, neighbors_new))
                        #voxels = voxels_to_list(voxel_grid > 0)
                        #drawVoxels(voxels, [30, 30, 30], 30, testVox)
                        return a
    global faces
    if np.any(np.where(neighbors[faces] == 6)): return [v]
    else: return [v]

def checkK5(neighbors, voxel_grid, v, neighbors_new):
    for i in range(3):
        for j in (0, 2):
            pB = np.roll(np.array([j, 1, 1]), i)
            pO = np.roll(np.array([2 - j, 1, 1]), i)
            p1 = np.roll(np.array([1, 0, 0]), i)
            p2 = np.roll(np.array([1, 0, 2]), i)
            p3 = np.roll(np.array([1, 2, 0]), i)
            p4 = np.roll(np.array([1, 2, 2]), i)
            if neighbors[pB[0], pB[1], pB[2]] < 6:continue# or neighbors[p1[0], p1[1], p1[2]] < 1 or neighbors[p2[0], p2[1], p2[2]] < 1 or neighbors[p3[0], p3[1], p3[2]] < 1 or neighbors[p4[0], p4[1], p4[2]] < 1: continue
            elif neighbors[pO[0], pO[1], pO[2]] < 1:
                return list(remove_Vertex(v, voxel_grid, neighbors_new))
            
    global faces
    if np.any(np.where(neighbors[faces] == 6)): return [v]
    else: return [v]

c = -1
def prune(voxel_grid, boundary):
    global c
    #voxel_grid_new = np.copy(voxel_grid)
    voxel_grid_new = (voxel_grid)
    new_boundary = []

    for v in boundary:
        neighbors = voxel_grid[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        neighbors_new = voxel_grid_new[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        k = voxel_grid[v]
        n = voxel_grid[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        c += 1
        p = False
        # if c == 4700:
        #     print("a")
        if k == 1:
            #continue
            new_boundary += map(tuple, checkK1(neighbors, voxel_grid, v, neighbors_new))
        elif k == 2:
            #continue
            new_boundary += map(tuple, checkK2(neighbors, voxel_grid, v, neighbors_new))
        elif k == 3:
            #continue
            new_boundary += map(tuple, checkK3(neighbors, voxel_grid, v, neighbors_new))                                    
        elif k == 4:
            #continue
            new_boundary += map(tuple, checkK4(neighbors, voxel_grid, v, neighbors_new))
        elif k == 5:
            new_boundary += map(tuple, checkK5(neighbors, voxel_grid, v, neighbors_new))
            
        #print(k, " removed: ", voxel_grid_new[v] == 0)
    if False and c >= 1 and c % 1 == 0: #check ~100k
        print(f"{c} k = {k}")
        t=voxel_grid_new[v]
        voxel_grid_new[v] = 1
        voxels = voxels_to_list(voxel_grid_new > 0)
        drawVoxels(voxels, [31,25,63], 15, voxel_grid_new, v)
        voxel_grid_new[v] = t
            
            #voxels = voxels_to_list(voxel_grid > 0)
            #drawVoxels(voxels, [64, 64, 64], 30, voxel_grid, v)
            #if c > 50000: c = 60000 -1
    return new_boundary, voxel_grid_new
 
import matplotlib.pyplot as plt


def tmpPlt(array):
    

    # Prepare data for plotting
    # Extract the indices of True and False values
    true_indices = np.argwhere(array)
    false_indices = np.argwhere(~array)

    # Create a 3D plot
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')

    # Plot True values
    ax.scatter(true_indices[:, 0], true_indices[:, 1], true_indices[:, 2], color='green', s=100, label='True')

    # Plot False values
    ax.scatter(false_indices[:, 0], false_indices[:, 1], false_indices[:, 2], color='red', s=100, label='False')




    plt.show()
    
def getSkeleton(voxel_grid, full = True):
    return voxel_grid
    voxel_grid_new = np.copy(voxel_grid)
    voxels = np.where(voxel_grid > 0)
    voxels = zip(voxels[0], voxels[1], voxels[2])


    # Diagonal
    wall2 = np.zeros((3, 3, 3), dtype=bool)
    wall2[0, (0, 1, 2), 0] = True
    wall2[1, (0, 1, 2), 1] = True
    wall2[2, (0, 1, 2), 2] = True
       
    steps = [wall2]
    for i in range(1, 4): steps.append(np.rot90(wall2, i, (0, 1)))
    steps.append(np.rot90(wall2, 1, (0, 2)))
    for i in range(1, 4): steps.append(np.rot90(steps[4], i, (0, 1)))
    steps.append(np.rot90(wall2, 2, (0, 2)))
    for i in range(1, 4): steps.append(np.rot90(steps[8], i, (0, 1)))
    steps.append(np.rot90(wall2, 3, (0, 2)))
    for i in range(1, 4): steps.append(np.rot90(steps[12], i, (0, 1)))
    steps.append(np.rot90(wall2, 1, (1, 2)))
    for i in range(1, 4): steps.append(np.rot90(steps[16], i, (0, 1)))
    steps.append(np.rot90(wall2, 3, (1, 2)))
    for i in range(1, 4): steps.append(np.rot90(steps[20], i, (0, 1)))
    
    # Sparse Diagonal
    wall2 = np.zeros((3, 3, 3), dtype=bool)
    wall2[(0, 0, 1, 1, 1, 2, 2), 
          (1, 2, 0, 1, 2, 0, 1), 
          (2, 1, 2, 1, 0, 1, 0)] = True
    sparse = [wall2]
    for i in range(1, 4): sparse.append(np.rot90(wall2, i, (0, 1)))
    sparse.append(np.rot90(wall2, 1, (0, 2)))
    for i in range(1, 4): sparse.append(np.rot90(sparse[4], i, (0, 1)))
    sparse.append(np.rot90(wall2, 2, (0, 2)))
    for i in range(1, 4): sparse.append(np.rot90(sparse[8], i, (0, 1)))
    sparse.append(np.rot90(wall2, 3, (0, 2)))
    for i in range(1, 4): sparse.append(np.rot90(sparse[12], i, (0, 1)))
    sparse.append(np.rot90(wall2, 1, (1, 2)))
    for i in range(1, 4): sparse.append(np.rot90(sparse[16], i, (0, 1)))
    sparse.append(np.rot90(wall2, 3, (1, 2)))
    for i in range(1, 4): sparse.append(np.rot90(sparse[20], i, (0, 1)))
    
    # Straight
    wall4 = np.zeros((3, 3, 3), dtype=bool)
    wall4[:, (0, 1), :] = True
    walls = [wall4]
    for i in range(1, 4): walls.append(np.rot90(wall4, i, (0, 1)))
    walls.append(np.rot90(wall4, 1, (1, 2)))
    walls.append(np.rot90(wall4, 3, (1, 2)))

    wall5 = np.zeros((3, 3, 3), dtype=bool)
    wall5[:, 1, :] = True
    #wall5[(0, 2), 1, 0] = False
    #wall5[(0, 2), 1, 1] = False
    
    thinWalls = [wall5]
    thinWalls.append(np.rot90(wall5, 1, (0, 1)))
    thinWalls.append(np.rot90(wall5, 1, (1, 2)))
    thinWalls.append(np.rot90(thinWalls[-1], 1, (0, 1)))
    

    
    
    # L shape
    #wall6 = np.zeros((3, 3, 3), dtype=bool)
    #wall6[:, 1, 1] = True
    #wall6[:, (0, 1), 0] = True
    #lWalls = [wall6]
    ##1-3 (0, 1)
    #for i in range(1, 4): lWalls.append(np.rot90(wall6, i, (0, 1)))
    ##(0,2) 1-3 (1, 2)
    #for i in range(1, 4): lWalls.append(np.rot90(np.rot90(wall6, 1 , (0,2)), i, (1,2)))
    ##2x(0,2) 1-3 (0, 1)
    #for i in range(1, 4): lWalls.append(np.rot90(np.rot90(wall6, 2 , (0,2)), i, (0,1)))
    ##3x(0,2) 1-3 (1, 2)
    #for i in range(1, 4): lWalls.append(np.rot90(np.rot90(wall6, 3 , (0,2)), i, (1,2)))
    ##1x(1,2) 1-3 (0, 2)
    #for i in range(1, 4): lWalls.append(np.rot90(np.rot90(wall6, 1 , (1,2)), i, (0,2)))
    ##3x(1,2) 1-3 (0, 2)
    #for i in range(1, 4): lWalls.append(np.rot90(np.rot90(wall6, 3 , (1,2)), i, (0,2)))
    #

    #tmpPlt(np.rot90(np.rot90(wall6, 1 , (0,2)), 1, (1,2)))
    #tmpPlt(np.rot90(np.rot90(wall6, 1 , (0,2)), 1, (1,2)))

    #tmpPlt(np.rot90(wall6, 0 , (0,1)))
    
    #negatives
    walls_n = [np.invert(i) for i in walls]
    thinWalls_n = [np.invert(i) for i in thinWalls]
    steps_n = [np.invert(i) for i in steps]
    sparse_n = [np.invert(i) for i in sparse]
    #lWalls_n = [np.invert(i) for i in lWalls]
    
    c = 0
    print("Cutting")
    for v in voxels:
        removed = False
        neighbors = voxel_grid[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        neighbors_new = voxel_grid_new[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        c += 1
        if full:
            if not removed:
                for i in range(len(steps)): 
                    if np.all(neighbors[steps[i]] > 0) and np.all(neighbors[steps_n[i]] < 1): 
                        #remove_Vertex(v, voxel_grid_new, neighbors_new)
                        voxel_grid_new[v] = 0
                        break
            if not removed:
                for i in range(len(sparse)):
                    if np.all(neighbors[sparse[i]]) > 0 and np.all(neighbors[sparse_n[i]] < 1): 
                        #remove_Vertex(v, voxel_grid_new, neighbors_new)                   
                        voxel_grid_new[v] = 0
                        break  
        if not removed:
            for i in range(len(thinWalls)):
                if np.all(neighbors[thinWalls[i]]) > 0 and (np.all(neighbors[thinWalls_n[i]] < 1) or not full): 
                    #remove_Vertex(v, voxel_grid_new, neighbors_new)                   
                    voxel_grid_new[v] = 0
                    break
        if False and c % 200 == 0 and voxel_grid_new[v] == 0:
            voxel_grid_new[v] = 1
            voxels = voxels_to_list(voxel_grid_new > 0)
            drawVoxels(voxels, [30, 30, 30], 30, voxel_grid, v)
            voxel_grid_new[v] = 0
            print("xD")
    
    return voxel_grid_new
 
def pr(testVox, b, center):
    
    b_old = set([])
    b_new = set(b)
    while b_new != b_old:
        b, testVox = prune(testVox, sorted(list(b), key=lambda x: testVox[x]))
        #b, testVox = prune(testVox, sorted(list(b), key=lambda x: abs(x[1] - center[1]/2)))
        #b, testVox = prune(testVox, sorted(list(b), key=lambda x: np.max(np.divide(np.absolute(np.subtract(x, center)), center))))
        # print("pruned")
        b_old = b_new
        b_new = set(b)
        
    return testVox

def padArray(testVox):
    dims = testVox.shape
    zeros_planeX = np.zeros((1, dims[1], dims[2]))
    testVox = np.concatenate((zeros_planeX, testVox, zeros_planeX), axis=0)
    zeros_planeY = np.zeros((dims[0] + 2, 1, dims[2]))
    testVox = np.concatenate((zeros_planeY, testVox, zeros_planeY), axis=1)
    zeros_planeZ = np.zeros((dims[0] + 2, dims[1] + 2, 1))
    testVox = np.concatenate((zeros_planeZ, testVox, zeros_planeZ), axis=2)
    
    return testVox

def skeletonization(testVox):
    nz = np.nonzero(testVox)  # Indices of all nonzero elements
    testVox = testVox[nz[0].min():nz[0].max()+1, 
                      nz[1].min():nz[1].max()+1,
                      nz[2].min():nz[2].max()+1]
    testVox = padToCube(testVox)

    testVox = padArray(testVox)
    testVox = padArray(testVox)
    testVox = padArray(testVox)
    original = np.copy(testVox)


    b, b2 = find_boundary(testVox, voxelType)
    b_original = b.copy()
    #voxels = voxels_to_list(testVox > 5)

    voxels1 = np.copy(testVox)
    voxels2 = np.copy(testVox)
    center = np.divide(testVox.shape, 2)
    a1 = pr(voxels1, b, center) > 0
    
    # 0 - width, 1 - depth, 2 - height
    #a2 = pr(voxels2, list(sorted(b, key=lambda x: x[2], reverse = True))) > 0
    #voxels = voxels_to_list(a2 > 0)
    #a1[int(a1.shape[0] / 2):, :, :] = a2[int(a2.shape[0] / 2):, :, :]
    #a1 = np.logical_and(a1, a2)
    voxels = voxels_to_list(a1 > 0)
    #drawVoxels(voxels, [30, 30, 30], 12, a1)
    #a1[:, :, 0:int(a1.shape[2] / 2)] = 1
    #a1[:, :, 0:int(a1.shape[2] / 2)] = 1
    #drawVoxels(voxels, [30, 30, 30], 12, a2)
    #testVox = np.logical_and(a1, a2).astype(int)
    testVox = a1
  
    
    skeleton = getSkeleton(testVox)
    t = np.zeros(skeleton.shape)
    for i in b_original:
        if skeleton[i] == 1: continue
        t[i] = 1
    

    
    return (original>0, testVox, np.logical_and(original>0, ~testVox), t)
    #voxels = voxels_to_list(t > 0)

    #voxels = voxels_to_list(testVox > 0)
    #drawVoxels(voxels, [30, 30, 30], 30)

def vtkToNumpy(data):
    temp = numpy_support.vtk_to_numpy(data.GetPointData().GetScalars())
    dims = data.GetDimensions()

    component = data.GetNumberOfScalarComponents()
    if component == 1:
        numpy_data = temp.reshape(dims[2], dims[1], dims[0])
        numpy_data = numpy_data.transpose(2,1,0)
    elif component == 3 or component == 4:
        if dims[2] == 1: # a 2D RGB image
            numpy_data = temp.reshape(dims[1], dims[0], component)
            numpy_data = numpy_data.transpose(0, 1, 2)
            numpy_data = np.flipud(numpy_data)
        else:
            raise RuntimeError('unknow type')
    return numpy_data


def cutoff(voxelGrid, ):
    voxels = np.where(voxelGrid > 0)
    voxels = list(zip(voxels[0], voxels[1], voxels[2]))
    voxels = sorted(voxels, key=lambda x: voxelGrid[x], reverse=True)
    
    template = np.array([False, True, True, False])
    
    
def thin(voxelGrid, short = True):

    
    voxels = np.where(voxelGrid > 0)
    voxels = list(zip(voxels[0], voxels[1], voxels[2]))
    
    template = np.array([False, True, True, False])
    # Apply thinning by x, y, z axis
    voxels = sorted(voxels, key=lambda x: voxelGrid[x], reverse=True)
    for v in voxels:
        neighbors1 = voxelGrid[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        neighbors2 = voxelGrid[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 2:v[2] + 1]
        neighbors3 = voxelGrid[v[0], v[1], v[2] - 2:v[2] + 2]
        
        
        if(np.all(np.array_equal(neighbors1 > 0, template))) or np.all(np.array_equal(neighbors2, template)) or np.all(np.array_equal(neighbors3, template)):
            voxelGrid[v] = 0
    
          
    return voxelGrid

def thin3(voxelGrid):
    voxels = np.where(voxelGrid > 0)
    voxels = zip(voxels[0], voxels[1], voxels[2])
    
    template = np.array([False, True, True, True, False])
    voxels = sorted(list(voxels), key=lambda x: voxelGrid[x], reverse=True)
    for v in voxels:
        neighbors = voxelGrid[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        neighbors1 = voxelGrid[v[0] - 2:v[0] + 3, v[1], v[2]] > 0
        neighbors2 = voxelGrid[v[0], v[1] - 2:v[1] + 3, v[2]] > 0
        neighbors3 = voxelGrid[v[0], v[1], v[2] - 2:v[2] + 3] > 0
        
        
        if(np.all(np.array_equal(neighbors3, template))):# or np.all(np.array_equal(neighbors2, template)) or np.all(np.array_equal(neighbors3, template))):
            #remove_Vertex(v, voxelGrid, neighbors)
            voxelGrid[v[0], v[1], v[2] - 2:v[2] + 3] = 0
            voxelGrid[v] = 1
       
    voxels = np.where(voxelGrid > 0)
    voxels = zip(voxels[0], voxels[1], voxels[2])
    voxels = sorted(list(voxels), key=lambda x: voxelGrid[x], reverse=True)
    for v in voxels:
        neighbors = voxelGrid[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        neighbors1 = voxelGrid[v[0] - 2:v[0] + 3, v[1], v[2]] > 0
        neighbors2 = voxelGrid[v[0], v[1] - 2:v[1] + 3, v[2]] > 0
        neighbors3 = voxelGrid[v[0], v[1], v[2] - 2:v[2] + 3] > 0
        
        
        if(np.all(np.array_equal(neighbors1, template))):# or np.all(np.array_equal(neighbors3, template))):
            #remove_Vertex(v, voxelGrid, neighbors)
            voxelGrid[v[0] - 2:v[0] + 2, v[1], v[2]] = 0
            voxelGrid[v] = 1
    
    return voxelGrid    
            
    voxels = np.where(voxelGrid > 0)    
    voxels = zip(voxels[0], voxels[1], voxels[2])
    voxels = sorted(list(voxels), key=lambda x: voxelGrid[x], reverse=True)
    for v in voxels:
        neighbors = voxelGrid[v[0] - 1:v[0] + 2, v[1] - 1:v[1] + 2, v[2] - 1:v[2] + 2]
        neighbors1 = voxelGrid[v[0] - 2:v[0] + 3, v[1], v[2]] > 0
        neighbors2 = voxelGrid[v[0], v[1] - 2:v[1] + 3, v[2]] > 0
        neighbors3 = voxelGrid[v[0], v[1], v[2] - 2:v[2] + 3] > 0
        
        
        if(np.all(np.array_equal(neighbors2, template))):
            #remove_Vertex(v, voxelGrid, neighbors)
            voxelGrid[v[0], v[1] - 2:v[1] + 2, v[2]] = 0
            voxelGrid[v] = 1
            
    return voxelGrid    


def voxToObj(voxelGrid, name = "test"):
    """
    Stores a 3d np array of voxels into a .obj file
    
    Parameters:
        voxelGrid (np.ndarray): A 3D boolean NumPy array.
        name (String): Name of the file.
    Returns:
        None
    """
    voxelList = [tuple(i) for  i in np.argwhere(voxelGrid > 0)]
    vertices = {}
    i = 1
    for v in voxelList:
        vertices[v] = i
        i += 1
    
    edges = []
    for v in voxelList:
        #offsets = np.array([
        #    [ 1,  0,  0],
        #    [-1,  0,  0],
        #    [ 0,  1,  0],
        #    [ 0, -1,  0],
        #    [ 0,  0,  1],
        #    [ 0,  0, -1]
        #])
        #neighbors = [tuple(i) for i in np.array(v) + offsets]
        #count = 0
        #for n in neighbors:
        #    if n in vertices and vertices[n] < vertices[v]:
        #        count += 1
        #        edges.append((vertices[v], vertices[n]))
        #if count > 0: continue
        offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
        neighbors = [tuple(i) for i in np.array(v) + offsets]
        for n in neighbors:
            if n in vertices and vertices[n] < vertices[v]:
                #count += 1
                edges.append((vertices[v], vertices[n]))
                
    print(f"Vertices: {len(vertices)}, edges: {len(edges)} saving")
    
    with open(f"{name}.obj", 'w') as f:
        for v in list(vertices):
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\n")
        for e in edges:
            f.write(f"l {e[0]} {e[1]}\n")
        
        



def getBoundaryAndSkeleton(filename, shortname, skip = False):

#testVox = np.load("edgeHouse.npy").astype(bool)
#toEdges(testVox)
#sys.exit()
# binvox car1.obj -d 32 -t vtk

    # filename = "/Objects/car.vtk"
    # filename = "C:/Users/Uporabnik/Desktop/Mag/Skeletonization/Objects/Cottage_02_fixed.vtk"
    # testVox = loadObj()
    # testVox = createTest()
    # testVox = loadVox()
    print(f"Reading {filename}")
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()  # Read the file
    vtk_object = reader.GetOutput()
    testVox = vtkToNumpy(vtk_object)

    print(f"\tSkeletonization")

    testVox = solid_from_shell(testVox)
    voxels = voxels_to_list(testVox > 0)
    print(testVox.shape)
    # drawVoxels(voxels, [31, 25, 63], 10)

# skip = False

    # voxels = voxels_to_list(body + boundary + testVox > 0)
    if not skip:
        (original, testVox, body, boundary) = skeletonization(testVox)
        boundaryArray = np.zeros(original.shape)
        interior = binary_erosion(original, structure=np.ones((3, 3, 3), bool), border_value=False)
        boundary2 = original  & ~interior
        cutter = testVox.shape
        # testVox[:, 0:int(cutter[1] / 4 * 2), :] = 0
        # body[:, 0:int(cutter[1] / 4 * 2), :] = 0
        # boundary[:, 0:int(cutter[1] / 4 * 2), :] = 0
        testVox = reduce(testVox > 0)
        voxels = voxels_to_list(testVox > 0)
        boundary = (boundary > 0) | (boundary2) & ~(testVox > 0)
        boundary = boundary.astype(int)
        # boundary[:, 0:int(cutter[1] / 4 * 2), :] = 0

        voxToCloud(voxels, name=shortname)

        np.save(f"{filename}.npy", testVox)
        np.save(f"{filename}Body.npy", body)
        np.save(f"{filename}Boundary.npy", boundary)
    else:
        testVox = np.load(f"{filename}.npy").astype(int)
        body = np.load(f"{filename}Body.npy").astype(int)
        boundary = np.load(f"{filename}Boundary.npy").astype(int)
    voxels = voxels_to_list(testVox > 0)
    # drawVoxels(voxels, [31, 25, 63], 10)
    # return
    #

    # ------------- SKELETON --------------
    # MIGHT BE NEEDED TEST LATER FOR REDUCE
    #testVox = padArray(testVox)
    #testVox = padArray(testVox)
    #testVox = padArray(testVox)
    print("\tReduction")




    # drawVoxels(voxels, [31, 25, 63], 10)

    # ------------- Body ------------------
        #coverSegments(body, boundary)




    voxels = voxels_to_list(boundary > 0)
    # drawVoxels(voxels, [31, 25, 63], 10)
    return testVox, body, boundary
#




            
        


