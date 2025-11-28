import numpy as np
from collections import deque
import colorsys
import heapq
from typing import List, Tuple
import time

def is_point_between(p1, p2, p):
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p = np.array(p)
    
    # Compute direction vectors
    d = p2 - p1
    v = p - p1
    
    # Find the projection factor t
    if np.all(d == 0):
        # Line segment is a single point
        return np.all(p == p1)
    
    t = np.dot(v, d) / np.dot(d, d)
    
    # Check if the point lies within the segment
    return t

def saveObj(vertices, edges, name):
    print(f"Vertices: {len(vertices)}, edges: {len(edges)} saving")
    with open(f"{name}.obj", 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\n")
        for e in edges:
            f.write(f"l {e[0]} {e[1]}\n")
            
def rotations(arr, dict = False):
    """Generate all 90-degree rotations of a 3D numpy array."""
    rotations = []
    rotDict = {}
    # Original orientation
    i = 0
    ori = [1, 2, 3]
    for x_rot in range(4):
        oriTmp = ori.copy()
        arr_x = np.rot90(arr, x_rot, axes=(1, 2))
        for y_rot in range(4):
            arr_y = np.rot90(arr_x, y_rot, axes=(0, 2))
            for z_rot in range(4):
                oriTmp = ori.copy()
                if x_rot % 2 == 1: oriTmp[1], oriTmp[2] = oriTmp[2], oriTmp[1]
                if y_rot % 2 == 1: oriTmp[0], oriTmp[2] = oriTmp[2], oriTmp[0]
                if z_rot % 2 == 1: oriTmp[0], oriTmp[1] = oriTmp[1], oriTmp[0]    
                rotDict[i] = tuple(oriTmp)

                i += 1
                arr_z = np.rot90(arr_y, z_rot, axes=(0, 1))
                rotations.append(arr_z)
    
    if dict: return (rotations, rotDict)
    return rotations


def filterRotations(rotations):
    """Filter the list of rotations to remove duplicates."""
    uniqueIndices = []
    seen = set()

    for i in range(len(rotations)):
        rotation = rotations[i]
        # Convert array to bytes
        rotation_bytes = rotation.tobytes()
        # Check if this rotation has already been seen
        if rotation_bytes not in seen:
            uniqueIndices.append(i)
            seen.add(rotation_bytes)
    
    return uniqueIndices

rotate_x_90 = np.array([
    [1, 0, 0, 0],
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

rotate_y_90 = np.array([
    [0, 0, -1, 0],
    [0, 1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])

rotate_z_90 = np.array([
    [0, -1, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

def createRotationMatrix(rotations):
    rotate_x = np.linalg.matrix_power(rotate_x_90, rotations[0]) 
    rotate_y = np.linalg.matrix_power(rotate_y_90, rotations[1])
    rotate_z = np.linalg.matrix_power(rotate_z_90, rotations[2])
    return rotate_z @ (rotate_y @ rotate_x)

def createTransformationMatrices():
    """Create transformation matrices for each 90-degree rotation."""
    matrices = []
    
    # Identity matrix represents no rotation
    identity_matrix = np.eye(4)

    # Define basic rotation matrices

    
    # Generate all combinations of rotations
    for x_rot in range(4):
        # Apply X rotations
        x_matrix = np.linalg.matrix_power(rotate_x_90, x_rot)
        for y_rot in range(4):
            # Apply Y rotations
            y_matrix = np.linalg.matrix_power(rotate_y_90, y_rot)
            for z_rot in range(4):
                # Apply Z rotations
                z_matrix = np.linalg.matrix_power(rotate_z_90, z_rot)
                
                # Compute the full rotation matrix
                rotation_matrix = z_matrix @ (y_matrix @ x_matrix)
                matrices.append(rotation_matrix)
    
    return matrices

def rotateArray(arr:np.ndarray, rotations:tuple[int, int, int]) -> np.ndarray:
    """Rotates a given 3d np array by 90-degrees a number of times defined by rotations. Rotations are performed in order x->y->z."""
    axes = [(1, 2), (0, 2), (0, 1)]
    # rotations = reversed(rotations)
    # axes = reversed([(1, 2), (0, 2), (0, 1)])

    for r, a in zip(rotations, axes):
        arr = np.rot90(arr, r, axes=a)
    return arr


def padToCube(arr:np.ndarray) -> np.ndarray:
    """Pad a 3D numpy array with zeros to make it a cube."""
    # Determine the maximum size needed for each dimension
    max_dim = max(arr.shape)

    # Calculate padding for each dimension
    pad_width = [(0, max_dim - s) for s in arr.shape]

    # Pad the array to make it a cube
    padded_array = np.pad(arr, pad_width, mode='constant', constant_values=0)

    return padded_array

def generateCoordinates(arr):
    """Generate an array of coordinates for a 3D numpy array."""
    # Get the shape of the array
    shape = arr.shape

    # Generate indices for each dimension
    indices = np.indices(shape)

    # Reshape the indices to have the coordinates in the last dimension
    coordinates = np.stack(indices, axis=-1)
    cFlat = coordinates.reshape(-1, 3)
    coordinate_tuples = np.empty(len(cFlat), dtype=object)
    coordinate_tuples[:] =  [tuple(coords) for coords in cFlat]
    
    coordinate_tuples = coordinate_tuples.reshape(shape)

    return coordinate_tuples
   
def log(n):
    with open(f"../log.txt", 'w') as f:
        j=1
        for i in n:
            f.write(f"{j} {i.shape} Case: \n{np.array2string(i)}\n\n")
            j+=1



def find_closest_points(list1, list2, s, ban, distHeap:List[Tuple[int, Tuple[np.ndarray, np.ndarray]]]):
    min_distance = float('inf')
    closest_pair = (None, None)
    if distHeap is None:
        distHeap = []
        for point1 in list1:
            if s[tuple(point1)] < 1: continue
            for point2 in list2:
                if s[tuple(point2)] < 1: continue
                # Skip pairs that were checked that can not be connected
                # print( f"{tuple(np.concatenate((point1,point2)))} -> {tuple(np.concatenate((point1,point2))) in ban}")
                pointTuple = tuple(np.concatenate((point1,point2)))
                if pointTuple in ban: continue
                # Calculate squared Euclidean distance

                distance_sq = ((point2[0] - point1[0]) ** 2 +
                               (point2[1] - point1[1]) ** 2 +
                               (point2[2] - point1[2]) ** 2)

                heapq.heappush(distHeap, (distance_sq, (point1, point2)))
                # Update if a closer pair is found
                # if distance_sq < min_distance:
                #     min_distance = distance_sq
                #     closest_pair = (point1, point2)
    if len(distHeap) > 0:
        min_distance, closest_pair = heapq.heappop(distHeap)
        ban.add(closest_pair[1])

    # print(f"Found closest pair: {closest_pair} with dist {min_distance}")
    return closest_pair, min_distance, distHeap   # Return actual distance


def shortest_path_3d(grid, start, end):
    """
    Finds the shortest path in a 3D grid using BFS.
    
    Parameters:
        grid (np.ndarray): 3D NumPy array with values 1 (walkable) and 0 (not walkable).
        start (tuple): Starting coordinate (x, y, z).
        end (tuple): Target coordinate (x, y, z).
    
    Returns:
        list: List of coordinates representing the shortest path, or [] if no path exists.
    """
    if grid[start] != 1 or grid[end] != 1:
        return []  # No valid path if start or end is not walkable

    # Directions: 6 orthogonal neighbors in 3D
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    # BFS initialization
    queue = deque([start])  # Queue of coordinates to explore
    visited = set()         # Set of visited coordinates
    visited.add(start)
    parent = {start: None}  # To reconstruct the path

    while queue:
        current = queue.popleft()

        # Check if we reached the end
        if current == end:
            print("Path Found")
            # Reconstruct the path
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]  # Return the path in correct order

        # Explore neighbors
        for dx, dy, dz in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)

            # Check bounds and whether the neighbor is walkable and unvisited
            if (0 <= neighbor[0] < grid.shape[0] and
                0 <= neighbor[1] < grid.shape[1] and
                0 <= neighbor[2] < grid.shape[2] and
                neighbor not in visited and
                grid[neighbor] == 1):
                
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor] = current  # Track how we reached this node

    # print("No path found")
    return []  # Return empty list if no path exists

def flip(current_vector, desired_vector, to_vector = np.array([0, 0, 0])):
    rotate_x_90 = np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    rotate_y_90 = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    rotate_z_90 = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])

    # Combine rotations for each axis
    rotations = [0, 0, 0]

    # Align current axis with the desired axis
    if not np.allclose(current_vector, desired_vector):
        for axis, rotation_matrix in zip([0, 1, 2], [rotate_x_90, rotate_y_90, rotate_z_90]):
            for i in range(4):  # At most 4 rotations
                rotated_vector = np.linalg.matrix_power(rotation_matrix, i) @ current_vector
                if np.allclose(rotated_vector, desired_vector):
                    rotations[axis] = i
                    to_vector =  np.linalg.matrix_power(rotation_matrix, i) @ to_vector
                    current_vector = rotated_vector
                    break

    return (rotations, to_vector)

def toVec(axis, direction):
    v = np.zeros(3)
    v[axis] = direction
    return v


def trimZeros(volume):
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(volume))
    return volume[slices]


def color2hex(color):
    c = color
    color = colorsys.hsv_to_rgb(color[0] / 255, color[1] / 255, color[2] / 255)
    color = tuple((np.array(color) * 255).astype(int))
    return f"0x2{color[0]:02X}{color[1]:02X}{color[2]:02X}"

def weighted_avg_tuple(d):
    if not d:
        return (0.0, 0.0, 0.0)  # or raise
    w_sum = sum(d.values())
    x = sum(k[0]*w for k, w in d.items()) / w_sum
    y = sum(k[1]*w for k, w in d.items()) / w_sum
    z = sum(k[2]*w for k, w in d.items()) / w_sum
    return (x, y, z)


def solid_from_shell(shell):
    # shell: bool array where True are shell voxels (from -e)
    a = shell.astype(bool)
    p = np.pad(~a, 1, constant_values=True)   # treat outside as empty space
    ext = np.zeros_like(p, dtype=bool)
    q = deque([(0,0,0)])                      # start from padded corner (outside)
    ext[0,0,0] = True
    # 6-neighborhood flood fill of exterior empties
    nbrs = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    while q:
        x,y,z = q.popleft()
        for dx,dy,dz in nbrs:
            nx,ny,nz = x+dx, y+dy, z+dz
            if 0 <= nx < p.shape[0] and 0 <= ny < p.shape[1] and 0 <= nz < p.shape[2]:
                if not ext[nx,ny,nz] and p[nx,ny,nz]:  # empty and not yet visited
                    ext[nx,ny,nz] = True
                    q.append((nx,ny,nz))
    interior = ~ext[1:-1,1:-1,1:-1]          # voxels not reachable from outside
    solid = a | interior                      # shell + filled interior
    return solid