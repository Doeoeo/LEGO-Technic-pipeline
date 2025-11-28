from re import A
from turtle import down
import numpy as np

from Util.Drawer import drawVoxels, voxels_to_list
from Skeletonization.Lines import toEdges


def countNeighbours(segment:np.ndarray, axis:int, prev = None):
    """
        Computes the number of foreground neigbours in both directions of a given axis (max 2 min 0)    
    Parameters:
        segment (np.ndarray): A 2D boolean NumPy array.
        axis (int): Axis in which to compute.
    Returns:
        segment (np.ndarray)
    """
    # Shift the array to the left and right and remove looped edges
    left = np.roll(segment, 1, axis = axis).astype(int)
    right = np.roll(segment, -1, axis = axis).astype(int)
    left[:, 0] = 0
    right[:, -1] = 0
    

    together = (left + right + segment) == 3
   
    return together

def countDiagonalNeighbours(segment:np.ndarray, axis:int, prev = None):
    """
        Computes the number of foreground neigbours in both directions of a given axis (max 2 min 0)    
    Parameters:
        segment (np.ndarray): A 2D boolean NumPy array.
        axis (int): Axis in which to compute.
    Returns:
        segment (np.ndarray)
    """
    down = np.roll(segment, 1, axis = 0).astype(int)
    up = np.roll(segment, -1, axis = 0).astype(int)
    right = np.roll(segment, 1, axis = 1).astype(int)
    left = np.roll(segment, -1, axis = 1).astype(int)
    axis = (0, 1)
    # Shift the array to the left and right and remove looped edges
    topRight = np.roll(segment, (-1, 1), axis = axis).astype(int)
    topLeft = np.roll(segment, (-1, -1), axis = axis).astype(int)
    botLeft = np.roll(segment, (1, -1), axis = axis).astype(int)
    botRight = np.roll(segment, (1, 1), axis = axis).astype(int)
    
    # This might need to also check for other options i.e.: botLeft+bot and topLeft+top
    downToUp1 = np.logical_and((botLeft + left) >= 1, (topRight + right) >= 1)
    downToUp2 = np.logical_and((botLeft + down) >= 1, (topRight + up) >= 1)
    upToDown1 = np.logical_and((topLeft + left) >= 1, (botRight + right) >= 1)
    upToDown2 = np.logical_and((topLeft + down) >= 1, (botRight + up) >= 1)

    together = np.logical_and(downToUp1 + upToDown1 + downToUp2 + upToDown2, segment)
    #together =  upToDown1

    
   


    return together

def thinX(voxelGrid:np.ndarray, order:list, neighbourFun = countNeighbours, show = False):
    """
        Thins the with a sliding plane. Each filled voxel removes its corresponding voxel in the next frame.    
    Parameters:
        voxelGrid (np.ndarray): A 3D boolean NumPy array.
        order (list[int]): direction of the sliding window.
        diagonalNo (int): Number of diagonal planes
        diagonalLen (int): Length of max diagonal plane
    Returns:
        None
    """    
    (xLen, yLen, zLen) = voxelGrid.shape
    prevFrameX = np.zeros((yLen, zLen)).astype(bool)
    prevFrameY = np.zeros((yLen, zLen)).astype(bool)
    prevFull =  np.zeros((yLen, zLen)).astype(bool)
    for i in range(len(order) -1):
        x = order[i]
        if i > 30 and show:
            a = np.array([voxelGrid[x+1, :, :], voxelGrid[x, :, :], prevFrameX + prevFrameY]).astype(bool)
            # print(np.shape(a))
            voxels = voxels_to_list(a > 0)
            drawVoxels(voxels, [31,25,63], 10)
            prevFull = np.copy(voxelGrid[x, :, :])
        frameX = neighbourFun(voxelGrid[x, :, :], 1, prevFrameX)
        frameY = neighbourFun(voxelGrid[x, :, :], 0, prevFrameY)
        voxelGrid[x][np.logical_and(prevFrameX + prevFrameY, voxelGrid[x + 1, :, :])] = False
        #voxelGrid[x][prevFrameY] = False
        prevFrameX = frameX
        prevFrameY = frameY    

def thinDiagonal(voxelGrid:np.ndarray, order:list, diagonalNo:int, diagonalLen:int, show = False, padSide = (1, 0), neighbourFun = countNeighbours):
    """
        Thins the with a diagonal sliding plane. Each filled voxel removes its corresponding voxel in the next frame.    
    Parameters:
        voxelGrid (np.ndarray): A 3D boolean NumPy array.
        order (list[int]): direction of the sliding window.
        diagonalNo (int): Number of diagonal planes
        diagonalLen (int): Length of max diagonal plane
    Returns:
        None
    """
    (xLen, yLen, zLen) = voxelGrid.shape
    y, x, z = np.meshgrid(np.arange(0, yLen, 1), np.arange(0, xLen, 1), np.arange(0, zLen, 1))
    prevMask = np.zeros(voxelGrid.shape).astype(bool)
    prevFrames = [np.zeros((1, zLen)).astype(bool), np.zeros((2, zLen)).astype(bool)]
    c = 0
    for i in order:
        mask = np.ma.masked_where(x + y == i, voxelGrid).mask
        maskNext = np.ma.masked_where(x + y == i + 2, voxelGrid).mask
        #frame = countNeighbours(voxelGrid[mask], 0)
        diagonalElements = len(voxelGrid[mask])
        cut = voxelGrid[mask].reshape((diagonalElements // zLen, -1))
        #cutNext = voxelGrid[maskNext].reshape((len(voxelGrid[maskNext]) // zLen, -1))
        #cutNext = adjustArray(cutNext, c, diagonalLen, diagonalNo, xLen, yLen)
        segment1 = neighbourFun(cut, 0)
        segment2 = neighbourFun(cut, 1)
        segment = segment1 + segment2
        #maskedSegment[prevFrames[c % 2]] = False
        
        voxelGrid[mask] = np.logical_and(cut.flatten(), ~(prevFrames[c % 2].flatten()))
        # if c > 67 and show:# and np.max(a) > 0:
        #     print(f"{c} -> |{segment1.shape} {prevFrames[c % 2].shape} | {diagonalNo} - {diagonalLen} = {diagonalNo - diagonalLen}")
            #a = np.copy(voxelGrid)
            #a[(mask)] = True
            #voxels = voxels_to_list(a > 0)
            #
            #drawVoxels(voxels, [31,25,63], 10)
            #a = np.array([cutNext, cut, prevFrames[c % 2]])#prevFrames[c % 2]])
            #voxels = voxels_to_list(a > 0)
            #drawVoxels(voxels, [31,25,63], 10)
        if c + 2 < diagonalLen: prevFrames[c % 2] = np.pad(segment, ((1, 1), (0, 0)))
        elif c + 2 == diagonalLen: 
            if (xLen + yLen) % 2 == 0: prevFrames[c % 2] = segment #np.pad(segment, ((1, 0), (0, 0))) MIGHT BE WRONG MIGHT BE IF ONE IS ODD 
            else: prevFrames[c % 2] = np.pad(segment, (padSide, (0, 0))) 
        elif c + 1 < diagonalNo - diagonalLen: 
            if i != c: prevFrames[c % 2] = np.pad(segment[:-1, :], ((1, 0), (0, 0)))
            else: prevFrames[c % 2] = np.pad(segment[1:, :], ((0, 1), (0, 0)))
        elif c + 1 == diagonalNo - diagonalLen: prevFrames[c % 2] = segment[1:, :] 
        else: prevFrames[c % 2] = segment[1:-1, :]   
        c += 1

def adjustArray(a:np.ndarray, c:int, diagonalLen:int, diagonalNo:int, xLen:int, yLen:int) -> np.ndarray:
        if c + 2 < diagonalLen: return a[1:-1, :]
        elif c + 2 == diagonalLen: 
            if xLen + yLen % 2 == 0:return a #np.pad(segment, ((1, 0), (0, 0))) MIGHT BE WRONG MIGHT BE IF ONE IS ODD 
            else: return a[1:, :] 
        elif c + 1 < diagonalNo - diagonalLen: 
            return np.pad(a[:-1, :], ((1, 0), (0, 0)))
        elif c + 1 == diagonalNo - diagonalLen: return np.pad(a, ((1, 0), (0, 0)))
        else: return np.pad(a, ((1, 1), (0, 0)))     
        

def thinXWrapper(voxelGrid:np.ndarray):
    """
        Wrapper for calling thinX    
    Parameters:
        voxelGrid (np.ndarray): A 3D boolean NumPy array.
    Returns:
        voxelGrid (np.ndarray)
    """
    voxelGrid1 = np.copy(voxelGrid)
    #voxelGrid2 = np.copy(voxelGrid)
    (xLen, yLen, zLen) = voxelGrid.shape
    thinX(voxelGrid1, list(range(xLen)))
    #thinX(voxelGrid2, reversed(list(range(xLen))))
    return voxelGrid1
    #return np.logical_or(voxelGrid1, voxelGrid2)

def thinDiagonalWrapper(voxelGrid:np.ndarray, kwargs) -> np.ndarray:
    """
        Wrapper for calling thinDiagonal    
    Parameters:
        voxelGrid (np.ndarray): A 3D boolean NumPy array.
    Returns:
        voxelGrid (np.ndarray)
    """ 
    voxelGrid1 = np.copy(voxelGrid)
    voxelGrid2 = np.copy(voxelGrid)
    (xLen, yLen, zLen) = voxelGrid.shape
    diagonalNo = xLen + yLen - 1
    diagonalLen = min(xLen, yLen)
    thinDiagonal(voxelGrid2, reversed(list(range(0, diagonalNo))), diagonalNo, diagonalLen, **kwargs)
    kwargs.update({'padSide':(0, 1), 'show': False})
    thinDiagonal(voxelGrid1, list(range(0, diagonalNo)), diagonalNo, diagonalLen, **kwargs)
    return voxelGrid1 + voxelGrid2

def squeeze(voxelGrid:np.ndarray):
    voxels = np.where(voxelGrid > 0)
    voxels = list(zip(voxels[0], voxels[1], voxels[2]))
    mid = np.array(voxelGrid.shape) // 2
    
    template = np.array([False, True, True, False])
    # Apply thinning by x, y, z axis
    voxels = sorted(voxels, key=lambda x: voxelGrid[x], reverse=True)
    i = 0
    for v in voxels:
        
        if voxelGrid[v] < 1: continue
        neighbors1 = voxelGrid[v[0] - 2:v[0] + 2, v[1],              v[2]]
        neighbors2 = voxelGrid[v[0],              v[1] - 2:v[1] + 2, v[2]]
        neighbors3 = voxelGrid[v[0],              v[1],              v[2] - 2:v[2] + 2]
        #print(f"{i} pre SUM:{np.sum(voxelGrid)} -> |{v} |{neighbors1}---{neighbors2}---{neighbors3}")
        if i == 190:
            print("lasda")
        if np.all(np.array_equal(neighbors1 > 0, template)):
            #print(f"Removed")
            
            v2 = v + np.array([-1, 0, 0])
            if sum(abs(v - mid)) < sum(abs(v2 - mid)): 
                voxelGrid[tuple(v2)] = 0
            else: 
                voxelGrid[v] = 0
                continue                
        elif np.all(np.array_equal(neighbors2 > 0, template)):
            #print(f"Removed")
            v2 = v + np.array([0, -1, 0])
            if sum(abs(v - mid)) < sum(abs(v2 - mid)): voxelGrid[tuple(v2)] = 0
            else: 
                voxelGrid[v] = 0
                continue
        elif np.all(np.array_equal(neighbors3 > 0, template)):
            #print(f"Removed")
            v2 = v + np.array([0, 0, -1])
            if sum(abs(v - mid)) < sum(abs(v2 - mid)): voxelGrid[tuple(v2)] = 0
            else: 
                voxelGrid[v] = 0
    
        #print(f"SUM:{np.sum(voxelGrid)} -> |{v} |{neighbors1}---{neighbors2}---{neighbors3}")
        i += 1
    return voxelGrid

def reduce(voxelGrid:np.ndarray) -> np.ndarray:
    """
        Reduces the given voxel grid so it consists of one voxel thick lines    
    Parameters:
        voxelGrid (np.ndarray): A 3D boolean NumPy array.
    Returns:
        voxelGrid (np.ndarray)
    """

    voxelGrid = thinXWrapper(voxelGrid)
    voxelGrid = np.rot90(voxelGrid, 1)
    voxelGrid = thinXWrapper(voxelGrid)
    voxelGrid = np.rot90(voxelGrid, 1, (0,2))
    voxelGrid = thinXWrapper(voxelGrid)   
    voxelGrid = np.rot90(voxelGrid, -1, (0,2))
    voxelGrid = np.rot90(voxelGrid, -1)
    
  
    
    voxelGrid = thinDiagonalWrapper(voxelGrid, {'show': True})
    voxelGrid = np.rot90(voxelGrid, 1)
    voxelGrid = thinDiagonalWrapper(voxelGrid, {})
    voxelGrid = np.rot90(voxelGrid, 1, (0, 2))
    voxelGrid = thinDiagonalWrapper(voxelGrid, {})
    voxelGrid = np.rot90(voxelGrid, 1)
    voxelGrid = thinDiagonalWrapper(voxelGrid, {})
    voxelGrid = np.rot90(voxelGrid, 1, (0, 2))
    voxelGrid = thinDiagonalWrapper(voxelGrid, {})
    voxelGrid = np.rot90(voxelGrid, 1)
    voxelGrid = thinDiagonalWrapper(voxelGrid, {})

    
    voxelGrid = thinDiagonalWrapper(voxelGrid, {'neighbourFun': countDiagonalNeighbours})
    voxelGrid = np.rot90(voxelGrid, -1)
    voxelGrid = thinDiagonalWrapper(voxelGrid, {'neighbourFun': countDiagonalNeighbours})
    voxelGrid = np.rot90(voxelGrid, -1, (0, 2))
    voxelGrid = thinDiagonalWrapper(voxelGrid, {'neighbourFun': countDiagonalNeighbours})
    voxelGrid = np.rot90(voxelGrid, -1)
    voxelGrid = thinDiagonalWrapper(voxelGrid, {'neighbourFun': countDiagonalNeighbours})
    voxelGrid = np.rot90(voxelGrid, -1, (0, 2))
    voxelGrid = thinDiagonalWrapper(voxelGrid, {'neighbourFun': countDiagonalNeighbours})
    voxelGrid = np.rot90(voxelGrid, -1)
    voxelGrid = thinDiagonalWrapper(voxelGrid, {'neighbourFun': countDiagonalNeighbours})

    #voxels = voxels_to_list(voxelGrid > 0)
    #drawVoxels(voxels, [31,25,63], 10)

    
    #toEdges(voxelGrid)

    return voxelGrid
    