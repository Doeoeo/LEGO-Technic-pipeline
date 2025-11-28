import cv2
import numpy as np
from PIL import Image

from Coloring.BlenderRunner import orthographicProjections
from Util.Drawer import voxels_to_list, drawVoxels
from Util.Support import trimZeros


def colorModel(name, voxelGrid, connectorGraph):
    dirDic = {0:(1, 2), 1:(0, 2), 2:(0, 1)}
    directions = ["-x", "+x", "+y", "-y", "-z", "+z"]

    def downsampleImage(image, voxelGrid, ax1, ax2):
        """
            Downsample the image to match the voxel grid dimensions.
        :param image: 2D NumPy array of the image
        :param voxelGrid: 3D NumPy array of the model
        :param ax1: [0, 1, 2] first axis of the point of view
        :param ax2: [0, 1, 2] second axis of the point of view
        :return: 3D NumPy array of the color
        """

        image = np.rot90(image, -1, (0,1))

        # Compute step size
        stepAx1 = np.round(np.linspace(0, image.shape[0], voxelGrid.shape[ax1] + 1)).astype(int)
        stepAx2 = np.round(np.linspace(0, image.shape[1], voxelGrid.shape[ax2] + 1)).astype(int)

        newImage = np.zeros((voxelGrid.shape[ax1], voxelGrid.shape[ax2], 3))
        for i in range(stepAx1.shape[0] - 1):
            for j in range(stepAx2.shape[0] - 1):
                x1 = stepAx1[i]
                x2 = stepAx1[i + 1]

                y1 = stepAx2[j]
                y2 = stepAx2[j + 1]

                # Determine the color
                # -- middle color --
                # newImage[i, j, :] = image[round((x2+x1)/2), round((y2+y1)/2), :]
                # Average color
                #
                arr = image[x1:x2, y1:y2, :]
                flat = arr.reshape(-1, arr.shape[2])
                uniques, counts = np.unique(flat, axis=0, return_counts=True)
                mode = uniques[counts.argmax()]
                newImage[i, j, :] = mode

        # cv2.imshow("Image", cv2.cvtColor(image.astype('uint8'), cv2.COLOR_HSV2BGR))
        # cv2.waitKey(0)
        return newImage

    def scanAlong(volume, axis, reverseDir=False):
        """
            Scans the volume along the given axis. Finds the indices of the first non-zero voxels.
        :param volume: 3D NumPy Boolean array of the volume.
        :param axis: [0, 1, 2] axis along which to scan.
        :param reverseDir: Boolean. Direction along which to scan.
        :return: 2D NumPy array of hit indices.
        """

        if reverseDir:
            # Flip volume to match direction
            voxelGridFlipped = np.flip(volume, axis=axis)
            # Find maximum value (all values are True/False so first is selected)
            idxReversed =  voxelGridFlipped.argmax(axis=axis)
            # Adjust the index because of flip
            idx = (volume.shape[axis] - 1) - idxReversed
        else:
            # Find maximum value (all values are True/False so first is selected)
            idx = volume.argmax(axis=axis)

        return idx

    def colorBricks(voxelGrid, colors, connectorGraph, colored):
        """
            Adds all possible colors to each brick in the graph.
        :param voxelGrid: volume
        :param colors: color volume
        :param connectorGraph: graph structure
        :return: set of all colored bricks
        """
        tmp = {}
        elements = set()
        # Find indices where the model exists and is colored
        idx = np.argwhere(colored)#& (voxelGrid > 0))
        zmax= 0
        a = set()
        for x,y,z in idx:
            k = voxelGrid[x, y, z]
            # if k != 1057: continue
            # Add color to the brick
            # connectorGraph.getBrick(voxelGrid[x, y, z]).addColor(colors[x, y, z])
            # Add brick to set for easier access
            elements.add(voxelGrid[x, y, z])
            l = tuple(colors[x, y, z, :])
            a.add(k)
            # print(f"{k} at {x},{y},{z} colored {l}")
            if k in tmp:
                if l in tmp[k]:
                    tmp[k][l] += 1
                else: tmp[k] = {l:1}
            else: tmp[k] = {l:1}
        print(a)
        return elements, tmp

    voxelGrid = np.rot90(voxelGrid, 1, (1,2))
    voxelGrid = np.rot90(voxelGrid, 2, (0,1))
    # Filter only LEGOS
    voxelGrid = trimZeros(voxelGrid)
    voxelGrid[voxelGrid > 0] = 0
    voxelGridOld = np.abs(voxelGrid)
    voxelGrid = voxelGrid  < 0
    # Color volume
    color = np.zeros(voxelGrid.shape + (3,))
    # Colored mask
    colored = np.zeros(voxelGrid.shape).astype(bool)
    # Get orthographic images
    images = orthographicProjections(name)
    counter = 0

    for axis in range(3):
        coloredStep = np.zeros(colored.shape).astype(bool)
        colorTemp = np.zeros(color.shape)
        for direction in (False, True):
            # Find model's surface along given direction
            idx = scanAlong(voxelGrid, axis, direction)
            # Downsample original image to match the volume's resolution
            img = downsampleImage(images[directions[counter]], voxelGrid, dirDic[axis][0], dirDic[axis][1])
            # Flip images that are oriented incorrectly
            if direction:
                img = np.flipud(img)
            if axis == 2:
                img = np.fliplr(img)

            mask = (idx > 0)
            cords = np.nonzero(mask)
            cords += (idx[cords[0], cords[1]],)

            # Rotate coordinates to match axis
            swappedCords = [0, 0, 0]
            swappedCords[axis] = cords[2]
            swappedCords[dirDic[axis][0]] = cords[0]
            swappedCords[dirDic[axis][1]] = cords[1]
            # cords = swappedCords

            # Add colors
            # img = np.flipud(np.rot90(img, -1, axes=(0,1)))
            colorTemp[swappedCords[0], swappedCords[1], swappedCords[2], :] = img[cords[0], cords[1], :]
            coloredStep[swappedCords[0], swappedCords[1], swappedCords[2]] = True
            colored[swappedCords[0], swappedCords[1], swappedCords[2]] = True
            if axis == 0:
                color[0, :, :, :] = img[:, :, :]
                voxelGrid[0, :, :] = 1
            elif axis == 1:
                color[:, 0, :, :] = img[:, :, :]
                voxelGrid[:, 0, :] = 1
            elif axis == 2:
                color[:, :, 0, :] = img[:, :, :]
                voxelGrid[:, :, 0] = 1
            counter += 1
        color[coloredStep] = colorTemp[coloredStep]


    # mask = np.zeros(colored.shape, dtype=bool)
    # mask = voxelGridOld == 513
    # color[mask] = np.array([150,250,250])
    # mask[:,:,38]=True
    # mask[:,:,33:37]=True
    # colored[~mask] = 0
    # color[~mask] = np.array([0,0,0])
    elements, tmp = colorBricks(voxelGridOld, color, connectorGraph, colored)
    # print(connectorGraph.getBrick(513))


    voxels = voxels_to_list(voxelGrid)
    # drawVoxels(voxels, [31,25,63], 10, colorVolume=color, testVox=voxelGridOld)
    return tmp

