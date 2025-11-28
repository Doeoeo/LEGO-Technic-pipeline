import numpy as np

from ToLegos.Bricks import bricks, specialConnectors, Brick
from Util.Drawer import voxels_to_list, drawVoxels, appendGridLinesToGrid
from Util.Support import createTransformationMatrices, rotateArray, rotations
from scipy.ndimage import convolve


def ldrToVox(name, pinGrid, volumeGrid, filename, skip=False):
    """
        Opens and reads a given .ldr file and converts all pieces to the voxel space.
        :param name: Name of the .ldr file contained in the SavedStates folder.
        :param pinGrid: np array containing the pins of each piece (and their directions)
        :param volumeGrid: np array containing the ownership of each piece.
        :return: connectors. The two grid parameter are adjusted and returns a new np array representing possible connector positions.
    """
    idDict={
        32524:16615,
        32523:17141,
        # 43857:64289,
        40490:64289,
    }
    def loadLDR(name):
        """Opens the .ldr file and stores the data of each piece """
        with open(f"SavedStates/{name}.ldr", "r") as f:
            lines = [line.split(" ") for line in f.read().splitlines()]
            lineCoordinates = [(np.round(np.array(line[2:5], dtype=float)) / 20).astype(int) + 1 for line in lines]
            lineNames = [
                int(line[-1]
                         .replace("p_", "5") # 4p_
                         .replace("o_", "6") # 1o_
                         .replace("_double", "7") # _double
                         .replace("_", "8") # _double
                         .replace(".dat", "")) for line in lines
            ]
            print(lineCoordinates)
            lineRotations = [np.array(line[5:-1], dtype=float) for line in lines]

            return (lineCoordinates, lineRotations, lineNames)

    def rotToMat(v:np.ndarray) -> np.ndarray:
        """Strips the transformation matrix to only contains the rotation information."""
        rotMat = np.identity(4, dtype=int)
        rotMat[:3, :3] = v.reshape(3, 3)
        rotMat[rotMat == -0] = 0

        return rotMat

    def bricksToDict() -> dict:
        """Stores all bricks into a dictionary keyed by their index."""
        brickDict = {}
        for b in bricks:
            brickDict[b.idx] = b
        for b in specialConnectors:
            brickDict[b.idx] = b
        return brickDict

    def rotsToDict() -> dict:
        """Stores all rotations into a dictionary keyed by their byte values of their rotation matrices."""
        def indexToRotations(n):
            base4 = np.base_repr(n, base=4).zfill(6)
            return [int(digit) for digit in base4[-3:][:]]
        transformDict = {}
        transforms = np.stack(createTransformationMatrices(), axis=0)
        i = 0
        for t in transforms:
            tBytes = t.tobytes()
            if not tBytes in transformDict: transformDict[tBytes] = indexToRotations(i)
            i += 1

        return transformDict

    def placeBrick(brick:Brick, pos, idx, pins, volume, pivot, rotation):
        def pinRemap(rotation, pins):
            """Remaps the pin directions 1, 2, 3 depending on the rotations around axis x,y,z"""
            # rotate the default orientation
            ori = [1, 2, 3]
            if rotation[2] % 2 == 1: ori[0], ori[1] = ori[1], ori[0]
            if rotation[1] % 2 == 1: ori[0], ori[2] = ori[2], ori[0]
            if rotation[0] % 2 == 1: ori[1], ori[2] = ori[2], ori[1]

            # swap around pin indices
            pin1 = pins == 1
            pin2 = pins == 2
            pin3 = pins == 3
            pins[pin1] = ori[0]
            pins[pin2] = ori[1]
            pins[pin3] = ori[2]

            return pins


        """Places each brick on its position on the grid."""
        # for consistency if the piece's pivot is between holes we make the same error every time
        if brick.odd:
            pivot = np.zeros(3, dtype=int)
        # compute rotated kernel center and the staring/end points of the kernel
        kernelCenter = (np.array(volume.shape) - 1) // 2
        startCrd = pos - pivot
        endCrd = startCrd + volume.shape

        # store pin locations
        pinGrid[startCrd[0]:endCrd[0], startCrd[1]:endCrd[1], startCrd[2]:endCrd[2]] = pinRemap(rotation, np.copy(pins))
        volumeGrid[startCrd[0]:endCrd[0], startCrd[1]:endCrd[1], startCrd[2]:endCrd[2]] = (volume > 0) * idx

    def adjustId(id):
        """Adjust brick idis to match equal pieces with multiple ids to the statically defined ones"""
        if id in idDict: return idDict[id]
        return id

    def incrementPins(pins):
        """Increment the pins along the appropriate axis x-1, y-2, z-3"""
        def convolvePins(pins, kernel, axis):
            """Apply a convolution kernel to propagate pins in the correct direction"""
            # reset other values
            pins[~(pins == axis)] = 0
            pinsTmp = convolve(pins, kernel, mode='constant')
            # this handles the case where two holes are seperated by one space
            pinsTmp2 = convolve(pins, np.abs(kernel), mode='constant') == 2 * axis
            # add the extra cases
            pinsTmp[pinsTmp2] = axis
            # find all elements that the kernel summed to the axis number
            return np.isin(pinsTmp, [axis, -axis])

        xCon = convolvePins(np.copy(pins), np.array([[[1]],[[0]],[[-1]]]),  1)
        yCon = convolvePins(np.copy(pins), np.array([[[1],[0],[-1]]]),      2)
        zCon = convolvePins(np.copy(pins), np.array([[[1,0,-1]]]),          3)

        pinsSummed = np.zeros(pins.shape)
        pinsSummed[zCon] = 3
        pinsSummed[yCon] = 2
        pinsSummed[xCon] = 1

        return pinsSummed


    # initialize
    idx = -1
    brickDict = bricksToDict()
    rotDict = rotsToDict()
    c = 0
    allBricks = []
    # obtain brick data
    lineCoordinates, lineRotations, lineNames = loadLDR(name)
    for i in range(len(lineCoordinates)):

        # obtain rotations from rotational matrix
        rotation = rotDict[rotToMat(lineRotations[i]).tobytes()]
        # adjust brick id
        lineNames[i] = adjustId(lineNames[i])
        # match brick to statically defined object
        if not lineNames[i] in brickDict:
            # ignore connectors
            if lineNames[i] != 2780 and lineNames[i] != 6558 and lineNames[i] != 451615462 and lineNames[i] != 45163706:
                c+=1
                print(f"Skiped {lineNames[i]}")
            continue
        brick = brickDict[lineNames[i]]
        # rotate pins and volume to match the rotation
        pins = rotateArray(brick.pins, rotation)
        volume = rotateArray(brick.volume, rotation)
        pivot = rotateArray(brick.pivot, rotation)
        # place piece on the grid
        placeBrick(brick, lineCoordinates[i], idx, pins, volume, np.argwhere(pivot)[0], rotation)
        # add brick to list
        allBricks.append((lineCoordinates[i],
                          brick,
                          rotToMat(lineRotations[i]),
                          abs(idx)))
        # decrement index
        idx -= 1
    print(f"Sum of skiped: {c}")

    connectors = incrementPins(pinGrid)
    connectors[volumeGrid < 0] = 0
    np.save(f"{filename}FramePinGrid.npy", pinGrid)
    np.save(f"{filename}FrameVolumeGrid.npy", volumeGrid)
    np.save(f"{filename}FrameBricks.npy", np.array(allBricks, dtype=object))
    return (pinGrid, volumeGrid, allBricks)



# if __name__ == '__main__':
#     pinGrid = np.zeros((150, 150, 150))
#     volumeGrid = np.zeros((150, 150, 150))
#     con = ldrToVox("car", pinGrid, volumeGrid)
#
#     appendGridLinesToGrid(volumeGrid)
#     # voxels = voxels_to_list(np.abs(volumeGrid) > 0)
#     a = (con == 3).astype(int) + (np.abs(volumeGrid) > 0).astype(int)*6
#     voxels = voxels_to_list(a)
#     drawVoxels(voxels, [150, 150, 150], 10, testVox=a)
#     # drawVoxels(voxels, [150, 150, 150], 10, testVox=volumeGrid)
#     # print("lala")