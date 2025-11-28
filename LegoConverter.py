#testVox = np.load("edgeHouse.npy").astype(bool)
#toEdges(testVox)
#sys.exit()
# binvox car1.obj -d 32 -t vtk
import time

import numpy as np
import vtk

from Coloring.BlenderRunner import orthographicProjections
from Coloring.Colorer import colorModel
from Lines.RunLines import obtainLines
from Skeletonization.Skeletonization import vtkToNumpy, getBoundaryAndSkeleton
from Skeletonization.Voxelizer import run_binvox
from ToLegos.Bricks import toLDR
from ToLegos.Fitter import coverSegments
from ToLegos.FrameGenerator import generateFrame
from Util.Drawer import voxels_to_list, drawVoxels
from Util.ldrConverter import ldrToVox

def takeTime(timers, startTime, step):
    timers[step] = (time.time() - startTime)
    print(f"{step} took {(time.time() - startTime)} seconds")
    return time.time()

def getDataOnModels():
    output = []
    names = ["Bug", "House", "Chopper", "Cat", "Car", "Plane"]
    scales = [32]
    for n in names:
        if n == "Car" or n == "Plane":
            scales = [32, 64, 80]
        for s in scales:
            run_binvox(n, str(s))
            testVox, body, boundary = getBoundaryAndSkeleton(f"./Objects/{n}.vtk", n)
            output.append(f"{n}_{s} frame:{np.sum(testVox > 0)}, body:{np.sum(body > 0)}, boundary:{np.sum(boundary > 0)}")

    print(output)


if __name__ == '__main__':
    getDataOnModels()

    suffixName = "32"
    shortName = "Bug"
    filename = f"./Objects/{shortName}.vtk"
    startTime = time.time()
    timers = {}
    # startStep = 1
    startStep = 1

    if startStep <= 1:
        run_binvox(shortName, suffixName)
        print("Step 1")
        testVox, body, boundary = getBoundaryAndSkeleton(filename, shortName)
    else:
        testVox, body, boundary = getBoundaryAndSkeleton(filename, shortName, skip=True)

    # find center
    center = np.average(np.argwhere(body > 0), axis=0).astype(int)
    startTime = takeTime(timers, startTime, "Step 1")
    if startStep <= 2:
        print("Step 2")
        points = obtainLines(shortName, filename)
        # Obtain the unused parts of the frame

    else: points = obtainLines(shortName, filename, skip=True)
    startTime = takeTime(timers, startTime, "Step 2")

    points = np.array(points)
    unusedFrame = np.zeros(testVox.shape)
    unusedFrame[points[:, 0],points[:, 1], points[:, 2]] = 1

    if startStep <= 3:
        print("Step 3")
        generateFrame(shortName)
        # ------------- Body ------------------
        # Skeleton to legos
        print(f"Converting ldr to vox")
        con, volume, bricks = ldrToVox(shortName, np.zeros(body.shape), np.zeros(body.shape), filename)
        print(f"Covering segments")
    else:
        con, volume, bricks = ldrToVox(shortName, np.zeros(body.shape), np.zeros(body.shape), filename, skip=False)
    startTime = takeTime(timers, startTime, "Step 3")
    body[volume < 0] = 0
    fullVolume = body + boundary + testVox + unusedFrame

    # voxels = voxels_to_list(np.abs(volume)+con > 0)
    # drawVoxels(voxels, [31,25,63], 10, testVox=np.abs(volume))

    if startStep <= 4:
        print("Step 4")
        # Add the unused frame parts to the volume
        voxels = voxels_to_list((boundary + (testVox - (volume < 0).astype(int))) > 0)
        drawVoxels(voxels, [31, 25, 63], 10)
        voxelGrid, connectorGraph, allBricks = coverSegments(body.astype(int), boundary + (testVox - (volume < 0).astype(int)), con, volume, bricks, fullVolume, name=shortName, center=center)
    else:
        print("Step 4")
        # Add the unused frame parts to the volume
        voxelGrid, connectorGraph, allBricks = coverSegments(None, None, None, None, None, None, skipEntireStep=True, name=shortName, center=center)


    startTime = takeTime(timers, startTime, "Step 4")
    if startStep <= 5:
        print("Step 5")
        tmp = colorModel(shortName, voxelGrid, connectorGraph)
        toLDR(allBricks, connectorGraph, tmp, center=center, name=shortName + suffixName+ "NonRemoved", filterGrid=voxelGrid)

        # Remove disconnected subgroups
        #Obtain main group
        connectorGraph.vtd = set()
        # connectorGraph.refreshMainDict()
        mainIdx = connectorGraph.findMain()
        for i in np.unique(voxelGrid):
            if i == 0 or i == 1:
                continue
            if connectorGraph.bricks[-i].idx == mainIdx:
                continue
            else:
                voxelGrid[voxelGrid == i] = 0
        tmp = colorModel(shortName, voxelGrid, connectorGraph)
        toLDR(allBricks, connectorGraph, tmp, center=center, name=shortName + suffixName + "Removed", filterGrid=voxelGrid)


    startTime = takeTime(timers, startTime, "Step 5")


    for s,t in zip(timers.keys(), timers.values()):
        print(f"{s} took {t} seconds")


#