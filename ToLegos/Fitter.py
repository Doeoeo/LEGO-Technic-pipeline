import math
import numpy as np
from collections import deque

from adodbapi.ado_consts import directions

from ToLegos.ConnectorGraph import ConnectorGraph
from Util.Drawer import voxels_to_list, drawVoxels

from Util.Drawer import drawSegment
from Util.Support import createTransformationMatrices, filterRotations, generateCoordinates, rotations, rotate_x_90, rotate_y_90, rotate_z_90
from ToLegos.Bricks import Brick, bricks, toLDR, brickConnectors
import torch
from torch import nn
from scipy.ndimage import convolve
from ToLegos.conConvolution import alternativeConv

# This should be removed onece a state is saved temp fix!!!
import ToLegos.Bricks as Bricks
import ToLegos.ConnectorGraph as cg
import sys
sys.modules['Bricks'] = Bricks
sys.modules['ConnectorGraph'] = cg

# Function to generate permutations
def generatePermutations(arr):
    # Create an identity matrix to use for adding/subtracting
    identity = np.eye(3, dtype=int)
    
    # Create the increments and decrements matrices
    increments = arr + identity
    decrements = arr - identity
    
    # Concatenate increments and decrements
    permutations = np.concatenate((increments, decrements), axis=0)
    
    return permutations

def followSegment(v, voxelGrid):
    segment = []
    proc = deque([tuple(v)])
    limit = tuple(np.array(voxelGrid.shape) - np.array([1, 1, 1]))
    while proc:
       v = proc.pop()
       if v <= limit and voxelGrid[v] < 1: continue
       
       segment.append(v)
       voxelGrid[v] = -1


       n = generatePermutations(v) 
       for i in n:
           # if np.any(np.array(voxelGrid.shape) <= np.array(i)): continue
           if voxelGrid[tuple(i)] < 1: continue
           proc.append(tuple(i))
        
    return segment



def findSegments(voxelGrid):
    segments = []
    voxelGrid = voxelGrid.astype(int)
    
    voxels = np.argwhere(voxelGrid)
    while len(voxels) > 0:
        seg = followSegment(voxels[0], voxelGrid)
        #print(f"segment  of len: {len(seg)}")

        

        
        segment = np.zeros(voxelGrid.shape)
        segment[voxelGrid == -1] = 1
        voxelGrid[voxelGrid == -1] = 0
        segments.append(np.copy(segment))

        
        voxels = np.argwhere(voxelGrid)

    return segments

import tensorflow as tf



def resetBricks():
    for b in bricks:
        b.cleared = False

def coverSegments(voxelGrid, boundary, frameConnectors, frameVolume, frameBricks, fullVolume, skipEntireStep = False, name = "test", center=None):




    t = 1
    if skipEntireStep:
        (allBricks, connectors, s, connectorGraph) = restoreStateCon(name)
        return s, connectorGraph, allBricks

    # voxelGrid[:,:,16:] = 0
    # boundary[:,:,16:] = 0
    # frameConnectors[:,:,16:] = 0
    # frameVolume[:,:,16:] = 0
    # fullVolume[:,:,:16:] = 0
    #
    # voxelGrid[:, 16:, :] = 0
    # boundary[:, 16:, :] = 0
    # frameConnectors[:, 16:, :] = 0
    # frameVolume[:, 16:, :] = 0
    # fullVolume[:, :16:, :] = 0
    #
    # voxelGrid[16:, :, :] = 0
    # boundary[16:, :, :] = 0
    # frameConnectors[16:, :, :] = 0
    # frameVolume[16:, :, :] = 0
    # fullVolume[:16:, :, :] = 0


    voxelGrid[boundary > 0] = boundary[boundary > 0]
    segments = findSegments(voxelGrid)
    #segments = [voxelGrid]
    allBricks = []
    index = np.min(frameVolume) - 1

    connector = Brick(None, np.array([[[0]]]), "Technic pin", 3647, np.array([0,0,0]))

    # Find segments that are too small and group them together
    smallSegments = np.zeros(voxelGrid.shape)
    segmentArray = np.zeros(voxelGrid.shape)
    indx = 0
    toRemove = []
    # Find small segments and add them to a joined segment
    for s in segments:
        if np.sum(s > 0) < 5:
            toRemove.append(indx)
            smallSegments[s > 0] = 1
        indx += 1
        segmentArray[s > 0] = indx
    # Remove small segments from the list
    for i in reversed(toRemove):
        segments.pop(i)
    # Add a small segments segment :)
    segments.append(smallSegments)


    voxels = voxels_to_list((voxelGrid) > 0)
    # drawVoxels(voxels, [31, 25, 63], 10, testVox=segmentArray)


    # List  of unique  rotations
    coordRotsAll = np.stack(rotations(generateCoordinates(boundary)), axis=0)
    boundaryRotsAll, rotationDict = rotations(boundary, True)
    transformsAll = np.stack(createTransformationMatrices(), axis=0)

    uniqueBoundary = filterRotations(boundaryRotsAll)
    boundaryRotsAll = np.stack(boundaryRotsAll, axis=0)


    connectors = np.zeros(voxelGrid.shape)
    ide = np.eye(4).astype(int)
    # Fill each segment
    colors = {}
    stopIndex = len(segments)
    stopCounter = 1
    for s in segments:
        if stopCounter == stopIndex:
            print("testStop")
        stopCounter += 1
        # break
        # print("lol")
        # Add frame to segment
        s[frameVolume < 0] = frameVolume[frameVolume < 0]
        # Reset bricks
        resetBricks()
        firstStep = True
        # DEBUG FLAG
        # if np.sum(s > 0) < 50:
        #     print("segment too small skipping")
        #     continue
        print(f"New Segment")



        prevSum = 0
        curSum = np.sum(s > 0)
        while prevSum != curSum:# and curSum > 3700:
            print("Fitting new piece")
            # Obtain all segment rotations and merge then into one array
            # TO DO for smaller symetrical segments there can be more symetrical segments
            rots = rotations(s)
            uniqueRots = list(set(filterRotations(rots)) | set(uniqueBoundary))
            rots = np.stack(rots, axis=0)[uniqueRots, :, :, :]
            boundaryRots = boundaryRotsAll[uniqueRots, :, :, :]
            transforms = transformsAll[uniqueRots, :, :]
            coordRots = coordRotsAll[uniqueRots, :, :]
            # break

            bestFit = None
            maxFit = 0
            brick = None
            # Obtain the boundary of the slice
            b = np.logical_and(boundaryRots, rots > 0).astype(float)
            for p in bricks:
                # Optimize and skip smaller pieces
                if maxFit >= p.optimalFit or p.cleared: continue
                print(f"Checking brick{p.name}")

                # Prepare data and send it to GPU
                segTensor = torch.tensor(rots.astype(float)).unsqueeze(1).cuda()
                bTensor = torch.tensor(b.astype(float)).unsqueeze(1).cuda()
                kernel_tensor = torch.tensor(p.volume.astype(float)).unsqueeze(0).unsqueeze(0)

                # Prepare conv3D function
                conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=p.volume.shape, stride=(1,1,1), padding=p.kernelCenter, bias=False)
                # Insert Kernel
                with torch.no_grad():
                    conv3d.weight = torch.nn.Parameter(kernel_tensor)
                # Send to GPU
                conv3d.cuda()

                # Compute convolution and return to CPU
                segFit = conv3d(segTensor).squeeze(1).detach().cpu().numpy()
                bFit = conv3d(bTensor).squeeze(1).detach().cpu().numpy()


                # Check where te piece fits
                fits = np.argwhere((segFit) == p.maxFit)
                # No fit
                if len(fits) == 0:
                    p.cleared = True
                    continue

                # One fit
                if len(fits) == 1:
                    # if bFit[tuple(fits[0])] < p.maxFit / 2: p.cleared = True
                    # if bFit[tuple(fits[0])] < p.maxFit: p.cleared = True
                    if bFit[tuple(fits[0])] <= 0:
                        p.cleared = True
                        continue
                    if bFit[tuple(fits[0])] < maxFit: continue
                    bestFit = fits[0]
                    maxFit = bFit[tuple(bestFit)]
                    brick = p
                    continue

                # Multiple fits
                # Find the best boundary coverage
                bFits = bFit[tuple(fits.T)]
                # if bFits[np.argmax(bFits)] < p.maxFit / 8: p.cleared = True
                # if bFits[np.argmax(bFits)] < p.maxFit: p.cleared = True
                if bFits[np.argmax(bFits)] < maxFit: continue
                if bFits[np.argmax(bFits)] <= 0: continue
                maxFit = bFits[np.argmax(bFits)]
                bestFit = fits[np.argmax(bFits)]
                brick = p
            if not brick:break

            # Obtain the start and  end coordinates of the kernel block. In rotated space.
            coords = (tuple(bestFit[1:]) - brick.kernelCenter).astype(int)
            endCoords = (coords + brick.shape).astype(int)

            # Obtain the coordinates of the kernel from the rotated to actual space
            tmp = np.array(coordRots[bestFit[0], coords[0]:endCoords[0], coords[1]:endCoords[1], coords[2]:endCoords[2]].flatten().tolist())
            # Remove the kernel from the segment
            print(f"\tInserting into s:{np.min(s)} and grid {np.min(voxelGrid)}")
            s[tuple(tmp.T[:,brick.vFlat > 0])] = (brick.vFlat[brick.vFlat > 0]) * index
            voxelGrid[tuple(tmp.T[:,brick.vFlat > 0])] = (brick.vFlat[brick.vFlat > 0]) * index
            print(f"\t\tInserted into s:{np.min(s)} and grid {np.min(voxelGrid)}")
            if np.min(voxelGrid) == 0:

                if firstStep == False:
                    print("DISASTER")
                    return None
            # Interchange types of connectors due to possible rotation
            tmpConnectors = np.copy(brick.pins)
            temp1 = tmpConnectors == 1
            temp2 = tmpConnectors == 2
            temp3 = tmpConnectors == 3
            tmpConnectors[temp1] = rotationDict[uniqueBoundary[bestFit[0]]][0]
            tmpConnectors[temp2] = rotationDict[uniqueBoundary[bestFit[0]]][1]
            tmpConnectors[temp3] = rotationDict[uniqueBoundary[bestFit[0]]][2]
            tmpConnectors = tmpConnectors.flatten()
            # Input connectors
            connectors[tuple(tmp.T[:,tmpConnectors > 0])] = tmpConnectors[tmpConnectors > 0]

            # For visualization
            colors[index] = np.random.rand(3).tolist()

            # Add piece
            allBricks.append((
                coordRots[tuple(bestFit + np.array(((0,) + tuple(brick.center - brick.kernelCenter))))],
                brick,
                transforms[bestFit[0], :, :].T,
                abs(index)))
            #bestFit[0]))

            # Clean up
            index -= 1
            prevSum = curSum
            curSum = np.sum(s > 0)
            print(f"Fitted {brick.name} old:{prevSum}, new:{curSum} for {brick.maxFit}")
            if prevSum - curSum != brick.maxFit:
                print(f"MISSMATCH ABOVE {prevSum - curSum} instead of {brick.maxFit}")


    # voxelGrid[boundary == 1] = 1
    storeState(allBricks, connectors, voxelGrid, name)
    # return 0
    (allBricks, connectors, s) = restoreState(name)
    if np.min(s) == 0:
        print("DISASTER")
        return None

    # Ugly fix to code change shoudl probably be fixed at some point
    rots = rotations(s)
    uniqueRots = list(set(filterRotations(rots)) | set(uniqueBoundary))
    coordRots = coordRotsAll[uniqueRots, :, :]

    connectors[frameConnectors > 0] = frameConnectors[frameConnectors > 0]
    s[frameVolume < 0] = frameVolume[frameVolume < 0]
    # s[(s == 0) & (fullVolume > 0)] = 1

    # Add frame to bricks
    print(f"extending with {len(frameBricks)}")
    allBricks.extend(frameBricks)

    # Create graph of disjointed pieces
    connectorGraph = ConnectorGraph(allBricks, s)

    connectorGraph.checkConnectionValidity(s)

    conPos = [connectors == 1, connectors == 2, connectors == 3]
    # xCon = connect(connectors, np.array([[[1]],[[0]],[[-1]]]),   1, allBricks, conPos, (1, 2), s,  connectorGraph)
    # yCon = connect(connectors, np.array([[[1],[0],[-1]]]),       2, allBricks, conPos, (0, 2), s,  connectorGraph)
    # zCon = connect(connectors, np.array([[[1,0,-1]]]),           3, allBricks, conPos, (0, 1), s,  connectorGraph)

    xCon = connect2(conPos[0], s, allBricks, connectorGraph, 0, (1,2))
    yCon = connect2(conPos[1], s, allBricks, connectorGraph, 1, (0,2))
    zCon = connect2(conPos[2], s, allBricks, connectorGraph, 2, (0,1))

    # Remap the segment mapping array
    connectorGraph.remap()
    toLDR(allBricks, connectorGraph, name="CutNonConect", center=center)
    # return 0

    con = np.zeros(s.shape)
    con[zCon == 1] = 3
    con[yCon == 1] = 2
    con[xCon == 1] = 1
    # return s, connectorGraph, allBricks

    flag = True
    # flag = False
    while flag and fitConnectors(s, con, uniqueRots, rotationDict, connectorGraph, coordRots, allBricks, transformsAll[uniqueRots,:, :], colors):
        connectorGraph.remap()
    if flag: storeStateCon(allBricks, connectors, s, connectorGraph, name)
    else: (allBricks, connectors, s, connectorGraph) = restoreStateCon(name)
    connectorGraph.checkConnectionValidity(s)

    toLDR(allBricks, connectorGraph, name="CutHalfConect", center=center)

    # Join remainig segments manually
    connectorGraph.tmpDebugSupport()
    while connectorGraph.connectGraph(s, allBricks):
        connectorGraph.remap()
        print(f"Segments remaining: {len(connectorGraph.activeParts)}")

    toLDR(allBricks, connectorGraph, name="CutConect", center=center)

    #85861
    #15573
    # for i in np.argwhere(s > 0):
    #     allBricks.append((
    #         i,
    #         connector,
    #         np.eye(4),
    #         -1))
    return s, connectorGraph, allBricks
    toLDR(allBricks, connectorGraph)

    s[s == 1] = 0
    # drawSegment(s, [31,25,63], 10, col=colors)
    print("new")
        

def fitConnectors(s, con, uniqueRot, rotationDict, connectorGraph: ConnectorGraph, coordRots, allBricks, transforms, colors):
    print(f"Searching for new connector")
    # Prepare data in all rotations
    connectors = brickConnectors
    canFit =  np.stack(rotations(s), axis=0)[uniqueRot, :, :, :]
    #qualFit = np.stack(rotations(connectorGraph.mappings > 0), axis=0)[uniqueRot, :, :, :]
    connectorGraph.mappings[s < 0] = 0
    segMaps = np.stack(rotations(connectorGraph.mappings), axis=0)[uniqueRot, :, :, :]
    
    bestFit = np.array([ 7, 17, 22, 10])
    coords = np.array([16, 22,  8])
    endCoords = np.array([19, 23, 13])
    tmp2 = np.array(coordRots[bestFit[0], coords[0]:endCoords[0], coords[1]:endCoords[1], coords[2]:endCoords[2]].flatten().tolist())
    brick = connectors[0]
    # Remap each value to the right rotation and take the unique rotations
    rots = rotations(con.astype(int))
    conRots = np.zeros((len(uniqueRot), 3) + s.shape).astype(int)
    for i,j in zip(uniqueRot, range(len(uniqueRot))):
        tmp = np.argsort(np.array((0,) + rotationDict[i]))[rots[i]]
        conRots[j, 0] = tmp == 1
        conRots[j, 1] = tmp == 2
        conRots[j, 2] = tmp == 3
    


    maxFit = -1
    fitRatio = 0
    brick = None
    bestFit = None


    # Try to fit each block
    for b in connectors:
        print(f"\tTrying new connector {b.name}")
        # Prepare data and send it to GPU
        k = np.array([
            (b.pins == 1).astype(float),
            (b.pins == 2).astype(float),
            (b.pins == 3).astype(float)])
        
        fitTensor =  torch.tensor(canFit.astype(float)).unsqueeze(1).cuda()
        #qualTensor = torch.tensor(qualFit.astype(float)).unsqueeze(1).cuda()
        conTensor =  torch.tensor(conRots.astype(float)).cuda()
        segTensor = torch.tensor(segMaps.astype(float)).unsqueeze(1).cuda()
        kernel_tensor_con = torch.tensor(np.expand_dims(k, 1))
        kernel_tensor = torch.tensor(b.volume.astype(float)).unsqueeze(0).unsqueeze(0)
        
        # Prepare conv3D function
        conv3d =    nn.Conv3d(in_channels=1, out_channels=1, kernel_size=b.volume.shape, stride=(1,1,1), padding=b.kernelCenter, bias=False)
        conv3dCon = nn.Conv3d(in_channels=3, out_channels=3, kernel_size=b.volume.shape, stride=(1,1,1), padding=b.kernelCenter, bias=False, groups=3)
        # Insert Kernel
        with torch.no_grad():
            conv3dCon.weight = torch.nn.Parameter(kernel_tensor_con)
            conv3d.weight = torch.nn.Parameter(kernel_tensor)
        # Send to GPU
        conv3d.cuda()
        conv3dCon.cuda()
        kernel_tensor = kernel_tensor.cuda()    
        kernel_tensor_con = kernel_tensor_con.cuda()    
        
        # Compute convolution and return to CPU
        convFit = conv3d(fitTensor).squeeze(1).detach().cpu().numpy()
        #convQual = conv3d(qualTensor).squeeze(1).detach().cpu().numpy()
        convCon = conv3dCon(conTensor).detach().cpu().numpy()
        convSeg = alternativeConv(segTensor, kernel_tensor, conTensor, kernel_tensor_con, COut = 1, kernelSize = b.volume.shape, pad = b.kernelCenter).squeeze(1).detach().cpu().numpy()

        # Sum up fit quality
        conSum = np.sum(convCon, axis = 1)
        
        # Obtail valid positions that fit
        fits = np.argwhere(convFit == b.maxFit)
        fitQual = convSeg[tuple(fits.T)]
        # Sort the fit quality reversed
        fitBest = np.flip(np.argsort(fitQual))
        for i in fitBest:
            posPosition = fits[i]
            # For now ignoroe
            ## Check if we leave a disconected graph
            if 1 in connectorGraph.connectability:
                pass
                ## Obtain the start and  end coordinates of the kernel block. In rotated space.
                #coords = (tuple(posPosition[1:]) - b.kernelCenter).astype(int)
                #endCoords = (coords + b.shape).astype(int)
                #
                ## Obtain the coordinates of the kernel from the rotated to actual space
                #tmp = np.array(coordRots[posPosition[0], coords[0]:endCoords[0], coords[1]:endCoords[1], coords[2]:endCoords[2]].flatten().tolist())
                ## Remove the kernel from the segment
                #s[tuple(tmp.T[:,brick.vFlat > 0])] = (brick.vFlat[brick.vFlat > 0])* index

            # If we found a viable piece all the remaining ones are equal or worse
            if maxFit < fitQual[i]:
                maxFit = fitQual[i]
                bestFit =  fits[i]
                brick = b
                break
            elif maxFit == fitQual[i]:
                ratio = fitQual[i] / brick.optimalFit
                if ratio > fitRatio:
                    maxFit = fitQual[i]
                    bestFit =  fits[i]
                    brick = b 
            else: break
        


    if bestFit is None:
        return False
    # Obtain the start and  end coordinates of the kernel block. In rotated space.
    coords = (tuple(bestFit[1:]) - brick.kernelCenter).astype(int)
    endCoords = (coords + brick.shape).astype(int)

    # Obtain the coordinates of the kernel from the rotated to actual space
    tmp = np.array(coordRots[bestFit[0], coords[0]:endCoords[0], coords[1]:endCoords[1], coords[2]:endCoords[2]].flatten().tolist())
    
    # Join newly connected segments
    # brick pin remaper
    pinRemap = np.array((0,) + rotationDict[uniqueRot[bestFit[0]]])[brick.pins.flatten()[brick.vFlat > 0]]
    bCoords = tuple(tmp.T[:,brick.vFlat > 0])
    # Possible brick segment that our piece connected
    seg = connectorGraph.mappings[bCoords]
    #Take only connected segments [brick pins remapped in correct rotation that match connectors on the same position]
    newCon = seg[pinRemap == con[bCoords]]
    newCon = newCon[newCon != 0]
    # Merge each segment to the first
    counter = 0
    for i in newCon[1:]:
        if i == 0:
            print("yahoo")
            continue
        counter += 1
        connectorGraph.merge(newCon[0], i)
        connectorGraph.checkConnectionValidity(s)

    if counter < 1:
        print("Couldn't find connector")
        return False
    # Add the new brick to the ConnectorGraph
    # Some temp variables to avoid a mess
    c = np.empty(4, dtype=object)
    c[:] = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
    conShift = np.array(list(c[pinRemap]))
    a = np.array(bCoords).T
    conCoords = np.concatenate((a + conShift, a - conShift), axis=0)
    # Remove the kernel from the segment
    # IMPORTANT! At the moment pins used to connect sub sections together aren't removed
    # because len 3 pins could be used to connect them
    s[bCoords] = (brick.vFlat[brick.vFlat > 0]) * -connectorGraph.i
    con[tuple(conCoords.T)] = np.concatenate((pinRemap, pinRemap), axis=0)
    con[s < 0] = 0
    colors[-150] = np.random.rand(3).tolist()

    idx = connectorGraph.addBrick(list(a),
                            conCoords,
                            np.concatenate((pinRemap, pinRemap), axis=0),
                            newCon[0],
                            s)
    connectorGraph.checkConnectionValidity(s)


    # Add piece
    allBricks.append((
        coordRots[tuple(bestFit + np.array(((0,) + tuple(brick.center - brick.kernelCenter))))],
        brick,
        transforms[bestFit[0], :, :].T,
        idx
    ))
  
    print(f"new connector {maxFit} - {brick.name} | Segments remaining: {len(connectorGraph.activeParts)}")
    return True
    
            
def connectorsToBricks(c, connectors, s, allBricks, axis, connectorGraph: ConnectorGraph):
    cOrigin = np.copy(c)
    c = tuple(c)
    # Move to origin brick
    cOrigin[axis-1] += connectors[c] / axis
    # Compute the brick idx
    bIdx = abs(s[tuple(cOrigin)])
    if bIdx == 0:
        s[c] = 1
        voxels = voxels_to_list(np.abs(s) > 0)
        z = np.zeros(s.shape)
        z[c] = 10

        drawVoxels(voxels, [31, 25, 63], 10, testVox=np.abs(s))
        print("No bricks found")
    # Skip storing possible connection if taken
    if s[c] < 1 : return (bIdx, abs(s[c]))
    
    connector = Brick(None, np.array([[[0]]]), "Technic pin", 85861, np.array([0,0,0]))
    rotations = [rotate_x_90,  rotate_z_90, rotate_y_90]
    
    # Add connection to brick
    connectorGraph.add(bIdx, c, abs(connectors[c]))

    return (bIdx, abs(s[c]))
                  
def connect(connectors, kernel, axis, allBricks, conPos, exclude, s, connectorGraph):
    # IMPORTANT NOTE! PROBABLY buggs out when two bricks are next to one another with one space in between
    rotations = [rotate_x_90,  rotate_z_90, rotate_y_90]
    connector = Brick(None, np.array([[[0]]]), "Technic pin", 3673, np.array([0,0,0]))
    # Remove unwanted connectors
    for i in exclude: connectors[conPos[i]] = 0
    # Compute convolution
    connectorsTmp = convolve(connectors, kernel, mode='constant')
    # This handles the case where two holes are seperated by one space
    connectorsTmp2 = convolve(connectors, np.abs(kernel), mode='constant') == 2 * axis
    connectorsTmp[connectorsTmp2] = axis
    connectorsReturn = np.isin(connectorsTmp, [axis, -axis])

    # Reset connectors
    k=0
    placed = set()
    for b in list(np.argwhere(connectorsReturn)):
        c = connectorsToBricks(b, connectorsTmp, s, allBricks, axis, connectorGraph)
        # Check if connection direction matches
        if connectors[tuple(b)] != axis: continue
        tmpFloat = b.astype(float)
        # Shift the possition between two blocks
        tmpFloat[axis - 1] += 0.5 * math.copysign(1, connectorsTmp[(tuple(b))])
        tmpFloat = tuple(tmpFloat)
        # Merge the two bricks
        connectorGraph.merge(c[1], c[0])
        if tmpFloat in placed: continue
        placed.add(tmpFloat)
        allBricks.append((
           tmpFloat,
           connector,
           rotations[axis-1],
           abs(0)
        ))
        k+=1 
    for i in exclude: connectors[conPos[i]] = i + 1

    print(f"placed {k} connectors on axis:{axis}")



    connectorsTmp[connectorsReturn == False] = 0
    return connectorsReturn

def connect2(connectors, s, allBricks, connectorGraph, axis, exclude):
    freeCons = np.zeros(connectors.shape)
    # Init connectors
    con = Brick(None, np.array([[[0]]]), "Technic pin", 3673, np.array([0,0,0]))
    longCon = Brick(None, np.array([[[0]],[[0]],[[0]]]), "Technic pin long", 6558, np.array([1,0,0]))
    rotations = [rotate_x_90,  rotate_z_90, rotate_y_90]
    direction = 1
    # Loop over all dimensions
    for y in range(s.shape[exclude[0]]):
        for z in range(s.shape[exclude[1]]):
            direction *= -1

            prev = False
            prevPos = (0, 0, 0)
            prevPrev = False
            prevPrevPos = (0, 0, 0)
            # Set current coordinate
            poss = [0, 0, 0]
            poss[exclude[0]] = y
            poss[exclude[1]] = z
            for x in range(s.shape[axis]):
                poss[axis] = x
                pos = tuple(poss)
                # if (pos == (29, 19, 27) and prevPos == (29, 19, 26)) or (prevPos == (29, 19, 27) and pos == (29, 19, 26)):
                #     direction *= -1
                #     toLDR(allBricks, connectorGraph)
                #     print("Found pairing")
                cur = connectors[pos]
                if prevPrev:
                    # can we place a connector through the middle position
                    bridge = prev or s[prevPos] == 0
                    if cur and bridge:
                        # We only connect c and pp if there is a bridge between them
                        connectorGraph.merge(abs(s[prevPrevPos]), abs(s[pos]))
                        # Add long con
                        allBricks.append((
                            prevPos,
                            longCon,
                            rotations[axis],
                            abs(0)
                        ))
                        # Remove connected values
                        cur = False
                        # Merge middle
                        if prev:
                            connectorGraph.merge(abs(s[prevPrevPos]), abs(s[prevPos]))
                            # Remove connected values
                            prev = False
                    elif prev:
                        # We always have to connect pp and p no matter what
                        connectorGraph.merge(abs(s[prevPrevPos]), abs(s[prevPos]))
                        # Remove connected values
                        prev = False
                        # Add con
                        incPos = np.array((0., 0., 0.))
                        incPos[axis] += 0.5 * direction
                        allBricks.append((
                            np.array(prevPrevPos if direction == 1 else prevPos).astype(float) + incPos,
                            con,
                            rotations[axis],
                            abs(0)
                        ))

                # fill connector array
                if prev:
                    if s[prevPrevPos] > -1:
                        freeCons[prevPrevPos] = 1
                        connectorGraph.add(abs(s[prevPos]), prevPrevPos, axis + 1)
                    if s[pos] > -1:
                        freeCons[pos] = 1
                        connectorGraph.add(abs(s[prevPos]), pos, axis + 1)

                # Increment values
                prevPrevPos = prevPos
                prevPrev = prev
                prevPos = pos
                prev = cur
                # toLDR(allBricks, connectorGraph)
    return freeCons






def storeStateCon(allBricks, connectors, s, conG, name):
    np.save(f"SavedStates/{name}BricksCon.npy", np.array(allBricks, dtype=object))
    np.save(f"SavedStates/{name}ConnectorsCon.npy", connectors)
    np.save(f"SavedStates/{name}SegmentCon.npy", s)
    np.save(f"SavedStates/{name}ConG", np.array([conG], dtype=object))

def restoreStateCon(name):
    allBricks = list(np.load(f"SavedStates/{name}BricksCon.npy", allow_pickle=True))
    connectors = np.load(f"SavedStates/{name}ConnectorsCon.npy").astype(int)
    s = np.load(f"SavedStates/{name}SegmentCon.npy").astype(int)
    conG = np.load(f"SavedStates/{name}ConG.npy", allow_pickle=True)
    
    return (allBricks, connectors, s, conG[0])

def storeState(allBricks, connectors, s, name):
    np.save(f"SavedStates/{name}Bricks.npy", np.array(allBricks, dtype=object))
    np.save(f"SavedStates/{name}Connectors.npy", connectors)
    np.save(f"SavedStates/{name}Segment.npy", s)
    
def restoreState(name):
    allBricks = list(np.load(f"SavedStates/{name}Bricks.npy", allow_pickle=True))
    connectors = np.load(f"SavedStates/{name}Connectors.npy").astype(int)
    s = np.load(f"SavedStates/{name}Segment.npy").astype(int)
    
    return (allBricks, connectors, s)

        