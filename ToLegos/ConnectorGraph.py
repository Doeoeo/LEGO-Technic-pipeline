from typing import Any

import numpy as np
import numpy.typing as npt
import time
import heapq
# from ToLegos.Bricks import bricks
from ToLegos.States import fillPath
from Util.Support import color2hex

from Util.Support import find_closest_points, shortest_path_3d



class ConnectorGraph():
    def __init__(self, bricks, s) -> None:
        self.bricks:dict[int, Graph] = {}
        self.positions = {}
        self.types = {}
        self.activeParts = set()
        self.mappings = np.zeros(s.shape).astype(int)
        self.typeMappings = np.zeros(s.shape).astype(int)
        self.connectability = {}
        self.ban = set()
        self.elementColor = {}

        # Variables for loging already attempted connectivity tests
        self.vtd = set()
        self.bannedCoordinates = set()
        self.aToNNDict = {}
        self.distanceBetweenPoints = {}

        self.i = 0
        for b in bricks:
            self.bricks[b[3]] = Graph(b[3])
            self.activeParts.add(b[3])
            self.i = max(b[3], self.i)

        self.i += 1

        # Coloring variable
        self.colours = {}
        self.colourIdx = 1

    def getBrick(self, i):
        """
            Getter for brick i
        :param i: int index of the brick
        :return: Graph (brick) object
        """
        if i in self.bricks: return self.bricks[i]
        return None

            
    def add(self, b: int, c: tuple[int, int, int], i:int) -> None:
        """
            Parameters:
                b(int): brick index
                c(tuple): connector possition
                i(int): connector type
        """
        # Add possible connection to graph
        self.bricks[b].addConnections(c, i)
        
        # Connect possible connector position with the graph
        if c in self.positions:
            self.positions[c].append(b)
            self.types[c].append(i)
        else: 
            self.positions[c] = [b]
            self.types[c] = [i]

    def merge(self, a: int, b: int) -> None:
        """
            b is merged onto a so that a remains
            Parameters:
                a(int): index of a Brick 1
                b(int): index of a Brick 2
        """
        if self.bricks[a].idx == self.bricks[b].idx:
            # print(f"Error merging the same segment {self.bricks[a].idx} {self.bricks[b].idx}")
            return
        # print(f"Merging segments {a} -> {self.bricks[a].idx} and  {b} -> {self.bricks[b].idx}")
        self.activeParts.remove(self.bricks[b].idx)
        l = self.bricks[b].bricks
        self.bricks[a].update(self.bricks[b])
        for i in l:           
            self.bricks[i] = self.bricks[a]
        #self.bricks[b] = self.bricks[a]
        #self.i -= 1

    def resetColour(self):
        self.colours = {}
        self.colourIdx = 1

    def colour(self, i: int) -> int:
        """
            Parameters:
                i(int): index of a Brick
        """
        if i in self.bricks:
            if self.bricks[i].hasColor():
                return self.bricks[i].getColor()
            else: return 1
        else: return 0
        # if i in self.bricks:
        #     if not self.bricks[i].idx in self.colours:
        #         self.colours[self.bricks[i].idx] = self.colourIdx
        #         self.colourIdx += 1
        #     return self.colours[self.bricks[i].idx]
        # else: return 0
    
    def recheck(self):
        self.connectability = {}
        for i in self.activeParts:
            l = len(self.bricks[i].allCon)
            if l in self.connectability: self.connectability[l].append(i)
            else: self.connectability[l] = [i]         

    def remap(self):
        self.recheck()
        self.mappings = np.zeros(self.mappings.shape).astype(int)
        # CAN BE SLOW
        maps = {}
        j  = 0
                
        for p, l in zip(self.positions.keys(), self.positions.values()):
            # For now we remove other connectors
            
            self.mappings[p] = self.bricks[l[0]].idx
            self.typeMappings[p] = self.types[p][0]

    def addPos(self, i:tuple[int, int, int], idx:int, type:int):
        if i in self.positions:
            self.positions[i].append(idx)
            self.types[i].append(idx)
        else: 
            self.positions[i] = [idx]
            self.types[i] = [idx]
            
    def addBrick(self, bCoord:list[tuple[int, int, int]], conCoord:list[tuple[int, int, int]], con:list[int], idx:int, s:npt.NDArray[np.integer]):
        """
            Parameters:
                bCoord(list[tuple[int, int, int]]): coordinates of the brick
                conCoord(list[tuple[int, int, int]]): coordinates of connectors
                con(list[int]): direction of connectors of conCoordinates
                idx(int): index of Graph group the brick belongs to
        """
        # MIGHT BE NEEDED
        for i in bCoord:
            # Remove affiliations where the new brick is placed
            posToRemove = self.positions.pop(tuple(i), None)
            # Remove the position from individual nodes
            print(f"Trying to remove {i} got {posToRemove}")
            if posToRemove:
                for p in posToRemove:
                    self.bricks[p].connections.pop(tuple(i), None)

        for i,t in zip(conCoord, con):
            # Skip accidental connector placement on existing blocks
            if s[tuple(i)] < 0: continue
            # Add affiliations where the new brick is placed
            self.add(idx, tuple(i), t)
        # Add connectors to the connector array

        # Add the brick to the actual Graph object
        self.bricks[self.i] = self.bricks[idx]
        self.bricks[idx].addBrick(self.i)
        self.i += 1

        print(f"Added brick")

        return self.i - 1
        
    def findMain(self) -> int | None:
        """
            Find the biggest connected group
        """
        maxSize = -1
        maxIdx = -1
        for i in self.activeParts:
            # if we already tried connecting this set we skip it
            if i in self.vtd: continue

            if len(self.bricks[i].bricks) > maxSize: 
                maxSize = len(self.bricks[i].bricks)
                maxIdx = i

        # if there are no more unvisited active parts left return none
        if maxSize == -1: return None
        # Add part to visited
        self.vtd.add(maxIdx)

        # Clear the NN dict as the anchoring segment has changed
        self.aToNNDict = {}

        return maxIdx
    
    def findOrigin(self, a:int, c:np.ndarray[int], s) -> tuple[int, int]:
        """
            Find brick coordinate to which the connector belongs to
            Parameters:
                a(int): index connected group to which the connector belongs to
                c(tuple[int, int, int]): coordinate of the connector
                s(3d np array int): area
        """
        aMove = self.bricks[a].connections[tuple(c)][0] - 1
        a1 = tuple((c + np.eye(1, 3, aMove, dtype=int)).squeeze())
        a2 = tuple((c - np.eye(1, 3, aMove, dtype=int)).squeeze())
        if  -s[a1] in self.bricks[a].bricks: return 1
        elif -s[a2] in self.bricks[a].bricks: return -1
        else:
            # If we can't find a match that spot has been claimed by another brick. Redirect it to that one
            # Obtain both indices
            option1 = self.bricks[-s[a1]].idx
            option2 = self.bricks[-s[a2]].idx
            newIndex = option1 if len(self.bricks[option1].bricks) >= len(self.bricks[option2].bricks) else option2
            print("Error no matching owner for connector")
            return None
        
    def findNN(self, a:int, s):
        """
            Find the closest two connecting points between the main connected section and
            the rest
            Parameters:
                a(int): index of the biggest connected group to compare to others
                s(3d np array int): area
        """


        bestPair = (-1, -1)
        bestDist = float('inf')
        bestIdx = -1
        for i in self.activeParts:
            if i == a: continue
            # If we have already found the points restore them else find them
            if (a, i) in self.aToNNDict:
                newPair, newDist = self.aToNNDict[(a, i)]
            else:
                if (a, i) in self.distanceBetweenPoints: distHeap = self.distanceBetweenPoints[(a, i)]
                else: distHeap = None
                newPair, newDist, distHeap = find_closest_points(list(self.bricks[a].connections.keys()), list(self.bricks[i].connections.keys()), s, self.bannedCoordinates, distHeap)
                self.distanceBetweenPoints[(a, i)] = distHeap
            self.aToNNDict[(a, i)] = (newPair, newDist)
            if newPair is None: continue
            if newDist == 0: continue
            if newDist < bestDist:
                bestPair, bestDist = (newPair, newDist)
                bestIdx = i


        # No pair found
        if bestIdx == -1:
            # print("yep")
            return None
        aOrigin = self.findOrigin(a, np.array(bestPair[0]), s)
        bOrigin = self.findOrigin(bestIdx, np.array(bestPair[1]), s)
        if aOrigin is None or bOrigin is None: return None

        # Remove the best pair from the dict as they are always getting baned and will have to be recomputed
        self.aToNNDict.pop((a, bestIdx), None)

        return bestPair, bestIdx, (aOrigin, bOrigin)
    
    def findPath(self, a:tuple[int, int, int], b:tuple[int, int, int], lastMove:int, endMove:int, origin:tuple[int, int], s):
        """
            Find and construct a labled path between points a and b
            Parameters:
                a(tuple[int, int, int]): the coordinate of the first point
                b(tuple[int, int, int]): the coordinate of the second point
                lastMove(int): Axis of the last move made in this case from the brick to the connector possition
                endMove(int): The axis of the final move to be made from the connector to the brick
                origin(tuple[int, int]): The direction (1/-1) on the axis for the last and end moves
                s(3d np array): the working area
        """
        path = shortest_path_3d(s, a, b)
        if not path: return None
        a, b = (np.array(a), np.array(b))
        pathType = []
        fromDir = [(lastMove, origin[0])]
        toDir = []
        lastPos = a
        for i in path[1:]:
            i = np.array(i)
            diff = lastPos - i
            move = np.where(diff != 0)[0][0]
            fromDir.append((move, np.sign(diff[move])))
            toDir.append((move, -np.sign(diff[move])))
            if move == lastMove: pathType.append(1)
            else: 
                pathType.append(0)
                lastMove = move
            lastPos = i
        toDir.append((endMove, origin[1]))
        
        if endMove == lastMove: pathType.append(1)
        else: 
            pathType.append(0)
            lastMove = move
        

        return (path, pathType, fromDir, toDir)
                  
    def joinSegments(self, a:tuple[int, int, int], b:tuple[int, int, int], aIdx:int, bIdx:int, origin:tuple[int, int], s):
        """
            Join two segments from point a to b
            Parameters:
                a(tuple[int, int, int]): the coordinate of the first point
                b(tuple[int, int, int]): the coordinate of the second point
        """
        aMove = np.array(self.bricks[aIdx].connections[a]).item(0) - 1
        bMove = np.array(self.bricks[bIdx].connections[b]).item(0) - 1
        # print(f"Trying to join {aIdx} and {bIdx}")
        tmp = self.findPath(a, b, aMove, bMove, origin, s)
        # No path found
        if not tmp: return None
        path, pathType, pathDir, toDir = tmp
        
        # Merge the newly connected graphs
        self.merge(aIdx, bIdx)

        return path, pathType, pathDir, toDir
        
    def connectGraph(self, s, allBricks):
        def produceNN():
            a = self.findMain()
            if not a: return None, None
            nn = self.findNN(a, s)
            # If findNN returns NONE this means that there is no valid attachment point to the main segment
            while not nn:
                a = self.findMain()
                # if findMain return None there is no section left to be connected
                if not a: return None, None
                nn =  self.findNN(a, s)
            return a, nn
        # Reset visited
        self.vtd = set()
        # A set to ban coordinate pairs that don't connect
        self.bannedCoordinates = set()

        # Find the main segment and its nearest neighbor
        a, nn = produceNN()
        print(f"Fresh start with {a} for {len(self.bricks[a].connections)} vs {len(self.positions) - len(self.bricks[a].connections)}")
        if not a: return None

        # Unwrap nn data
        (bestPair, bIdx, origin) = nn

        tmp = self.joinSegments(bestPair[0], bestPair[1], a, bIdx, origin, s)
        debugCounter = 0
        while not tmp:
            debugCounter += 1
            # Mark the non-connectable pair
            # print(f"\tBaning {bestPair[0]}, {bestPair[1]} and retrying")
            if debugCounter % 1000 == 0:
                print(f"processed {debugCounter} pairs")
            self.bannedCoordinates.add(tuple(np.concatenate((np.array(bestPair[0]), np.array(bestPair[1])))))
            # Recompute nn
            nn = self.findNN(a, s)
            # Check if main segment is not connectable
            if not nn:
                # Attempt to obtain a new main segment
                print(f"\t\tMain segment {a} is no longer valid trying to find a new one")
                a, nn = produceNN()
                print(f"\t\t\t{a}")
                # Nothing else is connectable
                if not a:
                    return None

            (bestPair, bIdx, origin) = nn
            # print(f"Trying {a} for {bestPair} {tuple(np.concatenate((bestPair[0], bestPair[1]))) in self.bannedCoordinates}")
            # Retry finding a path
            tmp = self.joinSegments(bestPair[0], bestPair[1], a, bIdx, origin, s)



        (path, pathType, pathDir, toDir) = tmp

        # path must be reversed for more optimal
        fillPath(path[::-1], pathType[::-1], pathDir[::-1], toDir[::-1], allBricks)

        return 1

    def checkConnectionValidity(self, s):
        for a in self.activeParts:
            allBrickPositions = np.zeros((0,3))
            for b in self.bricks[a].bricks:
                allBrickPositions = np.concatenate((allBrickPositions, np.argwhere(s == -b)))
            for c,t in zip(self.bricks[a].connections.keys(), self.bricks[a].connections.values()):
                npC = np.array(c)
                diff = np.sum(np.abs(allBrickPositions - npC),axis=1)
                if np.any(diff == 0):
                    print(f"Error 0 diff placed on element")
                elif np.all(diff != 1):
                    print(f"Error connector is too far from element")
                else:
                    if np.all(np.abs(allBrickPositions - npC)[diff == 1][:,t[0]-1] != 1):
                        print(f"Error connector is miss labeled")
        print(f"Checking complete")
        c = (17,16,33)
        if c in self.positions:
            print(f"{c} beolngs too { self.positions[c]}")

    def tmpDebugSupport(self):
        self.distanceBetweenPoints = {}
    def refreshMainDict(self):
        self.vtd = set()


class Graph():
    def __init__(self, brick:int) -> None:
        self.idx = brick
        self.bricks = set([brick])
        self.connections = {}
        self.allCon = set()
        self.colors = {}
    
    def addBrick(self, idx:int):
        """
            Parameters:
                coordinates(list[tuple[int, int, int]]): coordinates of connectors
                con(list[int]): direction of connectors of conCoordinates
                idx(int): index of the brick
        """    
        self.bricks.add(idx)

    def addConnections(self, c: tuple[int, int, int], i:int):
        if type(c) is list: print("added list in addConnections")
        if c in self.connections: self.connections[c].append(i)
        else: self.connections[c] = [i]
        self.allCon.add(c)
        
    # Merge g into self
    def update(self, g) -> None:
        self.bricks.update(g.bricks)
        gCon = g.connections

        for i in gCon.keys():
            if i in self.connections: self.connections[i].extend(gCon[i])
            else: self.connections[i] = gCon[i]
        self.allCon.update(g.allCon)

    def addColor(self, color):
        """
            Add color or count to the dict of colors
        :param color: tuple[int, int, int] - HSV
        :return: None
        """
        if color in self.colors:
            self.colors[color] += 1
        else:
            self.colors[color] = 1

    def hasColor(self):
        return len(self.colors) != 0

    def getColor(self):
        return color2hex(list(self.colors.keys())[0])
