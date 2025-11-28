from collections import deque
import numpy as np
from Skeletonization.Offsets import direction_dict
from Util.Support import is_point_between, saveObj

class Vertex:
    
    def __init__(self, position, index) -> None:
        self.neighbors = {}
        self.pos = position
        self.p = np.array(position)
        self.index = index
        #self.neighbors = offset_dict.copy()
        

    def add(self, coordinate:tuple, i:int) -> None:
        self.neighbors[direction_dict[coordinate]] = i
        
    def get(self, s:str) -> int:
        if s in self.neighbors: return self.neighbors[s]
        else: return None
    
    def isClose(self, a, mergeDir) ->  int:
        d = abs(self.p - a.p)
        if mergeDir: return np.max(d) if d[mergeDir] == 0 else 10
        else: return np.max(d)
    
    def immitate(self, v) -> None:
        self.index = v.index
        self.pos = v.pos
        self.p = v.p
class Edge:
    
    def __init__(self, startV:Vertex = None, endV:Vertex = None, c:int = -1) -> None:
        self.vertices = set()
        self.vertexOrder:deque[Vertex] = deque()
        self.start:Vertex = startV
        self.end:Vertex = endV
        self.c = c
        
    
    def add(self, v:Vertex) -> None:
        self.vertices.add(v)
        
        # Check if we're adding the first two points
        if len(self.vertices) == 1:
            self.vertexOrder.append(v)
            self.start = v
        elif len(self.vertices) == 2:
            if self.start.pos < v.pos:
                self.vertexOrder.append(v)
                self.end = v
            else:
                self.end = self.start
                self.start = v
                self.vertexOrder.appendleft(v)
        else: 
            # Check if the new point extends the edge
            t = is_point_between(self.start.pos, self.end.pos, v.pos)
            if t < 0: 
                self.start = v
                self.vertexOrder.appendleft(v)
            elif t > 0: 
                self.end = v
                self.vertexOrder.append(v)

    def check(self) -> bool:
        return len(self.vertices) > 1
    
    def len(self) -> int:
        return len(self.vertexOrder)
    
    def consume(self, e, s1, s2):
        if self.c == 448 or e.c == 448:
            print("lalal")
        tmp = deque([e.vertexOrder.popleft() for i in range(s2)])
        dq1 = e.vertexOrder
        e.vertexOrder = tmp
        if len(tmp) > 1:
            e.start = tmp[0]
            e.end = tmp[-1]
        else: e.start = None
        newEdge = None
        #if s2 != 0:
        #    newEdge = Edge()
        #    newEdge.add(tmp[-1])
        #    newEdge.add(dq1[0])
        tmp = list(self.vertexOrder)
        
        for i in range(s2, len(dq1)):
            v1:Vertex = dq1.popleft()
            v2:Vertex = tmp[i]
            
            #v1.immitate(v2)
            v1 = v2
        
        #return newEdge
            
    def __repr__(self) -> str:
        return f"{self.c}{self.start.p} -> {self.end.p} |{len(self.vertexOrder)}|"
            



def toEdges(voxelGrid:np.ndarray):
    voxelList = [tuple(i) for  i in np.argwhere(voxelGrid > 0)]
    
    coordinates = {}
    vertices = {}
    connections = {}
    
    # Store all vertices to be able to scan over them
    i = 1
    for v in voxelList:
        vertex = Vertex(v, i)
        vertices[v] = vertex
        coordinates[i] = vertex
        i += 1


    for v in voxelList:
        i = vertices[v]        

        offsets = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 3)
        neighbors = [tuple(i) for i in np.array(v) + offsets]
        for n, o in zip(neighbors, offsets):
            if voxelGrid[n]:
                i.add(tuple(o), vertices[n].index)

            
    
    # Scan over each direction to detect lines
    # Over x
    #edges = scan(coordinates, vertices, "l", "r", 0)
    #edges = scan(coordinates, vertices, "ldb", "ruf", 2)
    edges = scan(coordinates,      vertices, "l",   "r",   0)
    edges.extend(scan(coordinates, vertices, "f",   "b",   2))
    edges.extend(scan(coordinates, vertices, "u",   "d",   1))
    edges.extend(scan(coordinates, vertices, "ub",  "df",  None))
    edges.extend(scan(coordinates, vertices, "db",  "uf",  None))
    edges.extend(scan(coordinates, vertices, "lb",  "rf",  None))
    edges.extend(scan(coordinates, vertices, "rb",  "lf",  None))
    edges.extend(scan(coordinates, vertices, "lu",  "rd",  None))
    edges.extend(scan(coordinates, vertices, "ru",  "ld",  None))
    edges.extend(scan(coordinates, vertices, "ldb", "ruf", None))
    edges.extend(scan(coordinates, vertices, "rdb", "luf", None))
    edges.extend(scan(coordinates, vertices, "lub", "rdf", None))
    edges.extend(scan(coordinates, vertices, "rub", "ldf", None))

    # renumerate vertices
    newVertices = {}
    newEdges = []
    vertices = []
    i = 1
    c = 0
    for e in edges:
        if len(e.vertexOrder) == 0 or not e.start: continue
        e1 = e.start
        e2 = e.end
        if not (e1.index in newVertices):
                vertices.append(e1.pos)    
                newVertices[e1.index] = i
                i += 1
        if not (e2.index in newVertices):
                vertices.append(e2.pos)        
                newVertices[e2.index] = i
                i += 1
        newEdges.append((newVertices[e1.index], newVertices[e2.index]))
        c+=1
    saveObj(vertices, newEdges, "edges")
        

def scan(coordinates, vertices, firstDir, secDir, mergeDir):
    visited:set = set()
    vertexList = list(vertices.values())
    edges = []
    c = 0
    
    while vertexList:
        v = vertexList.pop()
        
        # Check if already visited
        if v.index in visited: continue
        else: visited.add(v.index)
        edge = Edge(c=c)
        c += 1
        edge.add(v)
        # Check left
        vLeft = v.get(firstDir)
        #print(f"{vLeft} -> {coordinates[vLeft].index} -> {coordinates[vLeft].get('l')}")
        while vLeft:
            visited.add(vLeft)
            edge.add(coordinates[vLeft])
            vLeft = coordinates[vLeft].get(firstDir)
        
        # Check right
        vRight = v.get(secDir)
        while vRight:
            visited.add(vRight)
            edge.add(coordinates[vRight])
            vRight = coordinates[vRight].get(secDir)    
        
        if edge.check(): 
            edges.append(edge)


    edges.extend(mergeEdges(edges, mergeDir))
    return edges
    

def mergeEdges(edges:list[Edge], mergeDir):
    addedEdges = []
    nEdges = len(edges)    
    print(f"======================= MERGING {nEdges * nEdges} =====================", flush=True)
    asdf = True
    for i in range(nEdges):
        e1 = edges[i]
        if i % 10 == 0: print(f"======================= Checked {i/nEdges*100} =====================")
        
        if len(e1.vertexOrder) < 2: continue
        
        for j in range(i + 1, nEdges):
            e2 = edges[j]            

            if len(e1.vertexOrder) < 2: break
            if len(e2.vertexOrder) < 2: continue
            asdf2 = True

            # Skip distant edges
            #dist1 = e1.vertexOrder[0].isClose(e2.vertexOrder[0])
            #dist2 = e1.vertexOrder[1].isClose(e2.vertexOrder[0])
            #if dist1 > 1 and dist1 < dist2: continue
            
            kc = 0
            consumed = False
            for k in e1.vertexOrder:
                # Skip distant edges
                #dist1 = k.isClose(e2.vertexOrder[0])
                #dist2 = k.isClose(e2.vertexOrder[1])
                #if dist1 > 1 and dist1 < dist2: continue
                lc = 0
                for l in e2.vertexOrder:
                    if 40 > l.p[0] > 38 and 40 > k.p[0] > 38 and 10 > l.p[2] > 7 and 10 > k.p[2] > 7:
                        print(f"{e1} ==> {e2} || {k.p}, {l.p}")
                        if asdf:
                            print("lol")
                        if asdf2:
                            print("lol")
                        asdf = False
                        asdf2 = False
                        print("a")
                    if k.isClose(l, mergeDir) <= 1:
                        if e1.len() - kc > e2.len() - lc:
                            #newEdge = e1.consume(e2, kc, lc)
                            e1.consume(e2, kc, lc)
                        else: 
                            #newEdge = e2.consume(e1, lc, kc)
                            e2.consume(e1, lc, kc)
                        consumed = True
                        #if newEdge: addedEdges.append(newEdge)
                        break
                    lc += 1
                if consumed: break
                kc += 1
            asdf2 = True
        asdf = True
        
    return addedEdges
        

                
                        
                    
                    
        