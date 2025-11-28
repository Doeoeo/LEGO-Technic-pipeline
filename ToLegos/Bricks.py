from typing import Tuple
import numpy as np
from pandas import pivot

from ToLegos.ConnectorGraph import ConnectorGraph
from Util.Support import color2hex, weighted_avg_tuple

flip1 = np.array([
    [1,  0,  0, 0],    
    [0, -1,  0, 0],    
    [0,  0, -1, 0],    
    [0,  0,  0, 1],    
])

rotationsAroundAxis = {
    1: np.diag([1,-1,-1]),
    2: np.diag([-1,1,-1]),
    0: np.diag([-1,-1,1])
}


class Brick:
    def __init__(self, pins, volume, name, idx, center, holes=False, rotation = None, odd = False):
        self.pins = pins
        self.volume = volume
        self.vFlat = volume.flatten()
        self.maxFit = np.sum(volume)
        self.center = center
        self.kernelCenter = (np.array(volume.shape) - 1) // 2
        self.shape = volume.shape
        self.name = name
        self.idx = idx
        self.optimalFit = np.sum(volume)
        self.tVolume = np.transpose(volume, (1, 0, 2))
        self.cleared = False

        # for pivot rotations
        self.pivot = np.zeros(self.shape)
        self.pivot[tuple(center)] = 1
        self.odd = odd
        if not (rotation is None): self.rotation = rotation
        else: 
            self.rotation = np.array([
                [1, 0, 0, 0],    
                [0, 1, 0, 0],    
                [0, 0, 1, 0],    
                [0, 0, 0, 1],    
            ])
 
        
def toLDR(bricks, connectorGraph: ConnectorGraph, tmp = None, name="test", center = None, filterGrid = None):
    """
        Converts a list of brick defining tuples to .ldr format file
        Parameters:
            bricks(List[
                tuple[tuple[int, int int]: coordinates
                int: brick id
                np.array[4,4]: rotation matrix
                int: colour number
            ]])
    """
    print(f"Bricks: {len(bricks)} saving")
    connectorGraph.resetColour()
    with (open(f"LDRModels/{name}.ldr", 'w') as f):
        f.write(f"0 Name: {name}\n")
        for b in bricks:
            # Filter removed elements
            print(f"{b[0]}")

            # print(f"{filterGrid[tuple(np.array(b[0], dtype=int))]}" )
            if (not filterGrid is None) and filterGrid[tuple(np.array(b[0], dtype=int))] == 0: continue


            i = b[3]
            # if b[1].idx != 64782 and b[1].idx != 3673: continue

            # Coloring
            if tmp:
                if i in tmp:
                    if len(tmp[i]) != 0:
                        # col = color2hex(max(tmp[i], key=tmp[i].get))
                        col = color2hex(weighted_avg_tuple(tmp[i]))
                    else:
                        col = 1
                else:
                    col = 0
            else: col = connectorGraph.colour(b[3])
            rot = b[2][0:3, 0:3]
            offset = np.array([0, 0, 0])

            # Handle shifting and rotation
            if b[1].idx == 43857:
                offset = b[2][:, 2] * 10
            elif b[1].idx == 320637:
                offset = b[2][:, 2] * 10
                b[1].idx = "32063_double"
            elif b[1].idx == 32449:
                b[1].idx = "32449_double"
            elif b[1].idx == 320567: b[1].idx = "32056_double"
            elif b[1].idx == 324497:
                offset = b[2][:, 2] * 10
                b[1].idx = "32449_double"
            elif b[1].idx == "32449_double":
                offset = b[2][:, 2] * 10
            elif b[1].idx == "32063_double":
                offset = b[2][:, 2] * 10
            elif (b[1].idx == 64782 or b[1].idx == 15458):
                 facingAxis = np.argwhere(b[2][:, 1])[0][0]
                 # print(f"{center[facingAxis]}  {b[0][facingAxis]} -> {facingAxis}")
                 if center[facingAxis] > b[0][facingAxis]:
                     # print(f"{ rotationsAroundAxis[facingAxis]}  {rot}")
                     rot =  rotationsAroundAxis[facingAxis] @ rot
            rot = rot.flatten()
            output = " ".join(map(str, rot))
            f.write(f"1 {col} {b[0][0] * 20 + offset[0]} {b[0][1] * 20 + offset[1]} {b[0][2] * 20 + offset[2]} {output} {b[1].idx}.dat\n")
            #f.write(f"1 {b[3]} {b[0][0] * 20} {b[0][1] * 20} {b[0][2] * 20} {output} {b[1].idx}.dat\n")
            i += 1
            # {b[1].idx}.dat\n")
        f.write("\n")
    
    
    
bricks = [
    # Flat
    # 5x11x1
    Brick(np.array([
            [[1,  1,  1,  1, 1]],
            [[2, 10, 10, 10, 2]],
            [[3, 10, 10, 10, 3]],
            [[3, 10, 10, 10, 3]],
            [[3, 10, 10, 10, 3]],
            [[3, 10, 10, 10, 3]],
            [[3, 10, 10, 10, 3]],
            [[3, 10, 10, 10, 3]],
            [[3, 10, 10, 10, 3]],
            [[2, 10, 10, 10, 2]],
            [[1,  1,  1,  1, 1]],            
    ]),
        np.array([
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],            
    ]), "Flat 5x11x1", 64782, np.array([5, 0, 2]), rotation=flip1),
    # Trapezium 5x11x1
    Brick(np.array([
            [[0, 0, 1, 1, 1]],
            [[0, 0, 2, 0, 2]],
            [[0, 0, 0, 0, 3]],
            [[0, 0, 0, 0, 3]],
            [[0, 0, 0, 0, 3]],
            [[0, 0, 0, 0, 3]],
            [[0, 0, 0, 0, 3]],
            [[0, 0, 0, 0, 3]],
            [[0, 0, 0, 0, 3]],
            [[2, 0, 2, 0, 2]],
            [[1, 1, 1, 1, 1]],            
    ]),
        np.array([
            [[0, 0, 1, 1, 1]],            
            [[0, 1, 1, 1, 1]],
            [[0, 1, 1, 1, 1]],
            [[0, 1, 1, 1, 1]],
            [[0, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 1, 1, 1, 1]],
    ]), "Trapezium 5x11x1", 18945, np.array([5, 0, 2])),
    # 11x1x3
    Brick(np.array([
            [[1,  1, 1]],
            [[2, 10, 2]],
            [[3, 10, 3]],
            [[3, 10, 3]],
            [[3, 10, 3]],
            [[3, 10, 3]],
            [[3, 10, 3]],
            [[3, 10, 3]],
            [[3, 10, 3]],
            [[2, 10, 2]],
            [[1,  1, 1]],         
    ]),
        np.array([
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],         
    ]), "Flat 3x11x1", 15458, np.array([5, 0, 1]), rotation=flip1),
    # 3x7x1
    Brick(np.array([
            [[1,  1, 1]],
            [[2, 10, 2]],
            [[3, 10, 3]],
            [[3, 10, 3]],
            [[3, 10, 3]],
            [[2, 10, 2]],
            [[1,  1, 1]],
    ]),
        np.array([
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
            [[1, 1, 1]],
    ]), "Panel 3x7x1", 71709, np.array([3, 0, 1]), rotation=flip1),              
    # Bent 3x5x4
    Brick(np.array([
            [
                [10, 10, 10, 10, 0],
                [10, 10,  0,  0, 0],
                [10,  0,  0,  0, 0],
                [0,   0,  0,  0, 0],
            ],
            [
                [10, 10, 10, 10, 1],
                [10,  0,  0,  0, 0],
                [10,  0,  0,  0, 0],
                [ 1,  0,  0,  0, 0],    
            ],
            [
                [10, 10, 10, 10, 10],
                [10, 10,  0,  0,  0],
                [10,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0],
            ],
        ]),
        np.array([
            [
                [1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],    
            ],
            [
                [1, 1, 1, 1, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ]), "Bent Panel 3x5x4", 24116, np.array([1, 0, 4])),

    # Fairing smooth 5x7x2
    Brick(
        np.array([
            [
                [10, 10, 10, 10, 10,  0,  0],
                [10, 10, 10, 10, 10, 10,  0],
                [10, 10, 10, 10, 10, 10, 10],
                [10, 10, 10, 10, 10, 10, 10],
                [ 3, 10,  2,  2,  2, 10, 10],
            ],
            [
                [3,  1, 10, 10, 0, 0, 0],
                [3,  0,  0,  0, 0, 0, 0],
                [3,  0,  0,  0, 0, 0, 0],
                [3,  0,  0,  0, 0, 0, 0],
                [3,  0,  0,  0, 0, 0, 0],      
            ],
        ]),
        np.array([
            [
                [1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
            ],
            [
                [1, 1, 1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0],      
            ],
        ]), "Fairing smooth 5x7x2", 64680, np.array([0, 4, 0])),
    # Bent Tapered 4x5x4 RIGHT
    Brick(
        np.array([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],            
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],      
        ]),
        np.array([
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],            
            ],
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],      
        ]), "Bent Tapered Right 4x5x4", 80272, np.array([2, 3, 0])),
    # Bent Tapered 4x5x4 LEFT
    Brick(
        np.array([
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 0, 0, 0],            
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],      
        ]),
        np.array([
            [
                [1, 1, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],            
            ],
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [1, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 0],
            ],      
        ]), "Bent Tapered Left 4x5x4", 80271, np.array([2, 3, 0])),    
    # Beam 2   MIDDLE is center LOL 
    Brick(
        np.array([
            [[2, 2,]],        
        ]),
        np.array([
            [[1, 1,]],        
        ]), "Beam 2", 43857, np.array([0, 0, 0]), odd=True),
    # Beam 3    
    Brick(
        np.array([
            [[2, 2, 2,]],        
        ]),
        np.array([
            [[1, 1, 1]],        
        ]), "Beam 3", 17141, np.array([0, 0, 1])),  
    # Beam 5    
    Brick(
        np.array([
            [[2, 2, 2, 2, 2,]],        
        ]),
        np.array([
            [[1, 1, 1, 1, 1]],        
        ]), "Beam 5", 32316, np.array([0, 0, 2])),    
    # Beam 7
    Brick(
        np.array([
            [[2, 2, 2, 2, 2, 2, 2,]],
        ]),
        np.array([
            [[1, 1, 1, 1, 1, 1, 1]],        
        ]), "Beam 7", 16615, np.array([0, 0, 3])),
    # Beam 9 
    Brick(
        np.array([
            [[2, 2, 2, 2, 2, 2, 2, 2, 2,]],        
        ]),
        np.array([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1]],        
        ]), "Beam 9", 64289, np.array([0, 0, 4])),
    # Beam 11
    Brick(
        np.array([
            [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
        ]),
        np.array([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        ]), "Beam 11", 32525, np.array([0, 0, 5])),
    # Beam 13    
    Brick(
        np.array([
            [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
        ]),
        np.array([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],        
        ]), "Beam 13", 41239, np.array([0, 0, 6])),
    # Beam 13
    Brick(
        np.array([
            [[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
        ]),
        np.array([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
        ]), "Beam 15", 32278, np.array([0, 0, 7])),
    # Beam 3x3-T    
    Brick(
        np.array([
            [[0, 0, 2]],
            [[2, 2, 2]],
            [[0, 0, 2]],        
        ]),
        np.array([
            [[0, 0, 1]],
            [[1, 1, 1]],
            [[0, 0, 1]],        
        ]), "Beam 3x3-T", 60484, np.array([1, 0, 0])),    
    # Beam 3x5 bent 90    
    Brick(
        np.array([
            [[2, 2, 2, 2, 2]],
            [[0, 0, 0, 0, 2]],
            [[0, 0, 0, 0, 2]],
        ]),
        np.array([
            [[1, 1, 1, 1, 1]],
            [[0, 0, 0, 0, 1]],
            [[0, 0, 0, 0, 1]],
        ]), "Beam 3x5-T", 32526, np.array([0, 0, 0])),
    Brick(
        np.array([
            [[2, 2, 2, 2]],
            [[0, 0, 0, 2]],
        ]),
        np.array([
            [[1, 1, 1, 1]],
            [[0, 0, 0, 1]],
        ]), "Beam 2x4 bent 90", 32140, np.array([0, 0, 0])),
    #
    # Extra combined elements from the article
    # 32063_double == 320637 || BAD CENTER ||
    Brick(
        np.array([
            [[2, 2, 2, 2, 2, 2]],
        ]),
        np.array([
            [[1, 1, 1, 1, 1, 1]]
        ]),"32063_double", 320637, np.array([0, 0, 2])),
    # 32449_double == 324497 || BAD CENTER ||
    Brick(
        np.array([
            [[2, 2, 2, 2]]
        ]),
        np.array([
            [[1, 1, 1, 1]]
        ]), "32449_double", 324497, np.array([0, 0, 1])),
    # 32056_double == 320567
    Brick(
        np.array([
            [[2, 2, 2]],
            [[2, 0, 0]],
            [[2, 0, 0]],
        ]),
        np.array([
            [[1, 1, 1]],
            [[1, 0, 0]],
            [[1, 0, 0]],
        ]), "32056_double", 320567, np.array([0, 0, 0])),
]


brickConnectors = [
    # 3x5 H shape
    Brick(
        np.array([
            [[2, 0, 0, 0, 2]],
            [[2, 1, 1, 1, 2]],
            [[2, 0, 0, 0, 2]],
        ]),
        np.array([
            [[1, 0, 0, 0, 1]],
            [[1, 1, 1, 1, 1]],
            [[1, 0, 0, 0, 1]],
        ]),
        "Beam 3x5 H shape", 14720, np.array([1, 0, 2])),
    # Connector toggle joint
    Brick(
        np.array([
            [[3, 0, 3],
             [3, 0, 3],
             [2, 2, 2]],
        ]),
        np.array([
            [[1, 0, 1],
             [1, 0, 1],
             [1, 1, 1]],
        ]),
        "Connector toggle joint", 87408, np.array([0, 2, 1])),        
    # Cross Block 1x3
    Brick(
        np.array([
            [[1],
             [3],
             [1]],
        ]),
        np.array([
            [[1],
             [1],
             [1]],
        ]),
        "Cross Block 1x", 32184, np.array([0, 1, 0])),
    # Connector
    Brick(
        np.array([
            [[1, 3]],
        ]),
        np.array([
            [[1, 1]],
        ]),
        "Connector", 32039, np.array([0, 0, 1])),        
    # Cross Block
    Brick(
        np.array([
            [[1], 
             [3]],
        ]),
        np.array([
            [[1], 
             [1]],
        ]),
        "Cross Block", 6536, np.array([0, 0, 0])),
    # Beam 3    
    Brick(
        np.array([
            [[2, 2, 2,]],        
        ]),
        np.array([
            [[1, 1, 1]],        
        ]), "Beam 3", 17141, np.array([0, 0, 1])),  
    # Beam 5    
    Brick(
        np.array([
            [[2, 2, 2, 2, 2,]],        
        ]),
        np.array([
            [[1, 1, 1, 1, 1]],        
        ]), "Beam 5", 32316, np.array([0, 0, 2])),    
    # Beam 7
    Brick(
        np.array([
            [[2, 2, 2, 2, 2, 2, 2,]],
        ]),
        np.array([
            [[1, 1, 1, 1, 1, 1, 1]],        
        ]), "Beam 7", 16615, np.array([0, 0, 3])),
    # Beam 9 
    Brick(
        np.array([
            [[2, 2, 2, 2, 2, 2, 2, 2, 2,]],        
        ]),
        np.array([
            [[1, 1, 1, 1, 1, 1, 1, 1, 1]],        
        ]), "Beam 9", 64289, np.array([0, 0, 4])),
    # Beam 3x3-T    
    Brick(
        np.array([
            [[0, 0, 2]],
            [[2, 2, 2]],
            [[0, 0, 2]],        
        ]),
        np.array([
            [[0, 0, 1]],
            [[1, 1, 1]],
            [[0, 0, 1]],        
        ]), "Beam 3x3-T", 60484, np.array([1, 0, 0])),    
    # Beam 3x5 bent 90    
    Brick(
        np.array([
            [[2, 2, 2, 2, 2]],
            [[0, 0, 0, 0, 2]],
            [[0, 0, 0, 0, 2]],
        ]),
        np.array([
            [[1, 1, 1, 1, 1]],
            [[0, 0, 0, 0, 1]],
            [[0, 0, 0, 0, 1]],
        ]), "Beam 3x5 bent 90", 32526, np.array([0, 0, 0])),
    # Beam 2x4 bent 90    
    Brick(
        np.array([
            [[2, 2, 2, 2]],
            [[0, 0, 0, 2]],
        ]),
        np.array([
            [[1, 1, 1, 1]],
            [[0, 0, 0, 1]],
        ]), "Beam 2x4 bent 90", 32140, np.array([0, 0, 0])),
    # Angle Connector
    Brick(
        np.array([
            [[3, 1]],
        ]),
        np.array([
            [[1, 1]],
        ]), "Angle Connector", 32013, np.array([0, 0, 1])), 
    # Angle Connector 90    
    Brick(
        np.array([
            [[0, 2],
             [3, 1]],
        ]),
        np.array([
            [[0, 1],
            [1, 1]],
        ]), "Angle Connector 90", 32014, np.array([0, 1, 1])),
    # Cross Block 2x2 bent 90    
    Brick(
        np.array([
            [[1, 2]],
            [[0, 3]],
        ]),
        np.array([
            [[1, 1]],
            [[0, 1]], 
        ]), "Cross Block 2x2 bent 90", 44809, np.array([0, 0, 1])),
    # Cross Block 2x4  SHIFTED on z
    Brick(
        np.array([
            [[3, 0, 0, 3],
             [1, 0, 0, 1]],
        ]),
        np.array([
            [[1, 1, 1, 1],
             [1, 0, 0, 1]],
        ]), "Cross Block 2x4  ", 80910, np.array([0, 0, 1])),
    # Cross Block 3x2
    Brick(
        np.array([
            [[0],
             [3]],
            [[1],
             [3]],
            [[0],
             [3]],
        ]),
        np.array([
            [[0],
             [1]],
            [[1],
             [1]],
            [[0],
             [1]],
        ]), "Cross Block 3x2", 63869, np.array([1, 0, 0])),        
]

# unusable connectors from the article
specialConnectors = [
    Brick(
        np.array([
            [[0, 0],
             [0, 0]],
            [[0, 0],
             [0, 0]],
        ]),
        np.array([
           [[1, 0],
            [1, 0]],
            [[1, 0],
             [1, 1]],
        ]), "2p_2o_15100_3", 25261510083, np.array([0, 0, 0])),
    Brick(
        np.array([
            [[0, 0],
             [0, 0]],
            [[0, 0],
             [0, 0]],
        ]),
        np.array([
            [[1, 1],
             [1, 1]],
            [[1, 1],
             [0, 0]],
        ]), "2p_2o_60483", 252660483, np.array([0, 1, 0])),
    Brick(
        np.array([
            [[0],
             [0],
             [0]],
            [[0],
             [0],
             [0]],
        ]),
        np.array([
            [[1], [1], [1]],
            [[0], [0], [0]],
        ]), "3p_2o_15100", 352615100, np.array([0, 0, 0])),
    Brick(
        np.array([
            [[0, 2, 0],
             [0, 0, 0]]
        ]),
        np.array([
            [[1, 1, 1],
             [1, 1, 1]],
        ]), "2p_2o_32523", 252632523, np.array([0, 0, 1])),
    Brick(
        np.array([
            [[0, 0],
             [0, 0]],
            [[0, 0],
             [0, 0]],
        ]),
        np.array([
            [[0, 1],
             [0, 1]],
            [[0, 1],
             [1, 1]],
        ]), "2p_2o_15100_2", 25261510082, np.array([0, 0, 0])),
    Brick(
        np.array([
            [[0]]
        ]),
        np.array([
            [[1]],
        ]), "single", 18654, np.array([0, 0, 0])),
]

stateConnectors = {
    43093: Brick(None, np.array([[[1]], [[1]]]), "Axle pin", 43093, np.array([0,0,0])),    
    65487: Brick(np.array([[[2]], [[0]]]), np.array([[[1]], [[1]]]), "Techinc pin with hole", 65487, np.array([0,0,0])),
    32013: Brick(np.array([[[3, 1]]]), np.array([[[1]], [[1]]]), "Angle connector", 32013, np.array([0,0,0])),    
    18651: Brick(None, np.array([[[1]], [[1]], [[1]]]), "Axle pin long", 18651, np.array([1,0,0])),    
    62462: Brick(None, np.array([[[1]],[[1]]]),"Pin joiner", 62462, np.array([0,0,0])),
    3673: Brick(None, np.array([[[1]], [[1]]]), "Technic pin", 3673, np.array([0, 0, 0])),

}