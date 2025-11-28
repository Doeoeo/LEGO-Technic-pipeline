#if TYPE_CHECKING:
#    from AnotherModule import Brick
    
#from Bricks import stateConnectors
import numpy as np

from Util.Support import flip, toVec, createRotationMatrix


# States for the path traversal for connecting dissconnected segments
# f -> forward move
# c -> corrner move
stateCon = {}

def addBrick(bricks:list, p, idx, s, d, to):
   d = toVec(*d)
   to = toVec(*to)
   rotations, tmp = flip(dirVectors[idx], d, toVectors[idx])
   rotations2, _ = flip(tmp, to)
   bricks.append((
        p, 
        stateCon[idx], 
        createRotationMatrix(rotations2) @ createRotationMatrix(rotations),
        5 - s))  

def ffc(bricks:list, step:int,  possition:tuple[int, int, int], direction:tuple[int, int], to:tuple[int, int]):
    # Add connector
    addBrick(bricks, possition, 32013, step, direction, to)
    # Add pin
    pinPos = np.array(possition) + np.eye(1, 3, direction[0]).squeeze() * direction[1] * 2
    addBrick(bricks, tuple(pinPos), 18651, step, direction, (direction[0], -direction[1]))
    return restart(f, c)

def fff(bricks:list, step:int,  possition:tuple[int, int, int], direction:tuple[int, int], to:tuple[int, int]):
    # Add pin joiner
    pinPos = np.array(possition) + np.eye(1, 3, direction[0]).squeeze() * direction[1] * 1.5
    addBrick(bricks, tuple(pinPos), 62462, step, direction, (direction[0], -direction[1]))
    # Add pin
    pinPos = np.array(possition) + np.eye(1, 3, direction[0]).squeeze() * direction[1] * 2.5
    addBrick(bricks, tuple(pinPos), 3673, step, direction, (direction[0], -direction[1]))

    return restart(f, c)

def ff(bricks:list, step:int,  possition:tuple[int, int, int], direction:tuple[int, int], to:tuple[int, int]):
    return restart(fff,  ffc)


def fc(bricks:list, step:int,  possition:tuple[int, int, int], direction:tuple[int, int], to:tuple[int, int]):
    # Add connector
    addBrick(bricks, possition, 32013, step, direction, to)
    # Add pin
    pinPos = np.array(possition) + np.eye(1, 3, direction[0]).squeeze() * direction[1] * 1.5
    addBrick(bricks, tuple(pinPos), 43093, step, direction, (direction[0], -direction[1]))
    return restart(f, c)

def f(bricks:list, step:int,  possition:tuple[int, int, int], direction:tuple[int, int], to:tuple[int, int]):
    return restart(ff,  fc)

def c(bricks:list, step:int,  possition:tuple[int, int, int], direction:tuple[int, int], to:tuple[int, int]):
    # Add connector
    addBrick(bricks, possition, 65487, step, direction, to)
    return restart(f, c)


def restart(f1, f2, *args):
    def f(*args):
        if len(args) > 0 and args[1]: 
            return f1(*args)
        else: return f2(*args)
    if args: return f(*args)
    else: return f
    

def fillPath(path, pathType, pathDir, toDir, bricks):
    from ToLegos.Bricks import stateConnectors
    global stateCon
    stateCon = stateConnectors
    state = restart(f, c)
    while path:
        state = state(bricks, pathType.pop(), path.pop(), pathDir.pop(), toDir.pop())
 
        



dirVectors = {
    32013: toVec(2, -1),
    65487: toVec(0, 1),
    43093: toVec(0, -1),
    18651: toVec(0, -1),
    62462: toVec(0, 1),
    3673: toVec(0, 1),
}

toVectors = {
    32013: toVec(0, 1),
    65487: toVec(1, 1),
    43093: toVec(0, 1),
    18651: toVec(0, 1),
    62462: toVec(0, -1),
    3673: toVec(0, -1),

}