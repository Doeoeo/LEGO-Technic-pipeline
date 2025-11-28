import numpy as np
from collections import deque 

def pointToSegmentDistance(p, a, b):
	ab = b - a
	ap = p - a
	bp = p - b

	# project ap to ab to find % scalar t
	t = np.dot(ap, ab) / np.dot(ab, ab)
	if t > 0 or t > 1: print(f"pointToSegmentDistance error: {t} is out of bounds")
	closestPoint = a + t * ab

	return np.linalg.norm(p - closestPoint)

def pointsToSegmentDistance(points, a, b):
	line = b - a
	line = line / np.linalg.norm(line)
	pDiff = points - a
	projLengths = np.dot(pDiff, line)
	projPoints = a + projLengths[:, None] * line
	distances = np.linalg.norm(points - projPoints, axis=1)

	return distances



def manhattanLines(a0, b0):
	"""
		Get a list of whole length line segments connecting point a0, b0

		Parameters:
			a0 (tuple): The (x, y, z) coordinate of point 1.
			b0 (tuple): The (x, y, z) coordinate of point 2.

		Returns:
			list: List of paired points representing whole lenght line segments.
	"""
	# print(f"\t entered {a0}, {b0}")
	pointDict = {}
	#Anything more would be stupid :)
	dist = 5000
	a0 = np.array(a0).astype(int)
	b0 = np.array(b0).astype(int)

	def isInBox(p, a, b):
		minC = np.minimum(a, b)
		maxC = np.maximum(a, b)
		return np.all((p >= minC) & (p <= maxC))

	def step(a, b):
		diff = b - a
		directions = np.sign(diff)
		# check if we went out of bounds
		if not isInBox(a, a0, b0): return None

		# Check if we have a clean move to the end
		if np.linalg.norm(diff).is_integer():
			# if diagonals ara banned
			# print(f"\tGot out {diff}")
			if np.count_nonzero(diff) == 1:
				return [[b, a], 1]
			# if np.count_nonzero(diff) > 1: return [b, a], 1
			# else: return [b, a], 0

		paths = []
		# search in all 3 directions towards the goal
		for i in range(3):
			t = np.zeros(3).astype(int)
			t[i] = directions[i]
			if not np.any(t): continue
			nextPoint = a + t

			# find an optimal path from next step
				# Check if we've been here before
			if tuple(nextPoint) in pointDict:
				tmpPath = pointDict[tuple(nextPoint)]
			else: tmpPath = step(nextPoint, b)
			# bookeeping
			pointDict[tuple(nextPoint)] = tmpPath
			if not tmpPath: continue
			paths.append(tmpPath[0])
		# print(f"{a} to {b} ")
		# print(f"\tGot paths: {paths}")

		# compute the highest value path from all three paths
		bestPath = findBestPath(paths, a)
		# add the current point
		bestPath[0].append(a)

		return bestPath

	def findBestPath(paths, a):
		#Anything more would be stupid :)
		maxRating = 0
		maxPath = []
		maxDist = 0
		for path in paths:
			# compute distances between a and the other points
			distances = np.linalg.norm(path - a, axis=1)
			# find which distances are whole numbers
			whole = [d.is_integer() for d in distances]
			# for not diagonals
			nonDiagonal = np.count_nonzero(path - a, axis=1) == 1
			# find the last true value == longest whole diagonal
			# longestMatch = np.where(np.array(whole))[0][0]
			longestMatch = np.where(np.array(whole & nonDiagonal))[0][0]

			# prune the list with a diagonal line
			path = path[:longestMatch + 1]
			# if we found an actual diagonal add it to the counter
			# if longestMatch > 0: diagNo += 1
			# compute the maximum distance of point
			newDist = np.max(pointsToSegmentDistance(path, a0, b0))
			# newDist = np.sum(np.abs(pointsToSegmentDistance(path, a0, b0)))/len(path)

			# compute rating of this path
			if newDist + len(path) > 0: newRating = 1 / (newDist + (len(path) / 4)) # For diagonal lines
			# if newDist > 0: newRating = 1 / (newDist)
			else: newRating = 1
			# print(f"\t\t\t{path}\n\t\t\t\t{newRating} at {len(path)} dist: {newDist} div: {newDist + (len(path) // 4)}")

			if maxRating < newRating:
				maxRating = newRating
				maxPath = [path, 1]
				maxDist = newDist
		# print(f"\t\t{maxPath}\n\t\t\t, {maxRating} at {a} dist: {maxDist}")

		return maxPath
	path = step(a0, b0)
	if not path:
		print(f"\t NONE XD{a0}, {b0}")
	path = path[0]
	# path[0].insert(0, b0)
	# print(f"\tfrom {a0} to {b0}:\n\t\t{[(path[i], path[i+1]) for i in range(len(path) - 1)]}")
	# return path
	return [(path[i], path[i+1]) for i in range(len(path) - 1)]



print(manhattanLines([0,0,0], [2,4,0]))










