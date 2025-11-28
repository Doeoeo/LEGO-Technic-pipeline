import numpy as np
from collections import defaultdict 

test = [(np.array([1,0,1]), np.array([1,6,1])),
		(np.array([1,2,1]), np.array([5,2,1])),
		(np.array([3,6,1]), np.array([3,1,1])),
		(np.array([3,6,1]), np.array([1,6,1])),
		(np.array([0,4,1]), np.array([3,4,1]))]

def removeOverlaps(lines):
	"""
		merge the given line semgents to remove line overlaps

		Parameters:
			lines (List): coordinate pairs representing straight line segments.

		Returns:
			list: List of coordinate pairs representing straight line segments that do not overlap.
	"""
	def linesToVox(lines):
		# obtain minimal bounding box
		lineList = np.array(lines).reshape(len(lines)*2, -1)
		bboxSize = np.max(lineList, axis=0) + 3 #- np.min(lineList, axis=0)

		grid = np.zeros(bboxSize, dtype=int)
		lineDict = defaultdict(list)

		index = 0
		for a,b in lines:
			# find dim of movement
			dim = np.where(a != b)[0][0]
			# create steps between points (include a, b)
			tmp = sorted([a[dim], b[dim]])
			steps = np.arange(tmp[0], tmp[1] + 1)
			# create points between a,b 
			points = np.zeros((len(steps), 3), dtype=int) + a
			points[:, dim] = steps
			# add 1 to all points to we avoid edge cases
			points = points + 1
			# add line indices to dict
			for p in points: 
				lineDict[tuple(p)].append(index)
			# set grid points to used
			grid[tuple(np.transpose(points))] += 1

			index += 1

		return grid, lineDict

	def voxToLines(grid, lineDict, lines):
		# convert tuples to lists
		lines = [list((tuple(i[0]), tuple(i[1]))) for i in lines]
		for p in np.argwhere(grid > 1):
			# get lines that occupy the voxel
			linesOnPoint = lineDict[tuple(p)]
			# print(f"on point{np.array(p)-1} we have lines {linesOnPoint}")
			for i in linesOnPoint:
				# print(f"\ton line {i} with {lines[i]} added point {np.array(p)-1} ")

				lines[i].append(tuple(p - 1))
		# sort line coordinates
		lines = [sorted(list(set(i))) for i in lines]
		# print(lines[0])
		newLines = set([])
		# for every list of line splits create individual lines added to a set to remove duplicates
		for l in lines:
			for a,b in zip(l[:-1], l[1:]):
				newLines.add((a, b))

		# convert lines back to nparrays
		newLines = [(np.array(i[0]), np.array(i[1])) for i in list(newLines)]

		return newLines


	# create a voxel representation of lines
	grid, lineDict = linesToVox(lines)
	# obtain a line representation of the grid
	lines = voxToLines(grid, lineDict, lines)
	return lines

# for a,b in removeOverlaps(test):
# 	print(f"{tuple(a[:2])} - {tuple(b)[:2]}")