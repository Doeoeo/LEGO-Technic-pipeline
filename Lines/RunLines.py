import pickle
import subprocess
import numpy as np
import math
from collections import deque, defaultdict
from itertools import chain
from Lines.ManhattanLines import manhattanLines
from Lines.RemoveOverlaps import removeOverlaps
def plotGNU(input_string):
	"""
	Runs a complex command where a dynamically generated string is passed to gnuplot via a pipe.

	:param input_string: The string to pass to gnuplot.
	:return: Tuple of (return_code, stdout, stderr).
	"""
	try:
		# Start the gnuplot process
		gnuplot_process = subprocess.Popen(
			["gnuplot", "-persist"],
			stdin=subprocess.PIPE,  # Allow passing input to gnuplot
			stdout=subprocess.PIPE,  # Capture stdout of gnuplot
			stderr=subprocess.PIPE,  # Capture stderr of gnuplot
			text=True				# Input/output as text
		)
		
		# Pass the string to gnuplot
		stdout, stderr = gnuplot_process.communicate(input=input_string)
		
		return gnuplot_process.returncode, stdout, stderr
	except Exception as e:
		return -1, "", str(e)


def remove_points_near_line(points, a, b, l, n):
	"""
	Removes points in `points` that are closer than distance `l` to the line defined by points `a` and `b`.

	Parameters:
		points (np.ndarray): Array of shape (n, 3), representing the points in space.
		a (np.ndarray): A point defining the line (shape (3,)).
		b (np.ndarray): Another point defining the line (shape (3,)).
		l (float): Distance threshold.

	Returns:
		np.ndarray: Filtered array of points.
	"""
	def findOptimalSegment(a, b, points, thr):
		"""
		Recursively prunes the segment [a, b] based on point density.

		Parameters:
			a (float): Start of the segment.
			a (float): End of the segment.
			points (np.ndarray): Projections within the segment.

		Returns:
			float: Density
			tuple: Retained segments [(a, b)].
		"""
		segLen = b - a
		nPoints = len(points)
		endpoints = (a, b)
		density = nPoints / segLen if segLen > 0 else 0
		# print(f"New call: Len: {segLen}, points: {nPoints}, density: {density}, endpoints: {endpoints} | thr {thr}")

		if nPoints < 2: 
			# print("Got out") 
			return 0, endpoints
		if density >= thr or segLen < 2:
			# print("Got out") 
			return density, endpoints

		midPoint = (a + b) / 2
		left = points[points <= midPoint]
		right = points[points > midPoint]

		lQuality, lEndpoints = findOptimalSegment(a, midPoint, left, thr)
		if lQuality > density:
			density = lQuality
			endpoints = lEndpoints
		if lQuality >= thr:
			# print("Got out") 
			return density, endpoints
		rQuality, rEndpoints = findOptimalSegment(midPoint, b, left, thr)
		if rQuality > density:
			density = rQuality
			endpoints = rEndpoints		

		return density, endpoints


	# Line direction vector
	v = b - a
	v_norm = np.linalg.norm(v)
	if v_norm == 0:
		raise ValueError("Points a and b must not be the same.")
	v_unit = v / v_norm  # Normalize the direction vector

	# Compute distances and filter points near the line
	diff = points - a
	projection_lengths = np.dot(diff, v_unit)  # Scalar projections onto the line
	projection_points = a + projection_lengths[:, None] * v_unit  # Projected points on the line
	distances = np.linalg.norm(points - projection_points, axis=1)  # Perpendicular distances

	# Mask for points within distance `l` of the line

	mask = distances < l
	harshMask = distances < 0.1 # we fake project points onto the line for now
	if np.sum(mask) < 2: return points, None, None, None

	uValues = projection_lengths[mask]
	uValues = np.sort(uValues)
	density, newEndpoints = findOptimalSegment(uValues[0], uValues[-1], uValues, 0.9)
	if density == 0: 
		# newEndpoints = [uValues[0], uValues[-1]]
		return points, None, None, None 
	mask = mask & (projection_lengths >= newEndpoints[0]) & (projection_lengths <= newEndpoints[1])
	
	# Find the two endpoints based on projection lengths
	if np.any(mask):
		min_proj_idx = int(np.argmin(projection_lengths[mask]))
		max_proj_idx = int(np.argmax(projection_lengths[mask]))
		endpoints = (np.round(projection_points[mask][min_proj_idx]).astype(int), np.round(projection_points[mask][max_proj_idx]).astype(int))
	else: return points, None, None, None 
	# print("------------------------------------------------------------------------------------------------------------------------")
	# mask[min_proj_idx] = False
	# mask[max_proj_idx] = False


	removed_points = points[mask & harshMask]
	remaining_points = points[(~mask)]
	mask[harshMask] = False
	projected_points = points[mask]

	return np.round(remaining_points).astype(int), np.round(removed_points).astype(int), endpoints, projected_points

def parse_string_to_tuples(data):
	"""
	Parses a multiline string into a list of tuples.

	:param data: The string containing multiple lines, each formatted as:
				 aX aY aZ bX bY bZ npoints
	:return: A list of tuples where each entry is ((aX, aY, aZ), (bX, bY, bZ)).
	"""
	result = []
	n = []
	uMax, uMin = 0, 0
	for line in data.strip().split("\n"):  # Split the string into lines
		parts = line.split()  # Split each line into components
		if len(parts) == 2:
			uMin = float(parts[0])
			uMax = float(parts[1])
		if len(parts) == 7:  # Ensure the line has the expected 7 values
			a = tuple(map(float, parts[:3]))  # First three values as (aX, aY, aZ)
			b = tuple(map(float, parts[3:6]))  # Next three values as (bX, bY, bZ)
			result.append((a, b))  # Append the pair of tuples to the result
			n.append(int(parts[6]))
	return result, (uMin, uMax), n

def run_complex_command(command):
	try:
		# Run the command in shell mode
		result = subprocess.run(
			command,
			shell=True,			 # Use the shell to interpret the pipe
			text=True,			  # Capture output as text
			capture_output=True,	# Capture stdout and stderr
			check=False			 # Don't raise exception for non-zero exit codes
		)
		return result.returncode, result.stdout, result.stderr
	except Exception as e:
		return -1, "", str(e)

def run_gnuplot_with_lines(datafile, lines):
	"""
	Runs Gnuplot dynamically to plot points from a data file and lines defined by endpoints.

	Parameters:
		datafile (str): Path to the data file containing points.
		lines (list of tuples): List of tuples where each tuple contains two endpoints (start, end).
	"""
	gnuplot_commands = []

	# Gnuplot settings
	gnuplot_commands.append("set datafile separator ','")
	gnuplot_commands.append("set parametric")
	gnuplot_commands.append("set xrange [1:45]")
	gnuplot_commands.append("set yrange [1:64]")
	gnuplot_commands.append("set zrange [1:54]")
	gnuplot_commands.append("set urange [0:1]")

	# Plot points from the data file
	plot_cmd = f"splot '{datafile}' using 1:2:3 with points palette"
	
	# Add each line dynamically
	for start, end in lines:
		if start is not None and end is not None:
			line_x = f"{start[0]} + u * ({end[0]} - {start[0]})"
			line_y = f"{start[1]} + u * ({end[1]} - {start[1]})"
			line_z = f"{start[2]} + u * ({end[2]} - {start[2]})"
			plot_cmd += f", \\\n	{line_x}, {line_y}, {line_z} with lines notitle lc rgb 'black'"
	plot_cmd += "\n"
	
	gnuplot_commands.append(plot_cmd)
	print("\n".join(gnuplot_commands))
	# Run Gnuplot as a subprocess
	process = subprocess.Popen(["gnuplot", "-persist"], stdin=subprocess.PIPE, text=True)
	process.communicate("\n".join(gnuplot_commands))
	print("Gnuplot executed successfully.")

def toObj(lines, name="testObj" ):
	vCounter = 1
	vertices = {}
	objLines = []
	for l in lines:
		v1 = tuple(np.round(l[0]).astype(int))
		if not v1 in vertices: 
			vertices[v1] = vCounter
			vCounter += 1
		v2 = tuple(np.round(l[1]).astype(int))
		if not v2 in vertices: 
			vertices[v2] = vCounter
			vCounter += 1

		objLines.append((vertices[v1], vertices[v2]))
	with open(f"Objects/{name}Lines.obj", 'w') as f:
		for v in vertices.keys():
			f.write(f"v {' '.join(map(str, v))}\n")
		for l in objLines:
			f.write(f"l {' '.join(map(str, l))}\n")

def find_shortest_paths(valid_coords, requieredEndpoints, valid_endpoints, removeDict, projected):
	"""
	Finds the closest pairings of endpoints using valid coordinates as intermediate paths.

	Parameters:
		valid_coords (np.ndarray): Array of shape (n, 3), representing valid coordinates in 3D space.
		valid_endpoints (np.ndarray): Array of shape (m, 3), representing valid endpoints in 3D space.

	Returns:
		list of tuple: List of endpoint pairs [(endpoint1, endpoint2)], representing the shortest connections.
	"""
	# Convert coordinates to a set for O(1) lookups
	valid_set = set(list(map(tuple, valid_coords)))
	endpoint_set = set(list(map(tuple, valid_endpoints)))
	projected_set = set(list(map(tuple, projected)))
	requieredEndpointsSet = set(list(map(tuple, [tup for pair in requieredEndpoints for tup in pair])))

	
	# Function to get neighbors
	def get_neighbors(coord, otherEndpoint):
		"""
		Get all 26 neighbors of a given coordinate in 3D space.

		Parameters:
			coord (tuple): The (x, y, z) coordinate.
			otherEndpoint (tuple): The (x, y, z) coordinate of the other endpoint.

		Returns:
			list: List of neighboring coordinates that are valid.
		"""
		x, y, z = coord
		neighbors = [
			(x + dx, y + dy, z + dz)
			for dx in [-1, 0, 1]
			for dy in [-1, 0, 1]
			for dz in [-1, 0, 1]
			if not (dx == 0 and dy == 0 and dz == 0)  # Exclude the point itself
		]
		# Convert tuples to NumPy arrays for vector calculations
		A = np.array(otherEndpoint)  # Previous point in the line
		B = np.array(coord)  # Current point
		dir_AB = B - A  # Direction from A to B
		# Normalize to unit vector
		if np.linalg.norm(dir_AB) > 0:
			dir_AB = dir_AB / np.linalg.norm(dir_AB)

		valid_neighbors = []
		for C in neighbors:
			C = np.array(C)
			dir_BC = B - C  # Direction from B to C

			# Normalize to unit vector
			if np.linalg.norm(dir_BC) > 0:
				dir_BC = dir_BC / np.linalg.norm(dir_BC)

				# Compute dot product between AB and BC
				dot_product = np.dot(dir_AB, dir_BC)

				# Only allow movement if angle is ≤ 90 degrees (dot product ≤ 0)
				if dot_product <= 0 and (tuple(C) in valid_set or tuple(C) in endpoint_set or tuple(C) in projected_set):
					valid_neighbors.append(tuple(C))
		return valid_neighbors
		# return [n for n in neighbors if (n in valid_set) or (n in endpoint_set)] 

	# BFS to find the shortest path between two endpoints
	def bfs(start, target_set, otherEndpoint):#ownLine, otherEndpoint):
		visited = set()
		queue = deque([(start, [start])])  # (current point, path to current point)
		# target_set = set(map(tuple, targets))
		# print(f"Staring on {start} endpoint on {otherEndpoint}")
		while queue:
			current, path = queue.popleft()
			# print(f"\tNext step {current}")
			
			# Check if we've reached a target endpoint
			# if current != start and (not current in ownLine) and current in target_set:
			if current != start and current in target_set:
				# print(f"\tFound a path {path}")
				return path
			
			if current in visited:
				continue
			visited.add(current)
			# Explore neighbors
			# tmp = get_neighbors(current)
			for neighbor in get_neighbors(current, otherEndpoint):
				if neighbor not in visited:
					queue.append((neighbor, path + [neighbor]))
		
		# print("\tNo path")
		return None  # No path found

	# Keep track of connections
	endpoint_pairs = []
	connected_endpoints = set()

	for endpoints in requieredEndpoints:
		for i in range(len(endpoints)):
			endpoint = endpoints[i]
			otherEndpoint = endpoints[(i + 1) % 2]
			if tuple(endpoint) in connected_endpoints:
				continue
			
			# Find the closest pairing for this endpoint
			path = bfs(tuple(endpoint), requieredEndpointsSet, otherEndpoint)# removeDict[tuple(endpoint)], otherEndpoint)
			
			if path:
				# The start and end of the path are the paired endpoints
				start, end = path[0], path[-1]
				# Get a line or lines that are of integer length
				lines = manhattanLines(start, end)
				# Add possible new connectable points MIGHT MAKE IT WORSE :)
				for l in lines:
					for k in l:
						k = tuple(k)
						if k not in requieredEndpointsSet:
							requieredEndpointsSet.add(k)



				# print(f"Path from {start} to {end}")
				endpoint_pairs.extend(lines)
				# endpoint_pairs.append((start, end))
				connected_endpoints.add(start)
				connected_endpoints.add(end)
	return endpoint_pairs

def fixLineLength(lines):
	print(f"--------{lines[:10]}")
	newLines = []
	while lines:
		a, b = lines.pop()
		diff = b - a
		if not np.any(diff): continue
		# print(f"{a},{b} == {np.linalg.norm(diff)} {np.linalg.norm(diff).is_integer()}")

		# if np.linalg.norm(diff).is_integer():
		# 	newLines.append((a, b))
		# else:
		newLines.extend(manhattanLines(a, b))

	return newLines

def obtainLines(shortname, filename, skip = False):
	if skip: return np.load(f"{filename}Lines.npy")

	command = (
		r"hough-3d-lines-master\hough3dlines Objects\\" + f"{shortname}.dat "
		"-dx 0.4 -minvotes 2 -nlines 30 -raw"
	)
	command2 = (
		r"hough-3d-lines-master\hough3dlines Objects\\" + f"{shortname}.dat "
		"-dx 0.4 -nlines 30 -minvotes 5 -gnuplot"
	)
	#| gnuplot -persist
	points = np.loadtxt(f"Objects/{shortname}.dat", dtype=int, delimiter=",").astype(int)
	np.savetxt(f"Objects/{shortname}Backup.dat", points, delimiter=",", fmt="%d")
	allLines = []
	allRemoved = []
	allProjected = []
	removeDict = {}
	removeCounter = 1
	streak = True
	# run_gnuplot_with_lines("hough-3d-lines-master/test.dat", [])

	while True:
	# while False:
		removed = []
		lines = []
		return_code, stdout, stderr = run_complex_command(command)
		c, s, e = run_complex_command(command2)
		stdout, u, n = parse_string_to_tuples(stdout)

		index_min = np.argsort(n)[-min(removeCounter, len(n) - 1)]
		print(f"\tNew set: {index_min}")
		stdout = [stdout[index_min]]
		for i in stdout:
			a0 = np.array(i[0])
			bDir = np.array(i[1])
			a = a0 + (u[0] + 5) * bDir
			b = a0 + (u[1] - 5) * bDir

			points, newRemoved, endpoints, projected = remove_points_near_line(points, np.array(a), np.array(b), 1.5, n.pop(0))
			if not endpoints: continue
			newRemovedSet = set(list(map(tuple, newRemoved)))
			removeDict[tuple(endpoints[0])] = newRemovedSet
			removeDict[tuple(endpoints[1])] = newRemovedSet
			lines.append(endpoints)
			removed.extend(newRemoved)
			allRemoved.extend(newRemoved)
			allProjected.extend(projected)
		if len(removed) == 0: removeCounter += 1
		else: streak = False
		if removeCounter > 29:
			break
			if streak: break
			else: 
				removeCounter = 1
				streak = True
		allLines.extend(lines)
		np.savetxt(f"Objects/{shortname}Backup.dat", points, delimiter=",", fmt="%d")
		np.savetxt(f"Objects/{shortname}BackupPlot.dat", removed, delimiter=",", fmt="%d")
		
		# plotGNU(s.replace("test2.dat", "test2.dat"))
		# run_gnuplot_with_lines("hough-3d-lines-master/test2.dat", tmp)
		print(f"\t\tRemoved: {len(removed)} | Remaining {len(points)}")
		# print(f"{allProjected}")

	with open(f"Objects/{shortname}lines.txt", 'wb') as fp:
		pickle.dump(allLines, fp)
	# with open ('hough-3d-lines-master/lines.txt', 'rb') as fp:
	# 	allLines = pickle.load(fp)	
	allLines = fixLineLength(allLines)

	extraLines = find_shortest_paths(points, allLines, allRemoved, removeDict, allProjected)
	allLines.extend(extraLines)
	allLines = removeOverlaps(allLines)
	# run_gnuplot_with_lines(f"Objects/{shortname}Backup.dat", allLines)

	toObj(allLines, name=shortname)
	np.savetxt(f"Objects/{shortname}BackupPlot.dat", allRemoved, delimiter=",", fmt="%d")

	np.save(f"{filename}Lines.npy", points)

	return points












#
#		PRESLIKAJ TOCKE NA PREMICO!!!!
#