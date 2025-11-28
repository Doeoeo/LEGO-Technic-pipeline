import os
import subprocess
from pathlib import Path

def run_complex_command(command):
	print(command)
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


def generateFrame(shortName):
	command = r"..\ComputeTechnic-main\lego_technic_main.exe -path_provide " + f"\"..\\Objects\\{shortName}Lines.obj\""
	print("\tObtaining frame from line object")
	# move working directory to contain all the files
	oldCwd = os.getcwd()
	os.chdir("Lines")
	rc, output, errs = run_complex_command(command)
	# restore working directory
	os.chdir(oldCwd)
	print("\t\tObtained frame from line object")
	print("\tMoving output file")
	path = Path(output.splitlines()[-1].split()[1][3:])
	newPath = Path("SavedStates") / (shortName + path.suffix)
	path.rename(newPath)

# if __name__ == '__main__':
#     generateFrame("car_NoWheels")
