import os
import subprocess


def run_binvox(obj_name, scale):
    original_cwd = os.getcwd()
    try:
        os.chdir("Objects")
        if os.path.exists(obj_name + ".vtk"):
            os.remove(obj_name + ".vtk")

        subprocess.run(
            ["binvox", obj_name + ".obj", "-d", scale, "-t", "vtk", "-e"],
            check=True
        )
    finally:
        os.chdir(original_cwd)



