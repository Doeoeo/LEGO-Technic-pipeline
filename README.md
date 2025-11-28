# LEGO Technic Pipeline
A fully automated pipeline for converting arbitrary 3D models into LEGO Technic representations.

## Requirements
Before running the code, several external sources must be placed into the corresponding folders included in this repository:

- **ComputeTechnic** (compiled version)  
  → https://github.com/xuhaocuhk/ComputeTechnic

- **Hough 3D Lines**  
  → https://github.com/cdalitz/hough-3d-lines

- **binvox** (for voxelization)  
  → https://www.patrickmin.com/binvox/

Place each resource into the empty folders provided in the repository structure.

## How to Run
1. Add your input model to the `Objects` folder in the following format:  
   - `Objects/<object_name>.obj`  
   - `Objects/<object_name>/object/<object_name>.glb`

2. Run the main conversion script:

```bash
python LegoConverter.py
