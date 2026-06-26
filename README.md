# LEGO Technic Pipeline
### License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

**Simon Cof¹, Ciril Bohak¹**
1. University of Ljubljana, Faculty of Computer and Information Science, Večna pot 113, Ljubljana, 1000, Slovenia

This repository contains a fully automated pipeline for converting arbitrary 3D models into LEGO Technic representations.

![Example of a 3D model converted into a LEGO Technic model.](Images/Plane.png "Example of a 3D model converted into a LEGO Technic model.")


## External sources
The pipeline uses several external sources that must be added to the corresponding folders for the Lego Technic Pipeline to function:

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
    python LegoConverter.py <object_name>
```
If you find this dataset useful, please cite our work.
```bibtex
@masterthesis{cof2026,
  author = {Simon Cof},
  title = {Conversion of 3D Mesh Geometry into Models Composed of LEGO Technic Elements},
  school = {University of Ljubljana, Faculty of Computer and Information Science},
  year = {2026}
}
```





