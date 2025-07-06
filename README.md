# 3D GSDR Dataset

The goal of this repository is to fabricate a stack of datasets to train 3D Gaussian Splatting models working with reflections. 

<div align="center">
  <img src="public/gif/single_obj.gif" width="35%" />
  <img src="public/gif/multiple_obj.gif" width="35%" />
</div>

## Set up:

Install Blender in home directory

```bash
cd ~
mkdir -p ~/software
cd ~/software
wget https://download.blender.org/release/Blender4.0/blender-4.0.0-linux-x64.tar.xz
tar -xf blender-4.0.0-linux-x64.tar.xz
```
Install PyYAML for Blender

```bash
# find Blender's python
find ~/software/blender-4.0.0-linux-x64 -name "python*" | grep bin

# install PyYAML 
~/software/blender-4.0.0-linux-x64/4.0/python/bin/python3.10 -m pip install pyyaml
```

## Project Structure 

```bash
project/
├── README.md
├── assets/
│   ├── paris.exr
│   └── ... # contains all hdri files
├── configs/
│   ├── cube.yaml
│   └── ... # contains all config files
├── output/
│   ├── cube/
|       ├── cube_000.png
|       └── ... 
│   └── ... # contains all image folders for each render
│   └── layout.md 
├── scripts/
│   ├── __init__.py
│   ├── cube.py
│   └── ... # contains all render scripts
└── .gitignore
└── requirements.txt
└── run.py # --> main landing script runs + saves images for all render scripts 

```

## How to run?

Run this from the project root, it will generate datasets for all scenes. Currently there is only one render, which is a simple reflecting cube with an environment map, but that is to change in due time. The rendered images will be saved in their respective subfolders in the 'output' directory. For example, the image files for cube rendering are generated in output/cube/cube_000.png through cube_035.png.

```bash
# run from project root
cd ~/gs-dataset
~/software/blender-4.0.0-linux-x64/blender --background --python run.py
```
> Add -- --cycles-device CUDA at the end of the command if you want GPU rendering