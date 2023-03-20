# OMM Sample

*Opacity Micro-Map sample* demostrates usage of Opacity Micro-Maps in Raytracing. Sample details:
- Opacity Micro-Map SDK is used for OMM baking. Both CPU baker and GPU baker are supported in the sample: https://github.com/NVIDIAGameWorks/Opacity-MicroMap-SDK
- Rendering and Denoising are based on NRDSample: https://github.com/NVIDIAGameWorks/NRDSample

## Build instructions

- Install [*Cmake*](https://cmake.org/download/) 3.15+
- Install on
    - Windows: latest *WindowsSDK* (22000+), *VulkanSDK* (1.3.231+)
- Build (variant 1) - using *Git* and *CMake* explicitly
    - Clone project and init submodules
    - Generate and build project using *CMake*
- Build (variant 2) - by running scripts:
    - Run `1-Deploy`
    - Run `2-Build`

### CMake options

- `USE_MINIMAL_DATA=ON` - download minimal resource package (90MB) - set ShaderBalls.obj scene via cmdargs
- `DISABLE_SHADER_COMPILATION=ON` - disable compilation of shaders (shaders can be built on other platform)
- `DXC_CUSTOM_PATH=custom/path/to/dxc` - custom path to *DXC* (will be used if VulkanSDK is not found)
- `USE_DXC_FROM_PACKMAN_ON_AARCH64=OFF` - use default path for *DXC*

## How to run

- Run `3-Run OMM sample` script and answer the cmdline questions to set the runtime parameters
- The executables can be found in `_Build`. The executable loads resources from `_Data`, therefore please run the samples with working directory set to the project root folder (needed pieces of the command line can be found in `3-Run OMM sample` script)

## Command Line Arguments
- If [Smart Command Line Arguments extension for Visual Studio](https://marketplace.visualstudio.com/items?itemName=MBulli.SmartCommandlineArguments) is installed, all command line arguments will be loaded into corresponding window
- `--api=D3D12`, `--api=VULKAN` for graphics API selection 
- `--width=x`, `--height=y` for setting window size
- `--scene=*path*` for scene selection
- `--help` to print all the available commands

## Minimum Requirements

Any RTX GPU:
- RTX 4000 series
- RTX 3000 series
- RTX 2000 series

## Usage

OMM:
- Set baker settings in the UI and press Bake OMMs
- For CPU baker it is recommended to use cache

Navigation:
- Right mouse button + W/S/A/D - move camera
- Mouse scroll - accelerate / decelerate
- F1 - toggle "gDebug" (can be useful for debugging and experiments)
- Tab - UI toggle
- Space - animation toggle

Notes:
- Check "Show all settings" in the UI to see rendering setting.

## OMM Workflow

1. Bake OMM data with Opacity Micro-Map SDK
2. Use OMM data to build OMM with corresponding API functions
3. Build BLAS using OMM