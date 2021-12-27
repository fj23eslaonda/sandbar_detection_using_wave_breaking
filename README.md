# Sand Bar Detection

This repository allows to identify sand bar using wave breaking patterns. To obtain wave breaking patterns is used a method propuse in [Wave-by-wave Nearshore Wave Breaking Identificationusing U-Net](https://github.com/fj23eslaonda/Wave_by_Wave_Identification) by Sáez et al. (2021). Main idea is to identify waves breaking frame by frame on a beach of interest and calculate the cumulative sum over all mask or patterns to create a cumulative breaking pixels map. Finally, algorithm detects all the maximum points on cumulative map to estimate the position of sand bar position.

## Inputs and parameters
**Parameters**
- `main_path`: Main folder where the repository is cloned    
- `image_path`: Path of frames folder 
- `output_path`: Path of mask folder 
- `beach_path`: Path of beach folder to save results (inside of main_path)
- `plot_mask`: Boolean variable to save plots or not
- `plot_mask_over_img`: Boolean variable to save plots or not
- `orientation`: Waves direction 

**Inputs**

It's necessary to have rectified images of your beach of interest and it's very important than wave direction is from top to bottom (`orientation`: vertical) or from right to left of images (`orientation`: horizontal). That's because the method propuse by Sáez et al. (2021) was trained with a specific direction (horizontal) then if you have vertical direction the algorithm will turn the images.

## Implementation
The algorithm uses tensorflow packages then it's necessary to create a new virtual environment. All packages are in `requirements.txt`.

