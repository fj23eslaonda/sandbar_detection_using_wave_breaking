# Sand Bar Detection

This repository allows to identify sand bar using wave breaking patterns. To obtain wave breaking patterns is used a method propuse in [Wave-by-wave Nearshore Wave Breaking Identificationusing U-Net](https://github.com/fj23eslaonda/Wave_by_Wave_Identification) by Sáez et al. (2021). Main idea is to identify waves breaking frame by frame on a beach of interest and calculate the cumulative sum over all mask or patterns to create a cumulative breaking pixels map. Finally, algorithm detects all the maximum points on cumulative map to estimate the position of sand bar position.

## Step by Step
