# GIS U-Net

A Workflow for applying a Convolutional Neural Network to Geospatial data.
Input data is a multi layer geotiff, output data is a geotiff.
This demo is semantic segmentation, however regression is also possible.



## How to use
>Google Colab lets you run interactive python scripts (Jupyter Notebooks).
>This demo has been set up to run by simply executing all cells one after another.
>To run each cell press Shift-Enter
>
>If you do not want to train the network skip the *"Training"* section.
>If you wan to train the network from scratch, skip the cell that loads the pre-trained model.
>
>[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tpet93/GIS_Neural_Network/blob/master/Lidar_Classifer_demo.ipynb)
>## Data Prep
>
> ### Raster generation
>
>Lidar data is converted into multiple raster layers. We used LasTools. Specifically LasCanopy.
>The Lidar point cloud (pcl) was flattened to move all ground points to 0m.
>The resulting flattened pcl is split into slices at differing heights above ground. rasters where generated using relative point densisty for ground layer ( # of points in slice/ # of total points)  and normalized point density(# of points in slice/ # of points in and below slice) with a grid of 4m.
>for the canopy and upper vegetation structure, percentile heights where used with a 1m grid.
>
>The resulting rasters should be merged into a single Geotiff (future updates may allow for independent files)
>
>
> ### Training / Test Data
>![Input Tile](https://github.com/tpet93/GIS_Neural_Network/raw/master/Images/InputTile.png) ![Training Tile](https://github.com/tpet93/GIS_Neural_Network/raw/master/Images/TrainingTile.png) 
>
>Training / Test data is created by drawing polygons using GIS software.
>Each polygon is labeled with a class (integer)
>two attributes "Training_Class" and "Test_Class" are used to split polygons between training and test.
>
>
> ### Google Drive
>It is recommended to mount and store datasets in a google drive folder, this will allow automatic saving of snapshots and output data to a persistent storage location.



