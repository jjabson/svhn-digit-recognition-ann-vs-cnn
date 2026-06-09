Overview:

The datasets used in this project are not stored directly in this GitHub repository because they exceed GitHub's recommended file size limits and would 
significantly increase repository size. To keep the repository lightweight and reproducible, the data can be downloaded from the links below and placed 
in this directory before running the notebooks.

Dataset

Dataset Name: SVHN (Street View House Numbers)

Description:

The Street View House Numbers (SVHN) dataset is a real-world image dataset containing digits extracted from house numbers in Google Street View images. 
It is commonly used for image classification and deep learning research.

Download link: https://drive.google.com/file/d/1W6JdRAKEdFir7AutJEfRkdBXBCJVyq7b/view?usp=drive_link

Download Dataset Example:

import h5py

with h5py.File("data/SVHN_single_grey1.h5", "r") as f:
    print(list(f.keys()))
