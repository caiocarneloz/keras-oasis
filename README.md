# koasis
Keras pre-trained deep neural networks for [Open Access Series of Imaging Studies (OASIS)](https://www.oasis-brains.org/) feature extraction.

## Getting Started
#### Dependencies
You need Python 3.7 or later to use **koasis**. You can find it at [python.org](https://www.python.org/).

You aso need numpy, pandas and keras packages, which is available from [PyPI](https://pypi.org). If you have pip, just run:
```
pip install numpy
pip install pandas
pip install keras
```
#### Installation
Clone this repo to your local machine using:
```
git clone https://github.com/caiocarneloz/keras-oasis.git
```

## Features
With this code, its possible to:
- Retrieve all the labels from clinical info (considering CDR)
- Extract features from any OASIS-1 image using Keras Applications
- Generate a dataset containing all non-zero features and corresponding labels


## Usage
Actually, you need to create a folder containing all OASIS images and CDR info.
Having this, it is necessary just call the oasis_extract function sending the folder path:
```
oasis_extract('folder/')
```

## Output
As output, for each pre-trained model, a .csv file containing the features and labels is generated:
```
            0         1         2    3  ...       6223       6224  6225       6226
0    0.751712  0.000000  4.072742  0.0  ...  42.221912   0.000000   0.0    Control
1    2.542512  0.000000  3.418418  0.0  ...  13.266520   5.047302   0.0    Control
2    0.904695  0.000000  3.447591  0.0  ...   7.464177   0.000000   0.0  Alzheimer
3    1.975773  0.000000  5.956251  0.0  ...  41.108479   0.000000   0.0    Control
4    0.000000  0.000000  0.584957  0.0  ...   2.805410   4.802437   0.0    Control
5    1.465851  0.000000  5.688468  0.0  ...  53.284348   0.000000   0.0    Control
6    0.000000  0.000000  6.295202  0.0  ...  20.517311   0.000000   0.0    Control
7    0.565955  0.000000  5.364164  0.0  ...  29.104387   0.000000   0.0    Control
8    2.452629  0.000000  3.623104  0.0  ...   8.846025   5.419124   0.0    Control
9    1.460919  0.000000  2.128028  0.0  ...  43.439747   0.000000   0.0    Control
10   3.632581  0.000000  4.830678  0.0  ...  23.841797  14.104014   0.0    Control
11   3.157856  0.000000  4.083702  0.0  ...  17.051870   6.060349   0.0  Alzheimer
```

## Citation
If you use OASIS dataset, please cite the original publication:

`
MARCUS, Daniel S. et al. Open Access Series of Imaging Studies (OASIS): cross-sectional MRI data in young, middle aged, nondemented, and demented older adults. Journal of cognitive neuroscience, v. 19, n. 9, p. 1498-1507, 2007.
`
