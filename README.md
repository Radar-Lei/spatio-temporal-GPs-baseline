# Spatio-Temporal Variational GPs baseline
![Implementation](/fig/st_vgp.gif)

This repository was forked from [Spatio-Temporal Variational GPs](https://github.com/AaltoML/spatio-temporal-GPs) and modified to work as one of the baselines for our study (Reference number: T-ITS-24-06-1995).

First, you should download the data via [Google Drive](https://drive.google.com/drive/folders/14VPjNlQQRd5FCXXHrBPYbety9fiWx7--?usp=drive_link), and put each data set folder under the folder datasets/.

Then, you can test ST-VGP on these data sets with varying missing ratio using the following:

For the PeMS7_I data set,
```bash
python main.py --dataset 'PeMS7_I' --missing_ratio 0.1
python main.py --dataset 'PeMS7_I' --missing_ratio 0.3
python main.py --dataset 'PeMS7_I' --missing_ratio 0.6
```

For the PeMS7_II data set,
```bash
python main.py --dataset 'PeMS7_II' --missing_ratio 0.1
python main.py --dataset 'PeMS7_II' --missing_ratio 0.3
python main.py --dataset 'PeMS7_II' --missing_ratio 0.6
```

For the Seattle data set,
```bash
python main.py --dataset 'Seattle' --missing_ratio 0.1
python main.py --dataset 'Seattle' --missing_ratio 0.3
python main.py --dataset 'Seattle' --missing_ratio 0.6
```




# The following is the original readme file
# Spatio-Temporal Variational GPs

This repository is the official implementation of the methods in the publication:

* O. Hamelijnck, W.J. Wilkinson, N.A. Loppi, A. Solin, and T. Damoulas (2021). **Spatio-temporal variational Gaussian processes**. In *Neural Information Processing Systems (NeurIPS)*. [[arXiv]](https://arxiv.org/abs/2111.01732)

## Citing this work:
```bibtex
@inproceedings{hamelijnck2021spatio,
	title={Spatio-Temporal Variational {G}aussian Processes},
	author={Hamelijnck, Oliver and Wilkinson, William and Loppi, Niki and Solin, Arno and Damoulas, Theodoros},
	booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
	year={2021},
}
```

## Experiment Setup

This has been tested on a Macbook Pro. All spatio-temporal VGP models have been implemented within the [Bayes-Newton package](https://github.com/AaltoML/BayesNewton). 

### Environment Setup

We recommend using conda:

```bash
conda create -n spatio_gp python=3.7
conda activate spatio_gp
```

Then install the required python packages:

```bash
pip install -r requirements.txt
```

### Data Download

#### Pre-processed Data

All data, preprocessed and split into train-test splits used in the paper is provided at https://doi.org/10.5281/zenodo.4531304. Download the folder and place the corresponding datasets into `experiments/*/data` folders.

#### Manual Data Setup

We also provide scripts to generate the data manually:

```bash
make data
```

which will download the relevant London air quality and NYC data, clean them, and split into train-test splits.

### Running Experiments

To run all experiments across all training folds run:

```bash
make experiments
```

To run an individual experiment refer to the `Makefile`.

#### Baselines used

- `GPFlow2` : https://github.com/GPflow/GPflow
- `GPYTorch`: https://github.com/cornellius-gp/gpytorch

## License

This software is provided under the [MIT license](LICENSE).
