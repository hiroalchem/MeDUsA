# MeDUsA

This repository is the code for MeDUsA, the software we published in [this paper](https://www.biorxiv.org/content/10.1101/2021.10.25.465674).
For more information on how to use this software, please refer to [this repository](https://github.com/SugieLab/MeDUsA).

## Installation
Windows users can download the exe from [here](https://zenodo.org/record/5548542) and use it.
Linux and Mac users should follow the instructions below.

Please clone or zip download this repository.
Then install the necessary libraries.
Then run main.py.
```angular2
git clone https://github.com/hiroalchem/MeDUsA.git
cd MeDUsA
pip install -r requirements.txt
python main.py
```
You can download pretrained model at zenodo: [https://zenodo.org/record/5533812](https://zenodo.org/record/5533812)

## Installing wxPython
If you encounter any problems regarding the installation of wxPython, please check the [official page](https://wxpython.org/pages/downloads/index.html).

# How to Cite <br>
Yohei Nitta, Hiroki Kawai, Jiro Osaka, Satoko Hakeda-Suzuki, Yoshitaka Nagai, Karolína Doubková, Takashi Suzuki, Gaia Tavosanis, Atsushi Sugie, MeDUsA: A novel system for automated axon quantification to evaluate neuroaxonal degeneration
<a href="https://doi.org/10.1101/2021.10.25.465674">https://doi.org/10.1101/2021.10.25.465674</a>


```
@article {Nitta2021.10.25.465674,
	author = {Nitta, Yohei and Kawai, Hiroki and Osaka, Jiro and Hakeda-Suzuki, Satoko and Nagai, Yoshitaka and Doubkov{\'a}, Karol{\'\i}na and Suzuki, Takashi and Tavosanis, Gaia and Sugie, Atsushi},
	title = {MeDUsA: A novel system for automated axon quantification to evaluate neuroaxonal degeneration},
	elocation-id = {2021.10.25.465674},
	year = {2021},
	doi = {10.1101/2021.10.25.465674},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2021/10/26/2021.10.25.465674},
	eprint = {https://www.biorxiv.org/content/early/2021/10/26/2021.10.25.465674.full.pdf},
	journal = {bioRxiv}
}
```