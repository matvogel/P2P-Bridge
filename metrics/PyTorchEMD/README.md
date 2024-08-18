# PyTorch Wrapper for Point-cloud Earth-Mover-Distance (EMD)

## Dependency

The code has been tested on Ubuntu 22.04, PyTorch 2.0.1, CUDA 11.7. 

The original repo does not seem to be maintained any longer, and I wished to have a version that worked well with conda, since I need to manage multiple versions of CUDA on my system regularly. 

## Setup

Get anaconda. I prefer [mamba](https://github.com/mamba-org/mamba). You can use conda, but it is much slower.

As of September 2023, you need to install pytorch with conda along with 
cudatoolkit-dev:

```mamba env create -n my_env python=3.10 pytorch pytorch-cuda=11.7 cudatoolkit-dev gxx=11.4 numpy -c pytorch nvidia conda-forge```

Note that you only need numpy to run the test script.

## Usage
Install this with
```
python setup.py install
```

You can now use it anywhere!

Example:
```
from emd import earth_mover_distance
d = earth_mover_distance(p1, p2, transpose=False)  # p1: B x N1 x 3, p2: B x N2 x 3
```

Run `test_emd_loss.py` to verify your installation.

## Original Credits

The cuda code is originally written by Haoqiang Fan. The PyTorch wrapper is written by Kaichun Mo with help from Jiayuan Gu.

## License

MIT

