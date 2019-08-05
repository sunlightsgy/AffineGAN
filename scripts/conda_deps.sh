set -ex
conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing imageio
conda install pytorch torchvision -c pytorch # add cudatoolkit=9.0 if CUDA 9
conda install visdom dominate -c conda-forge # install visdom and dominate