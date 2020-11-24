# kokkosPytorch
kokkos + pytorch



## create conda env

```
module use /opt/scorec/spack/v0154/lmod2/linux-rhel7-x86_64/Core
module load anaconda3
conda create --prefix ./env
conda init bash
# take the lines out of ~/.bashrc and put them into `setupConda.sh`
source setupConda.sh
conda activate ./env
conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
typing_extensions future six requests dataclasses
```

## build libtorch

```
module use /opt/scorec/spack/v0154/lmod2/linux-rhel7-x86_64/Core
module load anaconda3
source /path/to/setupConda.sh
module load gcc cmake # we want gcc 7.3
```

then follow the instructions here: 

https://github.com/pytorch/pytorch/blob/master/docs/libtorch.rst#building-libtorch-using-cmake

Note, this enables AVX512, MKL, and some other features our machines (or yours) don't have.  These can be disabled in the cmake command.

## build kokkos pytorch example

```
git clone git@github.com:cwsmith/kokkosPytorch.git
mkdir buildKkPytorch
cd $_
export CMAKE_PREFIX_PATH=/path/to/pytorch/install/dir
module load gcc cmake
cmake ../kokkosPytorch
make -j8
./hello-world ../kokkosPytorch/model.pt
```

