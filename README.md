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

## build pytorch_sparse 

Using the environment setup for building libtorch, run the following commands to
install the pytorch_sparse C++ library:

```
export Torch_DIR=/path/to/pytorch/install
git clone git@github.com:rusty1s/pytorch_sparse.git
mkdir build-pytorch-sparse
cd $_
cmake ../pytorch_sparse/ -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j8
make install
```

## build pytorch_scatter

Using the environment setup for building libtorch, run the following commands to
install the pytorch_scatter C++ library:

```
export Torch_DIR=/path/to/pytorch/install
git clone git@github.com:rusty1s/pytorch_scatter.git
mkdir build-pytorch-scatter
cd $_
cmake ../pytorch_scatter/ -DCMAKE_INSTALL_PREFIX=$PWD/install
make -j8
make install
```

## build kokkos pytorch example

```
git clone git@github.com:cwsmith/kokkosPytorch.git
mkdir buildKkPytorch
cd $_
export CMAKE_PREFIX_PATH=\
/path/to/pytorch/install/dir:\
/path/to/pytorch_sparse/install:\
/path/to/pytorch_scatter/install
module load gcc cmake
cmake ../kokkosPytorch
make -j8
./hello-world ../kokkosPytorch/gmodel.pt
```

