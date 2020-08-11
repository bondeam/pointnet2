# config.sh
#!/usr/bin/env bash

TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

nvcc_bin=/usr/local/cuda-10.1/bin/nvcc

cuda_include_dir=/usr/local/cuda-10.1/include
tensorflow_include_dir=$TF_INC
tensorflow_external_dir=$TF_INC/external/nsync/public

cuda_library_dir=/usr/local/cuda-10.1/lib64/
tensorflow_library_dir=$TF_LIB
