#/bin/bash
source ../config.sh
$nvcc_bin tf_grouping_g.cu -o tf_grouping_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

# TF1.4
#g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-8.0/lib64/ -L/usr/local/lib/python2.7/dist-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

g++ -std=c++11 tf_grouping.cpp tf_grouping_g.cu.o -o tf_grouping_so.so -shared -fPIC \
    -lcudart \
    -I $tensorflow_include_dir -l:libtensorflow_framework.so.2 \
    -I $cuda_include_dir \
    -I $tensorflow_external_dir \
    -L $cuda_library_dir \
    -L $tensorflow_library_dir \
    -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
