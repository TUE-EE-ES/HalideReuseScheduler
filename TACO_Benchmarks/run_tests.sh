#!/bin/bash
rm -f results.txt
touch results.txt
export HL_NUM_THREADS=8
export HPARAMS=8,8388608,40
echo "BILATERAL"
echo "bilateral_grid:" >> results.txt
cd bilateral_grid
make clean
make  test | tail -2  >> ../results.txt
cd ..

echo "CAMERA"
echo "camera_pipe:" >> results.txt
cd camera_pipe
make clean
make test | tail -2  >> ../results.txt
cd ..

echo "HARRIS"
echo "harris:" >> results.txt
cd harris
make clean
make test | tail -2  >> ../results.txt
cd ..

echo "INTERPOLATE"
echo "interpolate:" >> results.txt
cd interpolate
make clean
make test | tail -2  >> ../results.txt
cd ..

echo "LAPLACIAN"
echo "local_laplacian:" >> results.txt
cd local_laplacian
make clean
make test | tail -2  >> ../results.txt
cd ..

echo "MAXFILTER"
echo "max_filter:" >> results.txt
cd max_filter
make clean
make test | tail -2  >> ../results.txt
cd ..

echo "UNSHARP"
echo "unsharp:" >> results.txt
cd unsharp
make clean
make test | tail -2  >> ../results.txt
cd ..

echo "NLMEANS"
echo "nlmeans:" >> results.txt
cd nl_means
make clean
make test | tail -2  >> ../results.txt
cd ..

echo "stencil"
echo "stencil:" >> results.txt
cd stencil_chain
make clean
make test | tail -2  >> ../results.txt
cd ..

echo "LENSBLUR"
echo "lens_blur:" >> results.txt
cd lens_blur
make clean
make test | tail -2 >> ../results.txt
cd ..
#echo "VDSR"
#echo "VDSR:" >> results.txt
#cd VDSR
#make clean
#make HL_TARGET=host-cuda-cuda_capability_35 test #| tail -2  >> ../results.txt
#cd ..


