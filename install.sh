pip install -r requirements.txt

workdir=$(pwd)
nprocs=$(nproc)
# install metric packages
echo "Installing metric packages"
cd metrics
rm -rf */build/
cd chamfer3D
python setup.py build -j $nprocs
python setup.py install
cd ..
cd emd_assignment
python setup.py build -j $nprocs
python setup.py install
cd ..
cd PyTorchEMD
python setup.py build -j $nprocs
python setup.py install
cd $workdir

echo "Installing third party libraries"
cd third_party/openpoints/cpp
rm -rf */build/
cd pointops
python setup.py build -j $nprocs
python setup.py install
cd ..
cd pointnet2_batch
python setup.py build -j $nprocs
python setup.py install
cd ..
cd emd
python setup.py build -j $nprocs
python setup.py install
cd ..
cd chamfer_dist
python setup.py build -j $nprocs
python setup.py install
