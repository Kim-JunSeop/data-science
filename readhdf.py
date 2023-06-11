import torch
import h5py

f = h5py.File('point.hdf5', 'r')
dset = f['dset1'][:]
points2 = torch.from_numpy(dset)
f.close()

points2