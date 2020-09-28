import h5py

root_dir = 'I:\\学术\\CNN-Cell-profile-XRZhang\\result\\20200227\\train_data\\'

def load_matlab_file(fname,fpath=None):
    if fpath is None:
        raw_data = h5py.File(root_dir+fname,'r')
    else:
        raw_data = h5py.File(fpath+fname)
    return raw_data['X'],raw_data['y']

def load_infer_data(fname,fpath=None,data_name='X'):
    if fpath is None:
        raw_data = h5py.File(root_dir+fname,'r')
    else:
        raw_data = h5py.File(fpath+fname,'r')
    return raw_data[data_name]

def save_mat(fname,x):
    data = h5py.File(fname,'w')
    data.create_dataset('y',data=x)
    data.close()