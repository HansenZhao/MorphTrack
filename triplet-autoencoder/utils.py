import cv2
import numpy as np
import torch
from collections import OrderedDict,namedtuple
from itertools import product
import time
from torch.utils.data import Dataset
import sys

def pre_process_matlab_images(data,adj_contrast=True,clip_limit=2,grid_size=8):
    x = np.array(data)
    if adj_contrast:
        if x.dtype != 'uint16':
            x = x.astype(np.uint16)
        clahe = cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=(grid_size,grid_size))
        for c in range(x.shape[0]):
            for i in range(x.shape[3]):
                x[c,:,:,i] = clahe.apply(x[c,:,:,i])
    x = x.astype(np.float)
    x_min = np.min(x,axis=(1,2),keepdims=True)
    x_max = np.max(x,axis=(1,2),keepdims=True)
    return (x - x_min)/(x_max - x_min)

def pre_process_cells_images(data,transform=None):
    x = np.array(data).astype(np.float)
    x_max = np.max(x,axis=(1,2),keepdims=True)
    if transform is not None:
        r_mat = cv2.getRotationMatrix2D((x.shape[1]//2,x.shape[2]//2),transform,1.0)
        for i in range(x.shape[3]):
            for j in range(x.shape[0]):
                x[j,:,:,i] = cv2.warpAffine(x[j,:,:,i],r_mat,(x.shape[1],x.shape[2]))
    return x/x_max

def split_np_array(array,ratio=0.9):
    index = np.random.rand(array.shape[0]) <= ratio
    return array[index],array[np.logical_not(index)]

class data_prefetcher():
    def __init__(self,loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True).long()

    def next(self):
        x = self.next_input
        y = self.next_target
        self.preload()
        return x,y

class data_prefetcher_general():
    def __init__(self,loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.preload()

    def preload(self):
        try:
            self.next_sample = next(self.loader)
        except StopIteration:
            self.next_sample = None
            return

        with torch.cuda.stream(self.stream):
            self.next_sample = [x.cuda(non_blocking=True) for x in self.next_sample]

    def next(self):
        # torch.cuda.current_stream().wait_stream(self.stream)
        x = self.next_sample
        self.preload()
        return x

def build_grid_search_run(**params):
    Run = namedtuple('Run',params.keys())
    runs = []
    for v in product(*params.values()):
        runs.append(Run(*v))
    return runs

class RunManager():
    def __init__(self,fname,nRepeat=1):
        self.run_pool = []
        self.cur_run_index = None
        self.run_start_time = None
        self.epoch_count = None
        self.run_data = []
        self.f = open(fname+'.dldata','w+')
        self.nRepeat = nRepeat

    def add_run_list(self,runs):
        self.run_pool += runs
        self.cur_run_index = 0

    def get_cur_run(self):
        return self.run_pool[self.cur_run_index//self.nRepeat]

    def begin_run(self):
        self.epoch_count = 0
        while len(self.run_data) <= self.cur_run_index:
            self.run_data.append([])
        self.run_data[self.cur_run_index] = []
        self.__save_run(self.get_cur_run())
        self.f.flush()
        self.run_start_time = time.time()

    def tick_epoch(self,**params):
        tick_data = OrderedDict()
        tick_data['epoch'] = self.epoch_count
        tick_data['time'] = time.time() - self.run_start_time
        for k in params.keys():
            tick_data[k] = params[k]
        self.run_data[self.cur_run_index].append(tick_data)
        self.epoch_count += 1
        print(self.__record2str(tick_data,inlog=False))
        self.f.write(self.__record2str(tick_data,inlog=True))
        self.f.write('\n')
        self.f.flush()

    def end_run(self):
        self.cur_run_index += 1

    def save_file(self,fname):
        with open(fname+'.dldata','w+') as f:
            for i,run in enumerate(self.run_pool):
                if self.cur_run_index > i:
                    self.__save_run(f,run)
                    self.__save_data(f,self.run_data[i])

    def end_all(self):
        self.f.close()

    def __save_run(self,run):
        run_dict = run._asdict()
        for k in run_dict.keys():
            self.f.write(f'# {k}: {run_dict[k]}')
            self.f.write('\n')

    def __save_data(self,f,data):
        for d in data:
            f.write(self.__record2str(d))
            f.write('\n')

    def __record2str(self,r,inlog=True):
        str = ''
        for k in r.keys():
            if k.endswith('__silent'):
                if not inlog:
                    continue
                out_word = k.replace('__silent','')
            else:
                out_word = k
            if isinstance(r[k],float):
                str += f'{out_word},{r[k]:g},'
            else:
                str += f'{out_word},{r[k]},'
        return str[:-1]

    @property
    def is_on_going(self):
        return self.cur_run_index < (len(self.run_pool)*self.nRepeat)

class EarlyStopper(object):
    def __init__(self,patience=5,min_delta=0,mode=-1):
        self.val = pow(10,-25*mode)
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.wait_time = 0

    def stop_now(self,v):
        if self.mode * v > self.mode * (self.val + self.mode * self.min_delta):
            self.val = v
            self.wait_time = 0
        else:
            self.wait_time += 1
            print(f'best result: {self.val:g}, wait time: {self.wait_time}')
        return self.wait_time > self.patience

class ModelSaver(object):
    def __init__(self,threshold,path,min_delta=0,mode=-1):
        self.val = threshold
        self.min_delta = min_delta
        self.path = path
        self.mode = mode

    def check(self,v,debug=False):
        if self.mode * v > self.mode * (self.val + self.mode * self.min_delta):
            if debug:
                print(f'---+++---\nsave model, value: {v:g}, best {self.val:g}\n-----++++-----')
            self.val = v
            return True
        else:
            return False


    def save(self,v,iter,model,opt):
        torch.save({
            'iteration': iter,
            'watch_value': v,
            'model': model,
            'opt': opt
        }, self.path)

def loadModel(path,model):
    data = torch.load(path)
    print(f'load model saved at iter: {data["iteration"]}, watch value: {data["watch_value"]}')
    model.load_state_dict(data['model'])
    return model

def confuse_matrix(model,x,y,n_label):
    r = np.zeros((n_label,n_label))
    iou = []
    with torch.no_grad():
        yest = model(x).argmax(dim=1)
        for i in range(n_label):
            mat = torch.add((y==i).int(),(yest==i).int())
            insect = torch.sum(mat==2).item()
            union = torch.sum(mat>0).item()
            iou.append(insect/union)
            for j in range(n_label):
                r[i,j] = torch.sum(torch.mul(y==i,yest==j)).item()
    return r,iou

class TriCellDS(Dataset):
    def __init__(self,images_array):
        super().__init__()
        self.images = images_array
        self.image_size = self.images.shape[2]
        self.pair_index = np.random.permutation(self.images.shape[0])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        i = index
        j = self.pair_index[index]
        while(i==j):
            j = np.random.randint(0,len(self),1)[0]
        return (self.__get_image(i),self.__get_image(i,rotation=True),self.__get_image(j,rotation=True))

    def __get_ij(self,index):
        n = self.images.shape[0]
        j = index // (n-1)
        i = index % (n-1)
        if i >= j:
            i += 1
        return i,j

    def __get_image(self,pos,rotation=False):
        im = self.images[pos].copy()
        if rotation:
            r_mat = cv2.getRotationMatrix2D((self.image_size/2,self.image_size/2),np.random.randint(1,360),1)
            for i in range(im.shape[0]):
                im[i,:,:] = cv2.warpAffine(im[i,:,:],r_mat,(self.image_size,self.image_size))
        return im

def waitbar(n,total,**params):
    rate = n / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%,%d/%d' % ("=" * rate_num, " " * (100 - rate_num), rate_num, n, total)

    for k in params.keys():
        r += f' {k}: {params[k]:.4f}'

    sys.stdout.write(r)
    sys.stdout.flush()

def gen9DirWeight(index):
    w = torch.tensor(list(range(9)))
    w = -((w - index) == 0).float()
    w[4] = 1.0
    return  w.reshape((3,3))

def iou_with_logits(logits,label,n_class):
    pred = torch.argmax(logits,dim=1)
    pred = pred.contiguous().view(-1)
    label = label.contiguous().view(-1)
    iou = []
    for i in range(n_class):
        iou.append(
            ((pred==i) & (label==i)).sum().item() / ((pred==i) | (label==i)).sum().item()
        )
    return np.mean(iou),iou







