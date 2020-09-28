import torch
import torch.utils.data
import models
import model_blocks
import data_reader
import time
import utils
import torch.nn.functional as F
import numpy as np
import cv2
from scipy.spatial.distance import cdist,pdist,squareform

dataset_name = 'image.mat'
path = '/dat01/hszhao/data/'
#path = 'J:\\学术\CNN-Cell-profile-XRZhang\\result\\20200429-manuscript-v1\\Figure 2\\data\\'
runs = utils.build_grid_search_run(
    dataset_name = [dataset_name],
    model_width = [32],
    resize_method = [None],
    lr_max = [1e-3],
    lr_min = [1e-6],
    lr_T = [1000],
    batch_size = [256],
    max_epoch = [500000],
    early_stop_patience = [2500],
    code_width = [128],
    model_depth = [3],
    margin = [3.0],
    loss_factor = [500.0],
    model_name = ['pretrain.tar']
)

def get_code(net,loader):
    with torch.no_grad():
        images_code = []
        for data in loader:
            code = net.encoder(data[0].cuda())
            images_code.append(code.cpu().numpy())
    return np.concatenate(images_code)

def get_code_quick(net,X):
    with torch.no_grad():
        images_code = net.encoder(X.cuda())
    return images_code.cpu().numpy()

def transform_images(X,transform):
    # NCHW
    r_mat = cv2.getRotationMatrix2D((X.shape[2] // 2, X.shape[3] // 2), transform, 1.0)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j, :, :] = cv2.warpAffine(X[i, j, :, :], r_mat, (X.shape[2], X.shape[3]))
    return X

def get_topk_accuracy(net,k,data):
    code_0 = get_code(net, torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(torch.tensor(data)),
                    batch_size = run.batch_size*4,
                    drop_last=False
                ))

    code_1 = get_code(net, torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(transform_images(data.copy(),90))),
        batch_size=run.batch_size*4,
        drop_last=False
    ))

    dist = cdist(code_0,code_1)
    I = dist.argsort(axis=1)
    b = I[:,0:k] == np.array(range(code_1.shape[0]),).reshape(code_1.shape[0],1).astype(np.int)
    return np.sum(np.any(b,axis=1))/code_1.shape[0]

def get_topk_accuracy_quick(net,k,data):
    code_0 = get_code_quick(net, torch.tensor(data))

    code_1 = get_code_quick(net, torch.tensor(transform_images(data.copy(),90)))

    dist = cdist(code_0,code_1)
    I = dist.argsort(axis=1)
    b = I[:,0:k] == np.array(range(code_1.shape[0]),).reshape(code_1.shape[0],1).astype(np.int)
    return np.sum(np.any(b,axis=1))/code_1.shape[0]

manager = utils.RunManager('test_result_conti_2')
manager.add_run_list(runs)

print('read data...')
X = data_reader.load_infer_data(dataset_name,path,'data')
X = utils.pre_process_cells_images(X)
X = np.transpose(X,(3,0,1,2)).astype(np.float32) #from CHWN to NCHW
X = np.expand_dims(X[:,2,:,:],axis=1)
print(X.shape)
image_size = X.shape[2]
image_channel = X.shape[1]
print(f'Number of train samples: {X.shape[0]}')

while manager.is_on_going:
    run = manager.get_cur_run()
    train_loader = torch.utils.data.DataLoader(
        utils.TriCellDS(X),
        batch_size = run.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers = 2,
    )

    n_iter = len(train_loader)
    print(f'read done, number of iter:{n_iter:d}')

    print('setup model...')
    # net = models.CellAutoEncoder(model_blocks.UnetEncoder(image_size,image_channel,run.model_depth,run.model_width,run.code_width))
    net = models.RotInvarSiameseNet(model_blocks.UnetEncoder(image_size,image_channel,run.model_depth,run.model_width,run.code_width))
    net.float().cuda()
    opt = torch.optim.Adam(net.parameters(),lr=run.lr_max)
    train_proc_data = torch.load(run.model_name)
    print(f'load model at {train_proc_data["iteration"]} for value {train_proc_data["watch_value"]}')
    net.load_state_dict(train_proc_data['model'])
    opt.load_state_dict(train_proc_data['opt'])

    criterion = model_blocks.SiamInvarLoss(run.margin)

    lr_schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,run.lr_T,eta_min=run.lr_min)
    stopper = utils.EarlyStopper(patience=run.early_stop_patience)
    saver = utils.ModelSaver(0.25,'final.tar',0.001,1)
    print('set done')

    print('begin training...')
    manager.begin_run()
    for epoch in range(run.max_epoch):
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        i = 0
        prefetcher = utils.data_prefetcher_general(train_loader)
        images = prefetcher.next()

        while images is not None:

            logits = net(images)

            loss1 = criterion(logits)
            loss2 = F.mse_loss(logits[2],images[0],reduction='sum') + F.mse_loss(logits[3],images[2],reduction='sum')

            loss = run.loss_factor * loss1 + loss2
            # loss = loss2
            # loss = F.mse_loss(logits,images[0],reduction='sum')

            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()


            lr_schedular.step(epoch+i/n_iter)
            opt.zero_grad()
            loss.backward()
            opt.step()

            images = prefetcher.next()
            i += 1

        if epoch % 100 == 0 or total_loss1 < 0.001:
            train_accuracy = get_topk_accuracy(net,10,X)

        manager.tick_epoch(train_loss=total_loss,
                           loss1 = total_loss1,
                           loss2 = total_loss2,
                           train_accuracy = train_accuracy,
                           # eval_accuracy = eval_accuracy,
                           end_lr = lr_schedular.get_last_lr()[0],
                           watch_time=stopper.wait_time)

        if saver.check(train_accuracy,True):
            saver.save(total_loss,epoch,net.state_dict(),opt.state_dict())

        if stopper.stop_now(total_loss):
            print('Early stopping...')
            break

    manager.end_run()

manager.end_all()
