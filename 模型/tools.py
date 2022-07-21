import numpy as np
from skimage import measure
from module import *
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

def get_importance(grad,labels,k=5):
    #Refer to Grad-Cam to get the importance of different channels
    #grad_block:(list)Gradient obtained throught backward hook
    #label:(array)Label corresponding to data
    relation=dict()
    for label in np.unique(labels):
        relation[str(label)]=dict()
        weight=np.mean(grad[np.where(labels==label)],axis=(0,2,3))
        a=np.argpartition(-np.abs(weight),range(k))[range(k)]
        relation[str(label)]["strong"]=[[item] for item in a]
        a=np.argpartition(np.abs(weight),range(k))[range(k)]
        relation[str(label)]["weak"]=[[item] for item in a]
    return relation

def acc_label(net,test_loader, use_gpu = False):
    ans = {}
    total = {}
    with torch.no_grad():
        for images, labels in test_loader:
            label_list = torch.unique(labels)
            if use_gpu:
                images, labels = images.cuda(), labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            result = (predicted == labels)
            for label in label_list:
                label = label.item()
                other = (labels == label)
                if label not in ans:
                    ans[label] = 0
                    total[label] = 0
                total[label] += other.sum().item()
                ans[label] += torch.logical_and(other, result).sum().item()
        accuracy = {}
        for item in ans:
            accuracy[item] = 100 * ans[item] / total[item]
        return accuracy

def make_layers(kernels, func, zscore_module={}):
    # kernels:dict
    assert isinstance(zscore_module, dict)
    layers=dict()
    for where in kernels:
        meme_pool = kernels[where]
        layers[where] = dict()
        for size,memes in meme_pool.items():
            layers[where][size]=dict()
            meme_kernel=torch.from_numpy(np.array(memes["kernel"],dtype=np.float32))
            meme_channel=torch.Tensor([item[1] for item in memes["index"]]).long()
            if func == memes_ED_mask:
                layers[where][size] = func(meme_kernel, meme_channel)
            elif func == zscore_ED_mask:
                layers[where][size] = func(meme_kernel, meme_channel, zscore_module[where])
            elif func == zscore_sim_mask:
                threshold = meme_pool[size]['thresholds']
                std = Standardization(torch.Tensor(threshold))
                layers[where][size] = func(meme_kernel, meme_channel, zscore_module[where], std)
            else:
                layers[where][size] = func(meme_kernel)
    if func == zscore_sim_mask:
        mode = 'sim'
    else:
        mode = 'dist'
    return layers, mode

def numpy2loader(data,label,batch_size=64,shuffle=True):
    data_tensor=torch.from_numpy(data)
    label_tensor=torch.from_numpy(label)
    dataset = TensorDataset(data_tensor,label_tensor)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def get_parameters(model,classifier_keywords=["fc","classifier"]):
    #keywords:list
    #names/keywords for classifier
    classifiers_params = []
    for name,param in model.named_parameters():
        for keyword in classifier_keywords:
            if keyword in name:
                classifiers_params += [param]
                break
    all_params = model.parameters()
    base_params = list(filter(lambda param: id(param) not in list(map(id,classifiers_params)),all_params))
    return base_params,classifiers_params

def get_remains(model,param_list):
    #param_list:list of useful params
    exist_params=[]
    for param in param_list:
        exist_params+=list(map(id,param))
    all_params = model.parameters()
    return list(filter(lambda param: id(param) not in exist_params, all_params))

def get_optional_layers(layers, keywords=["fc","classifier"]):
    optional_layers={}
    break2=False
    for name,layer in layers.items():
        for keyword in keywords:
            if keyword in name:
                break2=True
                break
        if break2:break
        optional_layers[name]=layer
    return optional_layers

def backward(input,label,model,optimizer,loss_func):
    optimizer.zero_grad()
    output = model(input)
    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()
    return loss

def compute_min_distance(kernels,channel,feature_map,
            use_gpu = True, batch_size = 32):
    kernels = torch.from_numpy(kernels)
    channel = torch.Tensor(channel).long()
    feature_map = torch.from_numpy(feature_map)

    zeros = torch.zeros_like(kernels, dtype=torch.float)
    mask = torch.stack([zeros[i].index_fill_(0,channel[i],1) for i in range(len(zeros))])
    weight = kernels * mask

    if use_gpu:
        feature_map = feature_map.cuda()

    '''
    x2 = torch.nn.functional.conv2d(input=feature_map ** 2, weight=mask, stride=1, padding=0)
    x_m = torch.nn.functional.conv2d(input=feature_map, weight=weight, stride=1, padding=0)
    m2 = torch.sum(torch.sum(weight ** 2, dim=(2, 3), keepdim=True), dim=1)
    dist = torch.sqrt(torch.nn.functional.relu(x2 - 2 * x_m + m2))
    '''
    with torch.no_grad():
        dists = []
        for i in range(0, len(weight), batch_size):
            used_weight = weight[i:i+batch_size]
            used_mask = mask[i:i+batch_size]
            if use_gpu:
                used_weight = used_weight.cuda()
                used_mask = used_mask.cuda()
            x2=torch.nn.functional.conv2d(input=feature_map**2,weight=used_mask,stride=1,padding=0)
            x_m = torch.nn.functional.conv2d(input=feature_map, weight=used_weight,stride=1,padding=0)
            m2 = torch.sum(torch.sum(used_weight ** 2,dim=(2,3),keepdim=True),dim=1)
            dists.append(torch.sqrt(torch.nn.functional.relu(x2 - 2*x_m + m2)).cpu())
        dist = torch.cat(dists, dim = 1)
    return torch.min(torch.min(dist, 2).values, 2).values

def find_nearest_crop_thres(dist_map,thres=None,percentile=5,mode='dist'):
    if mode=='dist':
        threshold = np.percentile(dist_map, percentile)
        if thres is None:
            target = dist_map < threshold
        else:
            target = dist_map < min(threshold,thres)
    else:
        threshold = np.percentile(dist_map,100-percentile)
        if thres is None:
            target = dist_map > threshold
        else:
            target = dist_map > max(threshold, thres)
    idx=np.argwhere(target)
    amount = np.sum(dist_map > thres)
    if len(idx)==0:
        return (0,0),(0,0),0
    left_upper=(np.min(idx[:,0]),np.min(idx[:,1]))
    right_lower=(np.max(idx[:,0]),np.max(idx[:,1]))
    return left_upper,right_lower,amount

def crop_bbox(dist_map, thres, times = 1.5):
    binary_map = dist_map > thres
    labeled_img = measure.label(binary_map, connectivity=1, background=0)
    regions = []
    for i,region in enumerate(measure.regionprops(labeled_img)):
        max_value = np.max(dist_map[labeled_img == i + 1])
        if max_value >= thres * times:
            lu = (region.bbox[0],region.bbox[1])
            rl = (region.bbox[2], region.bbox[3])
            regions.append([lu, rl, region.area])
    return regions

class Standardization(nn.Module):
    def __init__(self, threshold, res = 0.2):
        super(Standardization, self).__init__()
        if isinstance(threshold, torch.Tensor):
            self.mode = 1
        else:
            self.mode = 0
        self.thres = threshold
        self.a = 10
        self.c = 1
        if self.mode:
            mask = torch.ones_like(threshold)
            self.res = res * mask
            ans = torch.pow(10, self.res)
            self.b = nn.Parameter(torch.where(self.thres == 0, mask, (self.a - ans * self.c) /
                                              (self.thres * ans - self.thres + 1e-8)))
        else:
            self.res = res
            ans = np.power(10, self.res)
            self.b = np.where(self.thres == 0, 1, (self.a - ans * self.c) /
                                           (self.thres * ans - self.thres + 1e-8))
    def __call__(self, data):
        if self.mode:
            return torch.log10((self.b.view(1,-1,1,1) * data + self.a) / (self.b.view(1,-1,1,1) * data + self.c))
        else:
            return np.log10((self.b * data + self.a) / (self.b * data + self.c))
