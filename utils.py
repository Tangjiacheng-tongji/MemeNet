import numpy as np
from tools import transform
from numba import njit

def compute_ed(shapelet1,shapelet2):
    #shapelet1:N1*channels*l*w
    #shapelet2:channels*l*w
    #output:N1
    assert shapelet1.shape[-1]==shapelet2.shape[-1]
    assert shapelet1.shape[-2]==shapelet2.shape[-2]
    diff=shapelet1-shapelet2
    norm=np.sqrt(np.sum(diff**2,axis=(1,2,3)))
    return norm

def ed_channel1(shapelet1,shapelet2):
    #shapelet1:N1*channels*l*w
    #shapelet2:M*channels*l*w
    #diff:N1*M*channels*l*w
    #norm:N1*M
    #norm2:N1*M*channels
    assert shapelet1.shape[-1]==shapelet2.shape[-1]
    assert shapelet1.shape[-2]==shapelet2.shape[-2]
    shapelet1=np.expand_dims(shapelet1,1).repeat(len(shapelet2),1)
    diff=shapelet1-shapelet2
    norm=np.sqrt(np.sum(diff**2,axis=(3,4)))
    return norm
    
def ed_channel2(shapelet1,shapelet2):
    #shapelet1:N*K*channels*l*w
    #shapelet2:M*channels*l*w
    #diff:N*K*M*channels*l*w
    #norm:N*K*M
    #norm2:N*K*M*channels
    assert shapelet1.shape[-1]==shapelet2.shape[-1]
    assert shapelet1.shape[-2]==shapelet2.shape[-2]
    shapelet1=np.expand_dims(shapelet1,2).repeat(len(shapelet2),2)
    diff=shapelet1-shapelet2
    norm=np.sqrt(np.sum(diff**2,axis=(4,5)))
    return norm

def shapelet2maps(fields,shapelets):
    #shapelets:a collection of shapelets(M*channels*l*w)
    #receptive field:list of N feature with all extracted receptive field
    #list of N lists, where a collection of shapelets is included
    #output:(array)N*M*channels
    if fields.ndim!=1:
        dist=np.min(ed_channel2(fields,shapelets),axis=1)
    else:
        dist=np.array([np.min(ed_channel1(i,shapelets),axis=0) for i in fields])
    return dist

def compute_percentile(candidates,n,k):
    #Set a threshold to distinguish similar memes
    shapelet1=candidates[np.random.choice(len(candidates),n,replace=True)]
    shapelet2=candidates[np.random.choice(len(candidates),n,replace=True)]
    dist=compute_ed(shapelet1,shapelet2)
    del shapelet1
    del shapelet2
    return np.percentile(dist,k)

def compute_percentile_channels(candidates,n,k,channels):
    #Set thresholds for specified channels
    threshold_dict=dict()
    shapelet1=candidates[np.random.choice(len(candidates),n,replace=True)]
    shapelet2=candidates[np.random.choice(len(candidates),n,replace=True)]
    dist = np.sqrt(np.sum((shapelet1-shapelet2)**2,axis=(-1, -2))).T
    for choice in channels:
        threshold_dict[transform(choice)]=np.percentile(dist[choice], k)
    del shapelet1
    del shapelet2
    return threshold_dict

@njit
def get_entropy(used_label):
    init_ent=np.zeros((np.max(used_label)+1))
    for i in np.unique(used_label):
        init_ent[i]=compute_entropy(np.where(used_label==i,1,0),len(used_label))
    return init_ent

@njit
def compute_entropy(data,length):
    assert data.ndim==1
    if length==0:
        return 0
    else:
        if data.ndim==1:
            p = np.sum(data) / length
            if p==0 or p==1:
                return 0
            else:
                return -(p*np.log2(p)+(1-p)*np.log2(1-p))

@njit
def compute_minent(data):
    min_entropy = 999
    length=len(data)
    for i in range(1,length-1):
        left, right = data[:i], data[i:]
        new_entropy = (i * compute_entropy(left,i) + (length-i) * compute_entropy(right, (length-i))) / length
        if min_entropy > new_entropy: min_entropy=new_entropy
    return min_entropy

@njit
def compute_batch_minent(real_idx):
    min_ent=[compute_minent(real_idx[i]) for i in range(len(real_idx))]
    return np.array(min_ent)

@njit
def compute_entropy_parallel(data):
    length = data.shape[1]
    p = np.sum(data,axis=1) / length
    return np.where((p!=0)&(p!=1),-(p*np.log2(p)+(1-p)*np.log2(1-p)),0)

@njit
def compute_minent_parallel(data):
    min_entropy = np.ones(len(data))*999
    target = np.zeros(len(data))
    length=data.shape[1]
    for i in range(1,length-1):
        left, right = data[:,:i], data[:,i:]
        new_entropy = (i * compute_entropy_parallel(left) + (length-i) * compute_entropy_parallel(right)) / length
        min_entropy[min_entropy > new_entropy] = new_entropy[min_entropy > new_entropy]
        target[min_entropy > new_entropy] = i - 1
    return min_entropy, target

@njit
def compute_minent(data):
    min_entropy = np.ones(len(data))*999
    target = np.zeros(len(data))
    length=data.shape[1]
    for i in range(1,length-1):
        for j in range(len(data)):
            left, right = data[j,:i], data[j,i:]
            new_entropy = (i * cal_entropy(left) + (length-i) * cal_entropy(right)) / length
            if min_entropy[j] > new_entropy:
                min_entropy[j]=new_entropy
                target[j]=i-1
    return min_entropy, target

@njit
def get_threshold(datas,idx,target):
    threshold=[]
    for i in range(len(datas)):
        data=datas[i]
        index=int(target[i])
        seq=idx[i]
        threshold.append((data[seq[index]]+data[seq[index+1]])/2)
    return threshold

@njit
def cal_entropy(label):
    p=np.sum(label)/len(label)
    if p==0:
        return 0
    else:
        return -(p*np.log2(p)+(1-p)*np.log2(1-p))

@njit
def compute_term(A,B):
    total = len(A)
    px = np.sum(A) / total
    py = np.sum(B) / total
    pxy = np.sum(A * B) / total
    if px == 0 and py == 0:
        return 1
    elif px == 0 or py == 0 or pxy == 0:
        return 0
    return pxy * np.log2(pxy / (px * py))

@njit
def cal_condition_ent(A,B):
    #compute mutual information
    return compute_term(A, B) + compute_term(1 - A, B) + \
           compute_term(A, 1 - B) + compute_term(1 - A, 1 - B)

@njit
def get_corr(i, meme_ans, dict_ans):
    #In order to avoid high mutual information of low value memes
    #The information gain rate is calculated
    item = meme_ans[i]
    ent = cal_entropy(item)
    sum_info = 0
    list_length = len(dict_ans)
    for j in range(i):
        sum_info += cal_condition_ent(item, meme_ans[j])
    for j in range(len(dict_ans)):
        sum_info += cal_condition_ent(item, dict_ans[j])
    if list_length + i != 0:
        return sum_info / ((list_length + i) * (ent+1e-6))
    else:
        return 0