import heapq
import json
import os
import cv2
import numpy as np
import utils
import extract
from utils import get_entropy
from tools import MemeEncoder, update_info, \
    get_regions, threshold_classify, memes2kernels
from search import recount,get_results,get_result,get_candidates_in_class

class Meme:
    def __init__(self, meme, label, ig, threshold, indices, channel, where=-1):
        # meme: array-(channels*length*width)
        # label: int
        # ig: list
        # threshold: list
        # indices: tuple-(a,b,c)
        # a:indice of img,(b,c):actucal direction
        # channel: int
        # specific channel
        # where: (int or str) - optional
        self.meme = meme
        assert isinstance(label,int)
        self.label = label
        if isinstance(ig,list):
            self.information_gain = ig
        else:
            self.information_gain = [ig]

        if isinstance(threshold,list):
            self.threshold = threshold
        else:
            self.threshold = [threshold]

        self.indices = indices

        assert isinstance(where, str) or isinstance(where, int)
        self.where = where
        assert isinstance(channel, int)
        self.channel = channel

    def append(self, ig, threshold):
        # ig: float
        # threshold: float
        self.information_gain.append(ig)
        self.threshold.append(threshold)

    def visualize(self, introduce=False):
        # used to introduce meme
        if introduce:
            print("Meme for label", str(self.label), ":")
        print("\tInformation gain:", np.mean(self.information_gain), "Size:", self.meme.shape[1], "×", self.meme.shape[2])
        print("\tOn channel", self.channel, ".")
        if isinstance(self.where, str):
            print("\tFrom", self.where)
        else:
            print("\tFrom layer", self.where)

    def __lt__(self, other):
        return np.mean(self.information_gain) < np.mean(other.information_gain)

class MemePools:
    def __init__(self, labels=[], where=[], capacity=2):
        # meme_dict: dict of memes for current model
        # key: label(str), value: label_dict(dict)
        # label_dict: dict of memes for specific label
        # key: layer(int), value: layer_dict(dict)
        # key: channels(int), value: memes(list)
        # e.g. : meme_dict={label:{channel1:[meme1]}}
        # labels: np.ndarray or list
        # capacity:int
        self.meme_dict = dict()
        for label in labels:
            self.meme_dict[label] = dict()
            for layer in where:
                self.meme_dict[label][layer] = dict()
        self.capacity = capacity

    def __len__(self):
        length = 0
        for label_pool in self.meme_dict.values():
            for layer_pool in label_pool.values():
                for channel_pool in layer_pool.values():
                    length+=len(channel_pool)
        return length

    def get_label(self):
        return list(self.meme_dict.keys())

    def census(self, func=print):
        # The meme of each label is counted separately.
        func("Start counting the memes")
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            func("For label " + str(label))
            for layer in label_pool:
                layer_pool = label_pool[layer]
                func("Memes on layer " + layer)
                output = ""
                for channel in layer_pool:
                    output += str(channel) + ":" + str(len(layer_pool[channel])) + "\t"
                func(output)

    def get_channels(self):
        channels = dict()
        for label_pool in self.meme_dict.values():
            for layer in label_pool:
                layer_pool = label_pool[layer]
                if layer not in channels:
                    channels[layer] = []
                for channel in layer_pool:
                    channels[layer].append(channel)
        return channels

    def sort(self):
        # Ranking memes for each channel according to average information gain
        for label_pool in self.meme_dict.values():
            for layer_pool in label_pool.values():
                for channel_pool in layer_pool.values():
                    channel_pool.sort(reverse=True)

    def eliminate(self,labels=None):
        # delete all memes of sellected labels
        if not labels:
            for label in self.meme_dict.keys():
                self.meme_dict[label]=dict()
        else:
            for label in labels:
                if label in self.meme_dict:
                    self.meme_dict[label] = dict()

    def update_pools(self, labels = {}, capacity = 2):
        # labels:dict
        # keys: different label, values(int): layers
        for label in labels:
            layers = labels[label]
            if label not in self.meme_dict:
                self.meme_dict[label] = dict()
            if isinstance(layers,list):
                for layer in layers:
                    if layer not in self.meme_dict[label]:
                        self.meme_dict[label][layer] = dict()
            else:
                if layers not in self.meme_dict[label]:
                    self.meme_dict[label][layers] = dict()
        if self.capacity > capacity:
            self.sort()
            for label_pool in self.meme_dict.values():
                for layer_pool in label_pool.values():
                    for channel_pool in layer_pool.values():
                        channel_pool = channel_pool[:capacity]
        self.capacity = capacity

    def update(self, candidate, label, ig, threshold, direction_info, channel, where=-1):
        # candidate: array(The same form as class Meme)
        # label: list(The same form as class Meme)
        # ig: float
        # best information gain calculated by algorithm
        # threshold: float
        # direction_info, channel, where:(The same form as class Meme)
        if label not in self.meme_dict or where not in self.meme_dict[label]:
            self.update_pools({label:where}, self.capacity)
        if channel not in self.meme_dict[label][where]:
            self.meme_dict[label][where][channel] = []
        meme = Meme(candidate, label, ig, threshold, direction_info, channel, where)
        if len(self.meme_dict[label][where][channel]) < self.capacity:
            heapq.heappush(self.meme_dict[label][where][channel], meme)
        else:
            heapq.heappushpop(self.meme_dict[label][where][channel], meme)

    def update_meme(self, meme):
        # meme: Meme
        channel_key = meme.channel
        if meme.label not in self.meme_dict or meme.where not in self.meme_dict[meme.label]:
            self.update_pools({meme.label:meme.where}, self.capacity)
        if channel_key not in self.meme_dict[meme.label][meme.where]:
            self.meme_dict[meme.label][meme.where][channel_key] = list()
        if len(self.meme_dict[meme.label][meme.where][channel_key]) < self.capacity:
            # More memes need to be included
            heapq.heappush(self.meme_dict[meme.label][meme.where][channel_key], meme)
        else:
            heapq.heappushpop(self.meme_dict[meme.label][meme.where][channel_key], meme)

    def output(self):
        print("Print the memes recorded.")
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            if len(label_pool) == 0:
                print("Memes for label", str(label), "is empty!")
            else:
                print("Memes for label", str(label), ":")
                for layer in label_pool:
                    layer_pool = label_pool[layer]
                    if len(label_pool) == 0:
                        print(" "*4 + "Memes on layer " + str(layer) + " is empty!")
                    else:
                        print(" "*4 + "Memes on layer " + str(layer) + " :")
                        for channel in layer_pool:
                            channel_pool = layer_pool[channel]
                            for i in range(len(channel_pool)):
                                print(" "*8 + "Meme " + str(i + 1) + " on channel " + str(channel) + ":")
                                channel_pool[i].visualize()

    def get_max_ig(self, visualize = True, func = print):
        # func: function
        # mode: working mode
        ig_dict = {}
        for label in sorted(self.meme_dict):
            label_pool = self.meme_dict[label]
            ig_dict[label] = 0
            if len(label_pool) == 0:
                if visualize:
                    output = "The pool for label " + str(label) + " is empty."
                    func(output)
            else:
                ig = []
                for layer in label_pool:
                    layer_pool = label_pool[layer]
                    for channel in layer_pool:
                        channel_pool = layer_pool[channel]
                        ig.append(np.mean(max(channel_pool).information_gain))
                if len(ig) > 0:
                    ig_dict[label] = max(ig)
                if visualize:
                    if len(ig) > 0:
                        output = str(label) + ":" + str(max(ig))
                        func(output)
        return ig_dict

    def max_ig(self, func=print):
        # func: function
        # Mainly focus on the average information gain.
        self.sort()
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            if len(label_pool) == 0:
                output = "The pool for label " + str(label) + " is empty."
                func(output)
            else:
                output = "Memes for label " + str(label) + ":"
                func(output)
                for layer in label_pool:
                    layer_pool = label_pool[layer]
                    if len(layer_pool) == 0:
                        output = "Memes on layer " + str(layer) + " is empty!"
                        func(output)
                    else:
                        output = "Memes on layer " + str(layer) + ":"
                        func(output)
                        for channel in layer_pool:
                            channel_pool = layer_pool[channel]
                            output = "Maximum average information gain for label " + str(label) + " on channel " + \
                                     str(channel) + " is " + str(np.mean(max(channel_pool).information_gain))
                            func(output)

    def retrieve(self, save_path="memes"):
        key_data = {}
        history = []
        ig_dict = self.get_max_ig(visualize = False)
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            for layer in label_pool:
                if layer not in key_data:
                    key_data[layer] = {"idx":[],"start":[],"shape":[],\
                                       "channel":[],"intensity":[],"name":[]}
                layer_pool = label_pool[layer]
                for channel in layer_pool:
                    channel_pool = layer_pool[channel]
                    count = 1
                    for item in channel_pool:
                        idx = item.indices[0]
                        if idx < 0:
                            filename = str(label) + '_' + str(layer) + '_' \
                                        + str(channel) + '_' + str(-idx) + '.png'
                            history.append(filename)
                        else:
                            filename = str(label) + '_' + str(layer) + '_' + str(channel)
                            fullname = filename + '_' + str(count) + '.png'
                            savename = os.path.join(save_path, fullname)
                            while os.path.exists(savename) or savename in key_data[layer]["name"]:
                                count += 1
                                fullname = filename + '_' + str(count) + '.png'
                                savename = os.path.join(save_path, fullname)
                            item.indices[0] = -count
                            key_data[layer]["idx"].append(idx)
                            key_data[layer]["start"].append(item.indices[1:])
                            key_data[layer]["shape"].append(item.meme.shape[1:])
                            key_data[layer]["channel"].append(channel)
                            key_data[layer]["name"].append(savename)
                            key_data[layer]["intensity"].append(np.mean(item.information_gain) \
                                                                / (ig_dict[label] + 1e-5))
        return key_data, history

    def rename_file(self, save_path="memes", depository = True, func = print):
        self.rename_data(save_path, depository, space = self.capacity*2)
        self.rename_data(save_path, depository)
        output = "Images renames."
        func(output)

    def rename_data(self, save_path="memes", depository = True, space = 0):
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            for layer in label_pool:
                layer_pool = label_pool[layer]
                for channel in layer_pool:
                    channel_pool = layer_pool[channel]
                    for count in range(len(channel_pool)):
                        item = channel_pool[count]
                        old_idx = -item.indices[0]
                        new_idx = space + count + 1
                        filename = str(label) + '_' + str(layer) + '_' + str(channel)
                        old_path = os.path.join(save_path, filename + '_' +str(old_idx) + '.png')
                        new_path = os.path.join(save_path, filename + '_' +str(new_idx) + '.png')
                        os.rename(old_path, new_path)
                        if depository:
                            store_path = "depository"
                            full_name = filename + '_' + str(old_idx) + '.png'
                            new_name = filename + '_' + str(new_idx) + '.png'
                            old_path = os.path.join(store_path, full_name)
                            if os.path.exists(old_path):
                                new_path = os.path.join(store_path, new_name)
                            else:
                                old_path = os.path.join(store_path, "L_" + full_name)
                                if os.path.exists(old_path):
                                    new_path = os.path.join(store_path, "L_" + new_name)
                            os.rename(old_path, new_path)
                        item.indices[0] = -new_idx

    def save_compress(self, filename, round_up = 8):
        with open(filename, 'w') as f:
            for label_pool in self.meme_dict.values():
                for layer_pool in label_pool.values():
                    for channel_pool in layer_pool.values():
                        for item in channel_pool:
                            data = dict()
                            data['scale'] = len(item.meme)
                            compress_meme = item.meme[item.channel]
                            for name in vars(item):
                                value = vars(item)[name]
                                if name== "meme":
                                    data["meme"] = compress_meme
                                elif name == "threshold" or name == "information_gain":
                                    data[name] = np.mean(value)
                                else:
                                    data[name] = value
                            data["lifespan"] = len(item.threshold)
                            json.dump(data, f, cls=MemeEncoder)
                            f.write('\n')
        print("Compressed memes saved as "+filename)

    def save(self, filename):
        with open(filename, 'w') as f:
            for label_pool in self.meme_dict.values():
                for layer_pool in label_pool.values():
                    for channel_pool in layer_pool.values():
                        for item in channel_pool:
                            data = dict()
                            for name in vars(item):
                                value = vars(item)[name]
                                if name == "threshold" or name == "information_gain":
                                    data[name] = np.mean(value)
                                else:
                                    data[name] = value
                            data["lifespan"] = len(item.threshold)
                            json.dump(data, f, cls=MemeEncoder)
                            f.write('\n')
        print("Memes saved as " + filename)

    def load(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                label = int(data["label"])
                meme = np.array(data["meme"])
                if "where" in data:
                    where = data["where"]
                else:
                    where = -1
                if label not in self.meme_dict or where not in self.meme_dict[label]:
                    self.update_pools({label:where}, self.capacity)
                channel = data["channel"]
                if "threshold" in data:
                    threshold = float(data["threshold"])
                else:
                    threshold = 0
                if "scale" in data:
                    length, width = meme.shape[-2], meme.shape[-1]
                    mask = np.zeros((data["scale"], length, width))
                    mask[channel]=meme
                else:
                    mask = meme
                if "lifespan" in data:
                    length = data["lifespan"]
                else:
                    length = 1
                self.update(mask, label, [data["information_gain"]]*length,\
                            [threshold]*length, np.array(data["indices"]), channel, where)

    def get_kernels(self):
        # key: where(str or int), value: kernel_dict(dict)
        # kernel_dict: key: size(str), value: kernel(array)
        kernels = dict()
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            for layer in label_pool:
                layer_pool = label_pool[layer]
                for channel_pool in layer_pool.values():
                    for count in range(len(channel_pool)):
                        item = channel_pool[count]
                        size = (item.meme.shape[1],item.meme.shape[2])
                        where = str(item.where)
                        if where not in kernels:
                            kernels[where] = dict()
                        if size not in kernels[where]:
                            kernels[where][size] = {"kernel": [], "thresholds":[], "index":[]}
                        kernels[where][size]["kernel"].append(item.meme)
                        kernels[where][size]["thresholds"].append(np.mean(item.threshold))
                        kernels[where][size]["index"].append([label,item.channel,layer,count])
        return kernels

    def get_representation(self):
        kernels = dict()
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            for layer in label_pool:
                layer_pool = label_pool[layer]
                for channel_pool in layer_pool.values():
                    for item in channel_pool:
                        size = (item.meme.shape[1],item.meme.shape[2])
                        where = str(item.where)
                        if where not in kernels:
                            kernels[where] = dict()
                        if label not in kernels[where]:
                            kernels[where][label] = dict()
                        if size not in kernels[where][label]:
                            kernels[where][label][size] = {"kernel": [], "channel":[]}
                        kernels[where][label][size]["kernel"].append(item.meme)
                        kernels[where][label][size]["channel"].append(item.channel)
        return kernels

    def parallel_get(self, result, where=-1):
        targets, labels, igs, thresholds, direction_infos, channels = result
        memes=[]
        length = len(targets)
        assert len(labels) == length
        assert len(igs) == length
        assert len(direction_infos) == length
        assert len(channels) == length
        assert len(thresholds) == length
        for i in range(length):
            memes.append(Meme(targets[i], int(labels[i]), igs[i], thresholds[i],
                              direction_infos[i], channel = int(channels[i]), where = where))
        return memes

    def parallel_update(self, result, where=-1):
        memes = self.parallel_get(result, where)
        for meme in memes:
            self.update_meme(meme)

    def get_important_choices(self):
        sort_dict = dict()
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            sort_dict[label] = dict()
            for layer in label_pool:
                layer_pool = label_pool[layer]
                avg_dict = dict()
                for channel in layer_pool:
                    channel_pool = layer_pool[channel]
                    avg_dict[channel] = sum(np.mean(meme.information_gain) for meme in channel_pool) / len(channel_pool)
                sort_result = sorted(avg_dict.items(), key=lambda x: x[1], reverse=True)
                sort_key = [sort_result[i][0] for i in range(0, len(sort_result))]
                sort_dict[label][layer] = sort_key
                '''
                sort_key = []
                for i in range(0, len(sort_result)):
                    sort_key.append([int(d) for d in sort_result[i][0].split(',')])
                sort_dict[label][layer] = sort_key
                '''
        return sort_dict

    def channel_pruning(self, sort_dict, k=5):
        # k:(int) max channels to keep
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            for layer in label_pool:
                layer_pool = label_pool[layer]
                channels = list(layer_pool.keys())
                for channel in channels:
                    if channel not in sort_dict[label][layer][:k]:
                        layer_pool.pop(channel)
                    '''
                    used_channel = [int(d) for d in channel.split(',')]
                    if used_channel not in sort_dict[label][layer][:k]:
                        layer_pool.pop(channel)
                    '''

    def meme_prune(self, keep_channel=20):
        # Retain k significant memes for each label.
        for label_pool in self.meme_dict.values():
            meme_pool = []
            for layer_pool in label_pool.values():
                channels = list(layer_pool.keys())
                for channel in channels:
                    meme_pool += layer_pool.pop(channel)
            meme_pool.sort(reverse=True)
            meme_pool = meme_pool[:keep_channel]
            for meme in meme_pool:
                self.update_meme(meme)

    def total_prune(self, k=50):
        # Retain k significant memes from all.
        meme_pool = []
        for label_pool in self.meme_dict.values():
            for layer_pool in label_pool.values():
                channels = list(layer_pool.keys())
                for channel in channels:
                    meme_pool += layer_pool.pop(channel)
        meme_pool.sort(reverse=True)
        meme_pool = meme_pool[:k]
        for meme in meme_pool:
            self.update_meme(meme)

    def prune_with_corr(self, map_dict, p=0.9):
        #Retain (1-p)% memes with small correlation value
        corr = self.get_xcorr(map_dict)
        max_keep = int(len(corr)*p)
        corr = sorted(corr.items(), key=lambda x: x[1], reverse=False)[:max_keep]
        self.meme_dict = dict()
        for (meme,_) in corr:
            self.update_meme(meme)

    def find_memes(self, search_field, search_data, search_label, search_info, compare_maps, compare_label, target_channels,
                   num_channel, num_label, where=-1, func=print, times = 1.5, meme_dict = {}, prune_iter = 2, p = 0.4):
        compare_map = compare_maps[where]
        assert len(compare_map) == len(compare_label)
        #Offline part: suitable for the case of model invariance
        available_label = np.unique(compare_label)
        init_ent = get_entropy(compare_label)
        memes = []
        count = 0
        for j in range(0, len(target_channels), num_channel):
            channels = target_channels[j:min(j + num_channel,len(target_channels))]
            for idx in range(0, len(available_label), num_label):
                label = available_label[idx:min(idx + num_label,len(available_label))]
                candidates, info, used_channel = get_candidates_in_class(search_field, search_data, search_label,
                                                                         search_info, channels, label, times, meme_dict)
                if len(candidates) > 0:
                    result = get_result(compare_map, compare_label, candidates, info, used_channel, init_ent)
                    memes += self.parallel_get(result, where)
            to_print = "Channel " + str(j) + "~" + str(min(j + num_channel, len(target_channels))) + " finished!"
            func(to_print)
            count += 1
            if count % prune_iter == 0:
                memes = self.update_with_label(memes, compare_maps, available_label, p)
        return memes

    def recount_memes(self, compare_dict, compare_label):
        #update the information gain and threshold
        kernels = self.get_kernels()
        init_ent = get_entropy(compare_label)
        for where in kernels:
            meme_pool = kernels[where]
            print("Update meme on "+str(where)+".")
            feature_map = compare_dict[where]
            for memes in meme_pool.values():
                meme_kernel = np.array(memes["kernel"], dtype=np.float32)
                meme_channel = [item[1] for item in memes["index"]]
                meme_label = [item[0] for item in memes["index"]]
                ig, thresholds = recount(meme_kernel, meme_channel, feature_map, compare_label, meme_label, init_ent)
                for i in range(len(ig)):
                    label,channel,layer,idx = memes["index"][i]
                    self.meme_dict[label][layer][channel][idx].append(ig[i],thresholds[i])
                print("Information gain and threshold updated!")

    def directly_update(self, search_field, search_info, compare_map, compare_label, target_channels,
               num_channels, where=-1, func=print):
        #Select the appropriate meme directly from the candidates
        memes = []
        assert len(compare_map) == len(compare_label)
        init_ent = get_entropy(compare_label)
        for j in range(0, len(target_channels), num_channels):
            channels = target_channels[j:min(j + num_channels, len(target_channels))]
            candidates = np.concatenate(search_field)
            info = np.concatenate(search_info)
            used_candidates = np.repeat(candidates, len(channels), axis=0)
            used_info = np.repeat(info, len(channels), axis=0)
            used_channel = np.reshape(np.repeat(np.expand_dims(channels, 0), len(candidates), axis=0), (-1, 1))
            result = get_result(compare_map, compare_label, used_candidates, used_info, used_channel, init_ent)
            memes += self.parallel_get(result, where)
        return memes

    def get_memes_from_field(self, search_dict, search_label, compare_dict, compare_label, target_channels,
                             target_stride, num_channels, num_labels, sizes, times = 2, meme_dict = {},
                             prune_iter = 2, p = 0.4):
        # input:
        # search_dict: dict
        # feature maps recorded
        # search_label: array
        # label for each data
        # compare_dict: dict
        # feature maps recorded
        # compare_label: array
        # label for each data
        # target_channels:array
        # the list of channels concerned in the extraction process
        # num_channels:int
        # Number of channels extracted at once(affect the calculating speed)
        # num_labels:int
        # Number of labels extracted at once(affect the calculating speed)
        # sizes:list of tuples
        for where in search_dict:
            record = search_dict[where]
            assert where in sizes
            assert where in compare_dict
            assert where in target_stride
            print("Search memes in " + str(where))
            used_dict = meme_dict[where] if len(meme_dict) and where in meme_dict else {}
            for size in sizes[where]:
                print("Search memes with shape of " + str(size[0]) + "×" + str(size[1]))
                target_channel = target_channels[where]
                search_field, search_infos = extract.generate_candidates(record,search_label,size,target_stride[where])

                memes = self.find_memes(search_field, record, search_label, search_infos, compare_dict, compare_label,
                                        target_channel, num_channels, num_labels, where=where, func=print,
                                        times = times, meme_dict = used_dict, prune_iter = prune_iter, p = p)
                memes = self.similarity_update(memes, compare_dict, p)
                for meme in memes:
                    if np.mean(meme.threshold) != 0:
                        #There is no classification value at this time
                        self.update_meme(meme)

    def reload_memes(self, search_dict, search_label, positions, compare_dict, compare_label,
                     target_channels, num_channels, meme_sizes, idx, func = print):
        for where in search_dict:
            func("Update memes in " + where + ".")
            record = search_dict[where]
            position = positions[where]
            regions = get_regions(record, positions[where])
            target_channel = target_channels[where]
            assert where in compare_dict
            for meme_size in meme_sizes[where]:
                search_field, search_info = extract.generate_candidates_with_position(regions,
                                               search_label, position, meme_size)
                search_info = update_info(search_info, idx)
                memes = self.directly_update(search_field, search_info, compare_dict[where], compare_label, target_channel,
                                        num_channels, where=where, func=func)
                self.similarity_update(memes, compare_dict)

    def reload_from_depository(self, mode = "load", **kargs):
        #mode: reload pattern
        #1.load: Read sample image through cv2
        #2.backup: Backup from tensor images
        #kargs: Related parameter information
        #load_path: str
        #images: np.array
        count = 0
        dataset = []
        labels = []
        positions = dict()
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            for layer in label_pool:
                if layer not in positions:
                    positions[layer] = []
                layer_pool = label_pool[layer]
                for channel in layer_pool:
                    channel_pool = layer_pool[channel]
                    for meme in channel_pool:
                        idx, x, y = meme.indices
                        _, length, width = meme.meme.shape
                        if mode == "backup":
                            assert "images" in kargs
                            data = kargs["images"][idx]
                        elif mode == "load":
                            if "load_path" in kargs:
                                load_path = kargs["load_path"]
                            else:
                                load_path = "depository"
                            filename = str(label) + '_' + str(layer) + '_' + channel + '_' + str(-idx)
                            storename = os.path.join(load_path, filename + ".png")
                            if os.path.exists(storename):
                                data = np.transpose(cv2.imread(storename), (2, 0, 1)) / 255
                            else:
                                storename = os.path.join(load_path, "L_" + filename + ".png")
                                data = np.expand_dims(cv2.imread(storename, 0), 0) / 255

                        dataset.append(data)
                        labels.append(label)
                        positions[layer].append([count, x, y, x + length, y + width])
                        count += 1
        reload_data = np.array(dataset, dtype=np.float32)
        return reload_data, labels, positions

    def get_xcorr(self, field_dict):
        #compute the relative value of current field data
        #avg time in 5000+memes:3.2s
        xcorr=dict()
        kernels = self.get_kernels()
        source,idx = threshold_classify(kernels, field_dict)
        length = len(source)
        assert len(idx)==length
        for i in range(length):
            item = source[i]
            ent = utils.cal_entropy(item)
            sum_info = 0
            for j in range(length):
                if j!=i:
                    target = source[j]
                    sum_info +=  utils.cal_condition_ent(item, target)
            label,channel,layer,list_idx = idx[i]
            target_meme = self.meme_dict[label][layer][channel][list_idx]
            xcorr[target_meme] = ent - sum_info / (length-1)
        return xcorr

    def update_with_label(self, memes, field_dict, labels, p=0.6):
        meme_dict = {}
        print("Remove high correlation objects from " + str(len(memes)) + " memes")
        for label in labels:
            meme_dict[label] = []
        for meme in memes:
            meme_dict[meme.label].append(meme)
        result = []
        for label in meme_dict:
            meme_list = meme_dict[label]
            if len(meme_list) > 0:
                result += self.similarity_update(meme_list, field_dict, p, show = False)
        print(str(len(result)) + " memes remain!")
        return result

    def similarity_update(self, memes, field_dict,
                          p = 0.6, show = True, batch_size = 128):
        #compute the relative value between memes and field
        #memes: list of meme,compare the similarity
        memes.sort(reverse=True)
        memes_kernels = memes2kernels(memes)

        meme_ans, meme_idx = threshold_classify(memes_kernels, field_dict)

        length = len(meme_ans)

        if len(self)!=0:
            dict_kernels = self.get_kernels()
            dict_ans,_ = threshold_classify(dict_kernels, field_dict)
        else:
            dict_ans = np.empty(shape=(0,meme_ans.shape[1]))

        if show:
            print("Remove high correlation objects from " + str(length) + " memes")

        np.random.shuffle(meme_ans)
        result = []
        for i in range(0, length, batch_size):
            xcorr = []
            used_ans = meme_ans[i : i + batch_size]
            for j in range(len(used_ans)):
                xcorr.append(utils.get_corr(j, used_ans, dict_ans))
            max_keep = int(len(xcorr) * p)
            idx = np.argpartition(xcorr, max_keep)[:max_keep]
            dict_ans = np.vstack((dict_ans, used_ans[idx]))
            for index in idx:
                result.append(memes[meme_idx[i + index][0]])
        if show:
            print(str(len(result)) + " memes remain!")
        
        return result

    '''
        xcorr = list()
        used_memes = list()
        for i in range(length):
            corr = utils.get_corr(i,length,meme_ans)
            used_memes.append(memes[meme_idx[i][0]])
            xcorr.append(corr)
        idx = np.where(xcorr < np.percentile(np.array(xcorr),
                            100 * p))[0]
        if show:
            print(str(len(idx))+" memes remain!")
        return [memes[i] for i in idx]
        '''


    def lookback(self, map_dict, zscore):
        for label_pool in self.meme_dict.values():
            for layer, layer_pool in label_pool.items():
                for channel in layer_pool:
                    channel_pool = layer_pool[channel]
                    for meme in channel_pool:
                        height, width = meme.meme.shape[1], meme.meme.shape[2]
                        idx, h, w = meme.indices
                        meme.meme = map_dict[layer][idx, :, h:h + height, w:w + width]
                        mean = zscore.zscore[layer].mean[channel].item()
                        std = zscore.zscore[layer].var[channel].item()
                        meme.threshold[-1] = meme.threshold[-1] * np.sqrt(std) + mean

    def cal_resuse(self,length):
        #Calculate the reuse rate of samples in the update process
        count = 0
        for label_pool in self.meme_dict.values():
            for layer_pool in label_pool.values():
                for channel_pool in layer_pool.values():
                    for meme in channel_pool:
                        idx = meme.indices[0]
                        if idx < 0 or idx >= length:
                            count += 1
        if len(self):
            return count / len(self)
        else:
            return 0

    def update_name(self, receptive_field, func=print):
        func("Meme location renamed.")
        name_dict={}
        for label_pool in self.meme_dict.values():
            for layer in label_pool:
                layer_pool = label_pool[layer]
                if layer not in name_dict:
                    name_dict[layer] = receptive_field.get_layer(layer)
                for channel_pool in layer_pool.values():
                    for meme in channel_pool:
                        meme.where = name_dict[layer]

    def manual_update(self, target_dir = "memes"):
        vis_list = os.listdir(target_dir)
        keep_memes = []
        for name in vis_list:
            filename = os.path.splitext(name)[0]
            infos = filename.split("_")
            if len(infos) == 4:
                label = int(infos[0])
                where = infos[1]
                channel = int(infos[2])
                idx = int(infos[3]) - 1
            keep_memes.append(self.meme_dict[label][where][channel][idx])
        self.eliminate()
        for meme in keep_memes:
            self.update_meme(meme)

    def rounded(self,round_up=8):
        for label in self.meme_dict:
            label_pool = self.meme_dict[label]
            for layer in label_pool:
                layer_pool = label_pool[layer]
                for channel in layer_pool:
                    channel_pool = layer_pool[channel]
                    for meme in channel_pool:
                        meme.meme = np.round(meme.meme,round_up)
                        meme.threshold = [round(i,round_up) for i in meme.threshold]
                        meme.information_gain = [round(i,round_up) for i in meme.information_gain]
