import time
import numpy as np
from module import *
from meme import MemePools
from tools import Zscore,get_layers
from model.memenet import MemeNet
from net import Net

from params import extract_iter, train_loader, test_loader,\
    targets, target_channels, target_stride, meme_sizes,\
    num_channels, num_labels, capacity, \
    momentum, origin_interval, prune_iter, p, keep_channel,\
    update_batch_size, max_size
from tools import printr

interval = origin_interval

net = ...

memes = MemePools(...)
memenet = MemeNet(...)

if __name__ == "__main__":
    rf = memenet.rf
    targets, target_channels, target_stride, meme_sizes = memenet.update_name(targets,
                                                  target_channels, target_stride, meme_sizes)
    zscore = Zscore(target_channels)

    memenet.load_prototype(...)
    memenet.demonstrate()

    for target in targets:
        memenet.add_monitor(target)

    record_image = []
    record_label = []

    t1 = time.time()
    memenet.switch_monitoring()

    for i, (images, labels) in enumerate(train_loader):
        memenet.get_representation(images)
        record_image.append(images)
        record_label.append(labels)
        if (i + 1) % extract_iter != 0:
            continue
        else:
            iter = i // extract_iter
            print("Iter " + str(iter))
            map_dict = memenet.fetch_record(info = True)
            map_dict = zscore.update(map_dict, True)
            batch_data = torch.cat(record_image).detach().numpy()
            used_labels = torch.cat(record_label).detach().numpy()
            record_image = []
            record_label = []
        if iter != 0:
            memes.recount_memes(map_dict, used_labels)

            interval = (1 + momentum) * interval
            print("Sampling interval:", interval)

            meme_dict = get_layers(memes.get_representation())

            memes.get_memes_from_field(search_dict=map_dict, search_label=used_labels, compare_dict=map_dict,
                                       compare_label=used_labels, target_channels=target_channels,
                                       target_stride=target_stride, num_channels=num_channels,
                                       num_labels=num_labels, sizes=meme_sizes,
                                       times=interval, meme_dict=meme_dict,
                                       prune_iter = prune_iter, p=p,
                                       update_batch_size = update_batch_size, max_size = max_size)
        else:
            print("Sampling interval:", interval)
            memes.get_memes_from_field(search_dict=map_dict, search_label=used_labels, compare_dict=map_dict,
                                       compare_label=used_labels, target_channels=target_channels,
                                       target_stride=target_stride, num_channels=num_channels,
                                       num_labels=num_labels, sizes = meme_sizes,
                                       times=interval, prune_iter = prune_iter, p=p,
                                       update_batch_size = update_batch_size, max_size = max_size)
        memes.lookback(map_dict,zscore)
        memes.meme_prune(keep_channel = ...)
        memes.total_prune(k = ...)
        print("Reuse rate:",memes.cal_resuse(len(batch_data)))
        reload_data, reload_label, positions = memes.reload_from_depository(mode = "backup", images = batch_data)
        key_dict, history = memes.retrieve()
        memenet.show_rf(..., ..., history = ..., transparency = ...)
        acc = memenet.test_prototype(test_loader)
        print(acc)
    memenet.switch_monitoring()
    memes.rename_file()
    print(time.time()-t1)

    memes.get_max_ig()

    memes.rounded(...)
    memes.save_compress('simple.json')
    zscore.save("zscore.json")

    kernels = memes.get_kernels()


    memenet.add_memes(kernels, zscore_ED_mask, zscore.zscore)

    memenet.visualize()
    channels = ...

    memenet.making_classifier((),channels)

    
    memenet.formal_train(train_loader, test_loader, epochs=..., lr=...)
    print("result:" + str(memenet.test_model_label(test_loader)))
    memenet.save_model(...)
