import torch
import numpy as np
import os
import cv2
from scipy import interpolate
import torch.nn as nn
from .receptive_field import receptive_fields
from .load import getLayers,show_layers,summary,extend_summary
from .tools import get_parameters, get_remains, get_optional_layers,\
    backward, make_layers, crop_bbox, Standardization, acc_label

class MemeNet(nn.Module):
    def __init__(self, prototype, loss_func = torch.nn.CrossEntropyLoss(),
                 keywords=["fc","classifier"],
                 classifier_layers = None, device_ids = [],
                 func=print):
        #prototype:Prototype network
        super(MemeNet, self).__init__()
        self.func = func

        if len(device_ids) == 0:
            self.model = prototype
            output = "Using cpu to train prototype and classifer."
        else:
            self.model = nn.DataParallel(prototype, device_ids = device_ids).cuda()
            output = "Using gpu " + ','.join(map(str, device_ids)) + " to train."
        self.func(output)
        self.use_gpu = device_ids

        if self.use_gpu:
            self.func("Due to the use of gpu for training, it is recommended to add module. before the name of the layer")

        self.keywords=keywords
        self.demonstrate()

        self.optional_node = get_optional_layers(getLayers(self.model),self.keywords)
        self.id2names = {id(self.optional_node[i]): i for i in self.optional_node}
        self.rf = receptive_fields(self.model)

        self.base,self.fc = get_parameters(self.model,self.keywords)
        self.loss_func = loss_func

        self.rec =False
        self.monitor = {}
        self.points = {}

        self.datas = {}
        self.labels = {}
        
        self.dist_maps = {}
        self.memes = {}
        self.num_memes = 0

        if classifier_layers is None:
            self.classifier = nn.Sequential()
            self.length = 0
        else:
            self.length = len(classifier_layers)
            self.classifier = classifier_layers
            if len(self.use_gpu):
                self.classifier.to(self.use_gpu[0])

        self.layers = {}
        self.mode = 'dist'

    def warm_only(self):
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = False

    def start_up(self):
        for param in get_remains(self.model,[self.base,self.fc]):
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = True
        for param in self.base:
            param.requires_grad = False
        for param in self.fc:
            param.requires_grad = False

    def warm_train(self, train_loader, valid_loader, lr = [5e-3, 1e-2], epochs=3):
        self.func("Start training prototype...")
        self.warm_only()
        if len(lr) == 2:
            lr_base, lr_fc = lr
        else:
            lr_base = lr[0]
            lr_fc = lr[0]
        warm_optimizer_specs = \
            [{'params': self.base, 'lr': lr_base},
             {'params': self.fc, 'lr': lr_fc},
             ]
        vis_iter = 5
        warm_optimizer = torch.optim.SGD(self.model.parameters(), lr = lr[0], momentum = 0.9)
        count = 0
        for epoch in range(epochs):
            self.func("epoch:" + str(epoch + 1) + "/" + str(epochs))
            for i, (images, labels) in enumerate(train_loader):
                if len(self.use_gpu):
                    images, labels = images.cuda(), labels.cuda()
                warm_optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                warm_optimizer.step()
                count += 1
                if count % vis_iter == 0:
                    self.func("iter " + str(count) + "| loss: " + str(loss.item()))
            warm_optimizer.zero_grad()
            accuracy = self.test_prototype(valid_loader)
            self.func("Acc: " + str(accuracy))
        self.func("final acc: " + str(accuracy))

    def warm_train_once(self, train_data, train_label):
        self.func("Start training prototype once...")
        self.warm_only()
        warm_optimizer_specs = \
            [{'params': self.base, 'lr': 5e-3},
             {'params': self.fc, 'lr': 1e-2},
             ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        if len(self.use_gpu):
            train_data, train_label = train_data.cuda(), train_label.cuda()
        loss = backward(train_data, train_label, self.model, warm_optimizer, self.loss_func)
        output = "Data amount:"+ len(train_data) + "| Loss: "+loss.item()
        self.func(output)

    def baseline_train(self, train_loader, valid_loader, epochs=3):
        self.func("Start training the prototype classifier...")
        self.warm_only()
        baseline_optimizer = torch.optim.Adam(self.fc,lr=1e-2)

        count = 0
        vis_iter = 5

        for epoch in range(epochs):
            self.func("epoch:" + str(epoch + 1) + "/" + str(epochs))
            for i, (images, labels) in enumerate(train_loader):
                if len(self.use_gpu):
                    images, labels = images.cuda(), labels.cuda()
                baseline_optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                baseline_optimizer.step()
                count += 1
                if count % vis_iter == 0:
                    self.func("iter " + str(count) + "| loss: " + str(loss.item()))
            baseline_optimizer.zero_grad()
            accuracy = self.test_prototype(valid_loader)
            self.func("Acc: " + str(accuracy))
        self.func("final acc: " + str(accuracy))

    def test_prototype(self, test_loader):
        self.func("Testing prototype...")
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if len(self.use_gpu):
                    images = images.cuda()
                    labels = labels.cuda()
                self.model.eval()
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            return accuracy

    def test_model(self, test_loader):
        self.func("Testing model...")
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                if self.use_gpu:
                    images = images.cuda()
                    labels = labels.to(self.use_gpu[0])
                features = self.get_features(images)
                outputs = self.classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            return accuracy

    def test_prototype_label(self, test_loader):
        self.func("Label by label classification...")
        return acc_label(self.model, test_loader,len(self.use_gpu))

    def test_model_label(self, test_loader):
        self.func("Label by label classification(main model)...")
        return acc_label(self, test_loader, len(self.use_gpu))

    def demonstrate(self,prototype=True):
        if prototype==True:
            self.func('The structure of the backbone is:')
            show_layers(self.model, func = self.func)
        else:
            assert len(self.classifier)!=0,"Classifier layers must be given"
            self.func('The structure of the classifier is:')
            show_layers(self.classifier, func = self.func)

    def visualize(self):
        self.func('The structure of the memenet is:')
        backbone_length_limit=20
        intermediate = 5
        intermediate_text = 10
        branch_length_limit = 21
        if len(self.memes) != 0:
            extend_summary(self.model, self.points, self.memes, keywords=self.keywords, intermediate = intermediate,
               intermediate_text = intermediate_text, branch_length_limit = branch_length_limit, 
               backbone_length_limit = backbone_length_limit, func = self.func)
            tab = (backbone_length_limit + 2 * intermediate + intermediate_text) * " "
            if len(self.classifier) != 0:
                summary(self.classifier, self.points, backbone_length_limit = branch_length_limit, tab=tab, func = self.func)
            else:
                self.func(tab+"output".center(branch_length_limit))
        else:
            summary(self.model, self.points, backbone_length_limit = backbone_length_limit, func = self.func)

    def add_monitor(self,name):
        print(self.optional_node)
        assert name in self.optional_node, "Layer "+str(name)+" monitoring is not allowed, available monitoring nodes: "+",".join(self.optional_node.keys())
        self.func("Adding agent monitoring the output of layer "+str(name))
        if name not in self.monitor:
            self.monitor[name]=[]
        def forward_point(module,input,output):
            if self.rec:
                self.monitor[name].append(output)
        handle = self.optional_node[name].register_forward_hook(forward_point)
        self.points[name] = handle

    def switch_monitoring(self):
        if self.rec:
            self.rec = False
            output = "All nodes stop monitoring, current status: " + str(self.rec)
        else:
            self.rec = True
            output = "All nodes start monitoring, current status: " + str(self.rec)
        self.func(output)

    def del_points(self):
        for point in self.points.values():
            point.remove()
        self.points={}
        self.monitor={}
        self.func("All monitor nodes deleted.")

    def add_memes(self, kernels, func, zscore_module = {}, used_target ={}):
        # layers: special layers
        # zscore_module: dict(optional)
        # used_target: dict(optional)
        # key: where(str), value: kernel size(list of tuples)
        keys = list(kernels.keys())
        for key in keys:
            real_key = self.get_name(key)
            if key != real_key:
                kernels[real_key]=kernels.pop(key)
        if len(used_target) == 0:
            used_kernels = kernels
        else:
            used_kernels = dict()
            for where in used_target:
                real_where = self.get_name(where)
                assert real_where in kernels
                used_kernels[real_where] = dict()
                sizes = used_target[where]
                for size in sizes:
                    assert size in kernels[real_where]
                    used_kernels[real_where][size] = kernels[real_where][size]
        layers, mode = make_layers(used_kernels, func, zscore_module)
        self.mode = mode
        self.add_layers(layers)

    def add_layers(self,layers):
        for where in layers:
            real_where = self.get_name(where)
            pools = layers[where]
            if where not in self.memes:
                self.memes[real_where]=dict()
            for shape in pools:
                layer = pools[shape]
                self.num_memes += layer.weight.shape[0]
                if self.use_gpu:
                    layer = layer.cuda()
                handle = self.optional_node[real_where].register_forward_hook(self.get_point_func(layer, where, shape))
                self.memes[real_where][shape]=handle
        if not self.layers:
            self.layers=layers
        self.func(str(self.num_memes)+" memes considered.")

    def get_point_func(self, layer, where, shape):
        # Calculate similarity and downsample
        def func(module, input, output):
            ED = layer(output.to(self.use_gpu[0]))
            key = where + '_' + str(shape[0]) + '-' + str(shape[1])
            if key not in self.dist_maps:
                self.dist_maps[key] = []
            self.dist_maps[key].append(ED.to(output.device))
        return func

    def add_classifier(self,num_memes,out_features):
        assert len(self.classifier)==1,"Not suitable for complex classifier"
        layer = self.classifier[0]
        add_memes = num_memes - layer.in_features
        add_output = out_features - layer.out_features
        new_layer=nn.Linear(num_memes, out_features)
        temp_weight = torch.cat((layer.weight, \
                            torch.zeros([layer.out_features, add_memes])), 1)
        new_weight = torch.cat((temp_weight, \
                                torch.zeros([add_output, temp_weight.shape[1]])), 0)
        new_bias = torch.cat((layer.bias, torch.zeros([add_output])),0)
        new_layer.weight.detach().copy_(new_weight)
        new_layer.bias.detach().copy_(new_bias)
        self.classifier = nn.Sequential()
        self.classifier.add_module('classifier', new_layer)
        self.demonstrate(False)

    def load_classifier(self,classifier):
        #classifier: Arbitrary model for output results
        self.classifier = classifier

    def making_classifier(self,num_hiddens,num_classes):
        assert self.length==0,"Available only if the classifier is not defined."
        assert isinstance(num_hiddens,tuple),"Params num_hiddens must be tuple."
        self.func("A "+str(len(num_hiddens)+1)+"-layers neural network is constructed. There are "+str(num_hiddens)+" neurons in the hidden layer, and there are "+str(num_classes)+" outputs in total.")
        self.classifier = nn.Sequential()
        if len(num_hiddens)!=0:
            for i in range(len(num_hiddens)):
                if i==0:
                    self.classifier.add_module('classifier_' + str(i + 1), nn.Linear(self.num_memes, num_hiddens[i]))
                else:
                    self.classifier.add_module('classifier_' + str(i + 1), nn.Linear(num_hiddens[i-1], num_hiddens[i]))
                self.classifier.add_module('relu_' + str(i + 1), nn.ReLU())
            self.classifier.add_module('classifier_' + str(len(num_hiddens)+1), nn.Linear(num_hiddens[i], num_classes))
        else:
            self.classifier.add_module('classifier', nn.Linear(self.num_memes, num_classes))
        self.demonstrate(False)
        self.length = len(self.classifier)
        if len(self.use_gpu):
            self.classifier.to(self.use_gpu[0])

    def del_memes(self):
        for meme_points in self.memes.values():
            for meme_point in meme_points.values():
                meme_point.remove()
        self.memes={}
        self.num_memes=0
        self.classifier = nn.Sequential()
        self.func("All memes deleted.")

    def get_features(self, x):
        for key in self.dist_maps:
            self.dist_maps[key] = []
        with torch.no_grad():
            self.model.eval()
            self.model(x)
            #the result is consistent
            dist_maps = self.get_sim()
            features = torch.cat(dist_maps, 1)
        return features

    def forward(self, x):
        assert self.num_memes != 0, "At least one meme must be given"
        features = self.get_features(x)
        if self.length == 0:
            return features
        else:
            return self.classifier(features)

    def gather_map(self,dist_map):
        if len(self.use_gpu) > 1:
            used_device = [item.device.index for item in dist_map]
            main_device = self.use_gpu[0]
            idx = [i[0] for i in sorted(enumerate(used_device), key=lambda x: x[1])]
            used_map = nn.parallel.gather([dist_map[i] for i in idx], self.use_gpu).to(main_device)
        else:
            used_map = torch.cat(dist_map)
        return used_map

    def get_sim(self):
        dist_maps = []
        for key in self.dist_maps.keys():
            dist_map = self.dist_maps[key]
            used_map = self.gather_map(dist_map)
            if self.mode == 'dist':
                ans = torch.min(torch.min(used_map, 2).values, 2).values
            else:
                ans = torch.max(torch.max(used_map, 2).values, 2).values
            dist_maps.append(ans)
        return dist_maps

    def formal_train(self, train_loader, valid_loader, epochs=3 ,lr=1e-3):
        self.func("Start formal training...")
        self.start_up()
        formal_optimizer = torch.optim.Adam(self.classifier.parameters(),lr=lr)
        count = 0
        for epoch in range(epochs):
            self.func("epoch:" + str(epoch + 1) + "/" + str(epochs))
            for i, (images, labels) in enumerate(train_loader):
                formal_optimizer.zero_grad()
                if self.use_gpu:
                    images = images.cuda()
                    labels = labels.to(self.use_gpu[0])
                features = self.get_features(images)
                #feature is consistent
                outputs = self.classifier(features)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                formal_optimizer.step()
                count += 1
                if count % 5 == 0:
                    self.func("iter " + str(count) + "| loss: " + str(loss.item()))
            formal_optimizer.zero_grad()
            accuracy = self.test_model(valid_loader)
            self.func("Acc: " + str(accuracy))
        self.func("final acc: " + str(accuracy))

    def formal_once(self, train_data, train_label):
        self.start_up()
        formal_optimizer = torch.optim.Adam(self.classifier.parameters(),lr=1e-3)
        if self.use_gpu:
            train_data, train_label = train_data.cuda(), train_label.cuda()
        loss = backward(train_data, train_label, self, formal_optimizer, self.loss_func)
        output = "Data amount:" + len(train_data) + "| Loss: " + loss.item()
        self.func(output)

    def load_prototype(self, filename, mode = "whole"):
        # mode: loading mode
        # "whole" save structure and params
        # "params": save parameters
        if self.use_gpu:
            if mode == "whole":
                self.model = nn.DataParallel(torch.load(filename), device_ids=self.use_gpu).cuda()
            else:
                self.model.load_state_dict(self.update_param(torch.load(filename),0))
        else:
            if mode == "whole":
                self.model = torch.load(filename,map_location='cpu')
            else:
                self.model.load_state_dict(self.update_param(torch.load(filename),0))
        self.base, self.fc = get_parameters(self.model, self.keywords)
        output = 'Load prototype from ' + filename
        self.func(output)

        self.optional_node = get_optional_layers(getLayers(self.model), self.keywords)
        self.id2names = {id(self.optional_node[i]): i for i in self.optional_node}

    def save_model(self,filename):
        output = 'Save the model as ' + filename
        self.func(output)
        features = []
        for layer in getLayers(self.classifier).values():
            if isinstance(layer,nn.Linear):
                features.append(layer.out_features)
        state = {"net": self.state_dict(), "layers" : self.layers,
                 "mode" : self.mode, "features" : features}
        torch.save(state, filename)

    def load_model(self, filename):
        output = 'Load the model from ' + filename
        self.func(output)
        checkpoint = torch.load(filename)
        if len(self.classifier):
            #delete existing classifier
            self.del_memes()
        self.add_layers(checkpoint['layers'])
        features = checkpoint['features']
        self.making_classifier(tuple(features[:-1]),features[-1])
        self.load_state_dict(self.update_param(checkpoint['net']))
        self.mode = checkpoint['mode']

    def update_param(self, params, index=1):
        names = list(params.keys())
        for name in names:
            parts = name.split(".")
            if self.use_gpu:
                if "module" not in parts:
                    if parts[0] != "classifier":
                        parts.insert(index,"module")
                        params[".".join(parts)] = params.pop(name)
            else:
                if "module" in parts:
                    parts.remove("module")
                    params[".".join(parts)] = params.pop(name)
        return params

    def get_representation(self, images, grad =False):
        if self.use_gpu: images = images.cuda()
        if grad: self.model(images)
        else:
            with torch.no_grad():
                self.model(images)

    def fetch_record(self, info = False):
        self.func("Fetching records...")
        dict = self.get_record()
        for buffer in self.monitor.values():
            assert len(buffer) == 0
        if info:
            self.func("Records fetched:")
            for key in dict:
                output = key + " : " + "*".join([str(i) for i in dict[key].shape])
                self.func(output)
        self.func("current recording status: " + str(self.rec))
        return dict

    def get_record(self,refresh=True):
        record={}
        for name in self.monitor:
            result = self.monitor[name]
            if len(result)!=0:
                if len(self.use_gpu)>1:
                    device_order = []
                    for i in range(len(result)):
                        device_order.append(result[i].device.index + i // len(self.use_gpu) * len(self.use_gpu))
                    used_result = [result[i[0]] for i in sorted(enumerate(device_order), key=lambda x: x[1])]
                else:
                    used_result = result
                record[name] = torch.cat([item.cpu() for item in used_result])
                if refresh:self.monitor[name]=[]
        return record

    def get_name(self,name):
        if self.use_gpu:
            if name[:7] == "module.":
                return name
            else:
                return "module." + name
        else:
            if name[:7] == "module.":
                return name[7:]
            else:
                return name

    def update_name(self, targets, target_channels,
                    target_stride, meme_sizes):
        #check names
        new_targets = []
        for target in targets:
            assert target in target_channels
            assert target in target_stride
            assert target in meme_sizes
            real_target = self.get_name(target)
            if real_target != target:
                output = "Changing layer name:" + target + "——>" + real_target
                self.func(output)
                new_targets.append(real_target)
                target_channels[real_target] = target_channels.pop(target)
                target_stride[real_target] = target_stride.pop(target)
                meme_sizes[real_target] = meme_sizes.pop(target)
            else: new_targets.append(target)
        return new_targets, target_channels, target_stride, meme_sizes

    def save_prototype(self,filename):
        output = 'Save the backbone as ' + filename
        self.func(output)
        torch.save(self.model.state_dict(), filename)

    def get_params(self):
        layer=self.classifier[0]
        weight = layer.weight.detach().numpy()
        bias = layer.bias.detach().numpy()
        return weight,bias

    def get_diff(self, imgs, layer, channel, start,
                 shape, batch_size = 32):
        img_tensor = torch.from_numpy(imgs)
        erfs = []
        for i in range(0, len(imgs), batch_size):
            erfs.append(self.get_diff_batch(img_tensor[i:i+batch_size],
                        layer, channel[i:i+batch_size],
                        start[i:i+batch_size], shape[i:i+batch_size]))
        return torch.cat(erfs, dim = 0)

    def get_diff_batch(self, img_tensor, layer,
                       channel, start, shape):
        length = len(img_tensor)
        img_tensor.requires_grad = True

        erf = torch.zeros_like(img_tensor)
        assert self.rec
        self.get_representation(img_tensor, True)
        output = self.get_record()[layer]
        mask = torch.zeros_like(output)
        for i in range(length):
            mask[i, int(channel[i]), int(start[i][0]):int(start[i][0] + shape[i][0]), \
                int(start[i][1]):int(start[i][1] + shape[i][1])] = 1

        output = torch.mean(output * mask, dim=(1, 2, 3))
        kernel = torch.eye(len(img_tensor))
        for i in range(length):
            output.backward(kernel[i], retain_graph=True)
            erf[i] = img_tensor.grad[i]
            img_tensor.grad.zero_()
        del img_tensor
        del kernel
        return erf

    def show_rf(self, images, key_dict, save_path="memes",
                  depository=True, mode = ["heat","erf","trf"],
                history = [], background_color = (255, 255, 255),
                transparency=0.7, bbox_color = (0,255,0),
                output_size = (1200, 1200)):
        #depository: save origional data
        #mode: different working mode
        #heat: Use heat map
        #erf: Show erf
        #trf: Use the bounding box to show trf
        self.func('Start visualizing!')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            vis_list = []
        else:
            vis_list = os.listdir(save_path)
        if depository:
            store_path = "depository"
            if not os.path.exists(store_path):
                os.mkdir(store_path)
                store_list = []
            else:
                store_list = os.listdir(store_path)
        for filename in history:
            vis_list.remove(filename)
            if filename in store_list:
                store_list.remove(filename)
            elif "L_" + filename in store_list:
                store_list.remove("L_" + filename)
        for layer in key_dict:
            infos = key_dict[layer]
            target = np.array(infos["idx"])
            if len(target) > 0 :
                imgs = images[np.array(infos["idx"])]
                start = np.array(infos["start"])
                shape = np.array(infos["shape"])
                channel = np.array(infos["channel"])
                intensitys = infos["intensity"]
                names = infos["name"]
                del infos

                if "erf" in mode:
                    erf = torch.abs(self.get_diff(imgs, layer, channel, start, shape))
                    erf = torch.where(torch.isnan(erf), torch.full_like(erf, 0), erf)
                    erf = torch.sum(erf, dim = 1, keepdim = True)
                    #Focus on the influence of the receptive field, so only consider the absolute value
                    max_value = torch.max(torch.max(erf, -2, keepdim = True).values, -1, keepdim = True).values + 1e-8
                    erf = (erf / max_value).detach().numpy()
                    #set transparency
                    erf = np.ones_like(erf) * (1 - transparency) + erf * transparency
                    if imgs.ndim != 3:
                        erf = erf.transpose((0, 2, 3, 1))

                if "trf" in mode:
                    lu, rl = self.rf.get_rf(layer, start, shape)

                if imgs.ndim != 3:
                    imgs = imgs.transpose((0, 2, 3, 1))

                used_imgs = imgs.copy()

                for i in range(len(used_imgs)):
                    used_img = used_imgs[i]
                    output = cv2.cvtColor(used_img, cv2.COLOR_BGR2RGB) *255

                    if "heat" in mode:
                        intensity = intensitys[i]
                        heatmap = cv2.applyColorMap(
                            np.uint8(255 * (intensity ** 2) * np.ones_like(used_img)),
                            cv2.COLORMAP_JET)
                        output = (0.7 * np.float32(output) + 0.3 * np.float32(heatmap))


                    if "erf" in mode:
                        used_erf = erf[i]
                        output = output * used_erf + background_color * (1 - used_erf)

                    if "trf" in mode:
                        img_shape = np.array(used_img.shape[:2])
                        left_upper = np.clip(lu[i], 1, img_shape - 1)
                        right_lower = np.clip(rl[i], 1, img_shape - 1)
                        cv2.rectangle(output, (int(left_upper[1]), int(left_upper[0])), \
                                      (int(right_lower[1]), int(right_lower[0])), bbox_color)

                    output = cv2.resize(output, output_size)

                    savename = names[i]
                    cv2.imwrite(savename, output)
                    if depository:
                        fullname = os.path.split(savename)[1]
                        if imgs.shape[-1] == 1:
                            fullname = "L_" + fullname
                        storename = os.path.join(store_path, fullname)
                        cv2.imwrite(storename, np.squeeze(imgs[i]) * 255)
                    output = "Memes saved as " + savename
                    self.func(output)
            output = "Delete redundant images."
            self.func(output)
            for filename in vis_list:
                to_del = os.path.join(save_path, filename)
                if os.path.exists(to_del):
                    os.remove(to_del)
            if depository:
                for filename in store_list:
                    to_del = os.path.join(store_path, filename)
                    if os.path.exists(to_del):
                        os.remove(to_del)

    def visualizing(self, input_data, kernels={}, func=Standardization, save_path = "visualization"):
        # input: Tensor
        print(self.classifier.classifier.weight.shape)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            self.func("Delete all pics in "+save_path+".")
            files = os.listdir(save_path)
            for file in files:
                filename = os.path.join(save_path,file)
                os.remove(filename)
        input_img = torch.unsqueeze(input_data, 0)
        if not self.rec:
            self.switch_monitoring()
        for name in self.monitor:
            self.monitor[name] = []
        self.forward(input_img)
        self.switch_monitoring()
        img_length = input_img.shape[-2]
        img_width = input_img.shape[-1]
        img = np.transpose(input_data.detach().numpy(),
                           (1, 2, 0))
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) * 255
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) * 255
        for target in self.dist_maps:
            where, size = target.split("_")
            length, width = size.split("-")
            shape = (int(length), int(width))
            if target not in self.rf.receptive_fields:
                self.rf.add_rf(where, shape)
            used_map = self.gather_map(self.dist_maps[target]).cpu()
            dist_map = used_map[0].detach().numpy()
            quantile = np.percentile(np.max(dist_map,axis=(1,2)), 90)
            for count in range(len(dist_map)):
                target_map = dist_map[count]
                if len(kernels) != 0:
                    label, channel, _, idx = kernels[where][shape]['index'][count]
                    filename = '_'.join(
                            [str(label), where, str(channel), str(idx + 1)]) + '.jpg'
                else:
                    filename = where + '_' + str(count) + '.jpg'

                if self.mode == 'sim':
                    thres = 0.2
                else:
                    if len(kernels) != 0:
                        threshold = kernels[where][shape]['thresholds'][count]
                    else:
                        threshold = 0.5
                    std = func(threshold)
                    target_map = std(target_map)
                    thres = std.res
                    quantile = std(quantile)

                x_coord, y_coord = self.rf.get_coord(target, target_map.shape)
                pad_dist_map = np.pad(target_map, ((1, 1), (1, 1)), 'constant', constant_values=0)
                pad_x_coord = np.pad(x_coord, (1, 1), 'constant', constant_values=(0, img_length))
                pad_y_coord = np.pad(y_coord, (1, 1), 'constant', constant_values=(0, img_width))

                if len(x_coord) == 1 or len(y_coord) == 1:
                    interpfunc = interpolate.interp2d(pad_y_coord, pad_x_coord, pad_dist_map, kind='linear')
                else:
                    interpfunc = interpolate.interp2d(pad_y_coord, pad_x_coord, pad_dist_map, kind='cubic')

                zn = interpfunc(np.arange(img_length), np.arange(img_width))

                regions = crop_bbox(zn, thres, times = 2)

                if len(regions) != 0:
                    bound_img = img.copy()
                    crop = False
                    for region in regions:
                        left_upper, right_lower, amount = region
                        if np.max(target_map) > quantile and self.rf.limiter(where, left_upper, right_lower, amount):
                            cv2.rectangle(bound_img, (left_upper[1], left_upper[0]), (right_lower[1], right_lower[0]),
                                              (0, 255, 0))
                            bound_pic = np.uint8(np.clip(bound_img, 0, 255))
                            crop = True
                    if crop:
                        boundname = os.path.join(save_path, filename)
                        cv2.imwrite(boundname, cv2.resize(bound_pic, (1200, 1200)))
                        self.func(boundname, "saved")
                heatmap = cv2.applyColorMap(np.uint8(np.clip(255 * zn, 0, 255)), cv2.COLORMAP_JET)
                transparency = 0.2
                overlayed_img = transparency * img + (1 - transparency) * heatmap
                heatname = os.path.join(save_path, "heat_" + filename)
                cv2.imwrite(heatname, cv2.resize(overlayed_img, (1200, 1200)))
                self.func(heatname, "saved")
