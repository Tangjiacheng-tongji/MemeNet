import torch
import numpy as np
import torchvision.models as models

#this file is intended to compute the receptive field
module_list=[torch.nn.Conv2d,torch.nn.MaxPool2d,torch.nn.AvgPool2d]
#the list of downsampling layers
end_list=[torch.nn.AdaptiveAvgPool2d,torch.nn.Linear]
#the list of specific layers which end tensor calculation

branchA = ["branch1x1", "branch5x5_1", "branch3x3dbl_1", "branch_pool"]
branchB = ["branch3x3", "branch3x3dbl_1", "branch_pool"]
branchC = ["branch1x1", "branch7x7_1", "branch7x7dbl_1", "branch_pool"]
branchD = ["branch3x3_1", "branch7x7x3_1", "branch_pool"]
branchE = ["branch1x1", "branch3x3_1", "branch3x3dbl_1", "branch_pool"]
branchres = ["downsample"]
cat_mode = {models.inception.InceptionA : branchA, models.inception.InceptionB : branchB,
            models.inception.InceptionC : branchC, models.inception.InceptionD : branchD,
            models.inception.InceptionE : branchE, models.resnet.BasicBlock: branchres}

recording_layer = ["branch3x3_2a", "branch3x3dbl_3a"]
# Changes in the receptive field of these channels will not be transmitted

def expand(data):
    if isinstance(data,tuple):
        return np.array(data)
    else:
        return np.array((data,data))

def insistence(item,instance_list):
    for instance in instance_list:
        if isinstance(item, instance):
            return True
    else:
        return False

def unfoldLayer(model, layer, prefix = '', complete = True):
    layer_list = list(model.named_children())
    for item in layer_list:
        name=item[0]
        module = item[1]
        sublayer = list(module.named_children())
        if len(sublayer) == 0:
            layer[prefix + name]=module
        elif isinstance(module, torch.nn.Module):
            if complete:
                unfoldLayer(module, layer, prefix = prefix + name + '.', complete = complete)
            layer[prefix + name] = module

def getLayers(model, complete = True):
    layer_dict = dict()
    unfoldLayer(model, layer_dict, complete = complete)
    return layer_dict

def get_sublayers(model):
    sublayers=[]
    layer_list = list(model.named_children())
    for item in layer_list:
        name,module = item
        sublayer = list(module.named_children())
        if len(sublayer) != 0:
            sublayers.append(name)
    return sublayers

def check_name(name, keywords_list):
    for keyword in keywords_list:
        if keyword == name:
            return True

def find_name(name, keywords_list):
    for keyword in keywords_list:
        if keyword in name:
            return True

def get_rf(module, origin_start, origin_r, origin_j, module_name="", branch = []):
    result = {}
    layers = getLayers(module, False)
    record_r = []
    start,j,r = origin_start,origin_j,origin_r
    for name in layers:
        if check_name(name, branch):
            record_r.append(r)
            start,j,r = origin_start,origin_j,origin_r
        item = layers[name]
        if insistence(item,end_list):
            if "Aux" in name: continue
            else: break
        if len(module_name) > 0:
            name = module_name + "." + name
        if len(list(item.named_children())) == 0:
            if insistence(item,module_list):
                filter_size = expand(item.kernel_size)
                stride = expand(item.stride)
                padding = expand(item.padding)

                r=r+(filter_size-1)*j
                start=start+((filter_size-1)/2-padding)*j
                j=j*stride
            result[name] = {"start":start,"j":j,"r":r}
        else:
            if type(item) in cat_mode:
                branch = cat_mode[type(item)]
                sub_result = get_rf(item, start, r, j, name, branch = branch)
            else:
                sub_result = get_rf(item, start, r, j, name)
            result.update(sub_result)
            if find_name(name, recording_layer):
                record_r.append(sub_result[name]["r"])
            else:
                start, j, r = sub_result[name]["start"],sub_result[name]["j"],sub_result[name]["r"]
    record_r.append(r)
    if len(module_name) > 0:
        result[module_name] = {"start":start,"j":j,"r":np.max(np.array(record_r),axis=0)}
    return result

def show_layers(model, prefix = "", func = print):
    sublayers = get_sublayers(model)
    layers = list(model.named_children())
    for name,structure in layers:
        used_name = prefix + str(name)
        if name in sublayers:
            output = str(used_name) +":"
            func(output)
            show_layers(structure, prefix +name + "." , func)
        else:
            output=str(used_name) + ": " + str(structure)
            func(output)

def summary(model, hooks = {}, in_submodule = 0, backbone_length_limit = 20, prefix = "", tab = "", func=print):
    sublayers = get_sublayers(model)
    layers = list(model.named_children())
    for name,structure in layers:
        backbone_name = prefix + name
        if backbone_name in hooks:
            left_output = (backbone_name + " (monitored)").center(backbone_length_limit)
        else:
            left_output = backbone_name.center(backbone_length_limit)
        func(tab+left_output)
        if name in sublayers:
            in_submodule += 1
            summary(structure, hooks, in_submodule, prefix = prefix + name + ".", func=func) 
            in_submodule -= 1
        if in_submodule == 0 and backbone_name!=layers[-1][0]:
            left_output = '|'.center(backbone_length_limit)
            func(tab+left_output)

def extend_summary(model, hooks = {}, memes = {}, in_submodule = 0, keywords=["fc","classifier"], prefix = "", 
               intermediate = 6, intermediate_text = 15, branch_length_limit = 21, 
               branch = False, backbone_length_limit = 20, func=print):
    sublayers=get_sublayers(model)
    layers = list(model.named_children())
    for count in range(len(layers)):
        name,structure = layers[count]
        backbone_name = prefix + name
        if backbone_name in hooks:
            left_output = (backbone_name + " (monitored)").center(backbone_length_limit)
        else:
            left_output = backbone_name.center(backbone_length_limit)
        if backbone_name in memes:
            kernel_size = [str(item[0])+"×"+str(item[1]) for item in memes[backbone_name].keys()]
            used_size = ','.join(kernel_size)
            middle_output = intermediate * '-' + used_size.center(intermediate_text) + intermediate * '-'
            right_output = '+'.center(branch_length_limit)
            branch = True
        elif branch:
            middle_output = (2*intermediate+intermediate_text) * ' '
            right_output = '|'.center(branch_length_limit)
        else:
            middle_output = ''
            right_output = ''
        func(left_output+middle_output+right_output)
        if name in sublayers:
            in_submodule += 1
            branch = extend_summary(structure, hooks, memes, in_submodule, keywords, prefix = prefix + name + ".", branch = branch, func=func) 
            in_submodule -= 1
        if in_submodule == 0 and backbone_name!=layers[-1][0]:
            left_output = '|'.center(backbone_length_limit)
            if branch:
                middle_output = (2*intermediate+intermediate_text) * ' '
                right_output = '|'.center(branch_length_limit)
                func(left_output + middle_output + right_output)
            else:
                func(left_output)
       
    return branch