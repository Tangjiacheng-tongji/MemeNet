import numpy as np
import warnings
from .load import get_rf

class receptive_fields():
    def __init__(self, net):
        start = np.ones(2)*0.5
        j = np.ones(2).astype(np.int32)
        r = np.ones(2).astype(np.int32)
        self.receptive_fields = get_rf(net, start, j, r)
        self.receptive_fields["start"] = {"start": start, "j": j,
                                          "r": r}

    def add_rf(self,where,shape):
        layer_rf = self.receptive_fields[where]
        new_r = layer_rf["r"]+(np.array(shape)-1)*layer_rf["j"]
        new_start = layer_rf["start"]+((np.array(shape)-1)/2)*layer_rf["j"]
        new_j=layer_rf["j"]
        key = where + "_" + str(shape[0]) + "-" + str(shape[1])
        self.receptive_fields[key]={"start":new_start,"j":new_j,"r":new_r}

    def get_receptive_field(self,where,idx):
        #where:target layer to analysis(name)
        #idx:direction of the picture
        real_where = self.get_layer(where)
        center=self.receptive_fields[real_where]['start']+\
                idx*self.receptive_fields[real_where]['j']
        left_upper=center-self.receptive_fields[real_where]['r']/2
        right_lower=center+self.receptive_fields[real_where]['r']/2
        return left_upper,right_lower

    def get_coord(self,target,shape):
        #Get coordinate information
        x_coord=self.receptive_fields[target]['start'][0]+\
                np.arange(shape[0])*self.receptive_fields[target]['j'][0]
        y_coord=self.receptive_fields[target]['start'][1]+\
                np.arange(shape[1])*self.receptive_fields[target]['j'][1]
        return x_coord,y_coord

    def get_rf(self, target, start, shape):
        #Get receptive field of several pixels
        #real_where = self.get_layer(target)
        real_where = target
        end = start + shape - 1
        rf_info = self.receptive_fields[real_where]
        center_lu = rf_info['start'] + start * rf_info['j']
        center_rl = rf_info['start'] + end * rf_info['j']
        left_upper = center_lu - rf_info['r'] / 2
        right_lower = center_rl + rf_info['r'] / 2
        return left_upper, right_lower

    def limiter(self,target,left_upper,right_lower,amount):
        #Stop considering small receptive field
        rf_info = self.receptive_fields[target]
        #bounder = expand(right_lower) - expand(left_upper)
        pixels = rf_info['j'][0] * rf_info['j'][1]
        return (amount - pixels) >= 0

    def get_optional_layers(self, layer_dict):
        optional_layers = {}
        for name in self.receptive_fields.keys():
            if name in layer_dict:
                optional_layers[name] = layer_dict[name]
        return optional_layers
