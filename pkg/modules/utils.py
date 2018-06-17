import torch.nn as nn
from collections import OrderedDict


class SequentialMaker:
    def __init__(self):
        self.dict_ = OrderedDict()

    def add_module(self, name, module):
        if hasattr(module, "weight"):
            module = nn.utils.weight_norm(module)
        self.dict_[name] = module

    def __call__(self):
        return nn.Sequential(self.dict_)

    @staticmethod
    def print_shape(seq, inputs, intent_size=0):
        dict = OrderedDict()
        dict["inputs"] = list(inputs.size())
        cur = inputs
        for name, m in seq.named_children():
            cur = m(cur)
            dict[name] = list(cur.size())
        max_name = 0
        max_num = []
        for k in dict:
            v = dict[k]
            if len(k) > max_name:
                max_name = len(k)
            for i in range(len(v)):
                # add new col
                if i >= len(max_num):
                    max_num.append(0)
                if len(str(v[i])) > max_num[i]:
                    max_num[i] = len(str(v[i]))
        for k in dict:
            v = dict[k]
            print(" " * intent_size + k + " " * (max_name - len(k)), end=": [ ")
            for i in range(len(v)):
                s = str(v[i])
                print(" " * (max_num[i] - len(s)) + s, end=", " if i + 1 < len(v) else "] \n")
