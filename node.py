import torch.nn as nn
import torch.nn.functional as F
import preprocess as pre
from torch.autograd import Variable
import torch

num_classes = len(pre.classes)
kernel_size = 3
depth_of_input = 3
conv_multiplier = 2
input_shape = (3, 32, 32)
pooling_size = 3
tree_trainer_epoch = 20
fully_connected_divisor = 100 #speed is drastically changed when this increased
batch_size = pre.batchsize

class node(nn.Module):
    def __init__(self, node_depth, input_shape=input_shape):        
        super(node, self).__init__()
        
        self.average_loss = 0
        self.touch_count = 0 # for averaging loss
        
        self.left_node = None
        self.right_node = None        
        self.node_depth = node_depth
        self.items = []  # distribution of examples
        self.input_shape = input_shape
        
        #  conv in and out sizes decided by depth of the node
        self.conv_in = input_shape[0]
        self.conv_out = self.conv_in * conv_multiplier
        
        #  define layers
        self.conv = nn.Conv2d(self.conv_in, self.conv_out, kernel_size)
        self.pool = nn.MaxPool2d(pooling_size, pooling_size)
        
        #  dynamically decide size, based on input shape
        self.n_size = self._get_conv_output()
        self.fc = nn.Linear(self.n_size, int(self.n_size/fully_connected_divisor))
        self.fc2 = nn.Linear(int(self.n_size/fully_connected_divisor), num_classes)
        self.output_shape = self._get_output_dimensions()
        
        #  print the layer
        self._print_layer()
        
        
    def update_loss(self, loss):
        if self.touch_count == 0:
            self.average_loss = loss
        else:
            self.average_loss = (self.average_loss * self.touch_count + loss) / (self.touch_count + 1)
        self.touch_count += 1
      
        
    def get_total_loss(self):  # if frequency of path also matters
        return self.average_loss * self.touch_count
        
    
    def _get_conv_output(self):
        bs = 1
        input = Variable(torch.rand(bs, *(self.input_shape)).cuda())
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size


    def _forward_features(self, x):
        x = self.conv(x)
        #x = self.pool(x)
        x = F.relu(x)
        return x
    
    
    def _get_output_dimensions(self):
        bs = 1
        input = Variable(torch.rand(bs, *(self.input_shape)).cuda())
        output_feat = self._forward_features(input)
        return tuple(output_feat.data.size()[1:])
    
    
    def _print_layer(self):
        print("Node Depth: "+str(self.node_depth))
        print("\tInput: "+str(self.input_shape))
        print("\tOutput: "+str(self.output_shape))
        print("\tConvIn: "+str(self.conv_in))
        print("\tConvOut: "+str(self.conv_out))
        print("\tSerializedSize: "+str(self.n_size))
    
    

    