import node as n
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats


class decision_tree(nn.Module):
    def __init__(self):
        super(decision_tree, self).__init__()
        self.exit_node = None # to keep track of loss in leaf
        self.root = n.node(1)  # root node starts from depth 1
        
        output_dimensions = self.root._get_output_dimensions()
        self.root.left_node = n.node(2, output_dimensions)
        self.root.right_node = n.node(2, output_dimensions)
        
        self.size = 3
        self.height = 2
        #self.leaves = [self.root]
        self.leaves = [self.root.left_node, self.root.right_node]
        

    def extend(self, node):
        depth = node.node_depth
        output_dimensions = node._get_output_dimensions()
        node.left_node = n.node(depth+1, output_dimensions)
        node.right_node = n.node(depth+1, output_dimensions)
        self.size += 2
        self.height += 1
        self.leaves.remove(node)
        self.leaves.append(node.left_node)
        self.leaves.append(node.right_node)
        print("Node extended")
        print("Tree size:"+str(self.size))
        print("Tree height:"+str(self.height))
        
        
    def propagation_criteria(self, y):
       #print(scipy.stats.entropy(y.cpu().data.numpy()[0]))
        if np.sum((y.cpu().data.numpy()[0])) == 0:
            return 0.5
        #print(scipy.stats.entropy(y.cpu().data.numpy()[0]))
        return scipy.stats.entropy(y.cpu().data.numpy()[0])
        
        
    def forward(self, x):
        next_node = self.root
        left_count = 0
        right_count = 0
        while True:
            self.exit_node = next_node # this will be the leaf of path
            x = next_node.conv(x)
            x = F.relu(x)
            
            y = next_node.pool(x)
            y = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
            y = F.relu(next_node.fc(y))
            y = F.relu(next_node.fc2(y))
              
            if self.propagation_criteria(y) < 0.5 :
                next_node = next_node.left_node
                left_count += 1
            else : 
                next_node = next_node.right_node
                right_count +=1
            
            if next_node == None:
                #print("Fw prop left_count: "+str(left_count))
                #print("Fw prop right_count: "+str(right_count))
                return y

    