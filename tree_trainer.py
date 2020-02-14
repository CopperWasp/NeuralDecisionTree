import node as n
import torch.nn as nn
import preprocess as pre
from torch.autograd import Variable
import torch.optim as optim
import torch
import torchvision
import decision_tree as dt
import visualize as v

# starting with 3 nodes improved
class tree_trainer:
    def train_tree(self):
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.tree = dt.decision_tree().cuda()
        print(self.tree)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.tree.parameters(), lr=0.001, momentum=0.9)
      
        
        for epoch in range(n.tree_trainer_epoch):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(pre.trainloader, 0):
                inputs, labels = data
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.tree(inputs)
                g = v.make_dot(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.data[0]
                # update leaf loss statistics
                self._update_leaf_loss(loss.data[0], self.tree.exit_node) # must be called after backwards for accurate exit node
                
                if i % 2000 == 1999:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0
            # after every epoch, grow the tree
            if self.tree.size * 3 == epoch:
                self._grow_tree()
        print('Finished Training')
        g.view()
        


    def _update_leaf_loss(self, loss, leaf_node):
        # check if that node is still a leaf, error checks
        if leaf_node in self.tree.leaves:
            leaf_node.update_loss(loss)
        else:
            print("update leaf loss something wrong")
        

    def test_tree_visual(self):
        dataiter = iter(pre.testloader)
        images, labels = dataiter.next()
        #pre.imshow(torchvision.utils.make_grid(images))
        print('GroundTruth: ', ' '.join('%5s' % pre.classes[labels[j]] for j in range(n.batch_size)))
        outputs = self.path(Variable(images).cuda())
        _, predicted = torch.max(outputs.data, 1)
        print('Predicted: ', ' '.join('%5s' % pre.classes[predicted[j]] for j in range(n.batch_size)))
                   

    def test_tree(self):
        indv_perfs = []
        correct = 0
        total = 0
        for data in pre.testloader:
            images, labels = data
            outputs = self.tree(Variable(images).cuda())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        for data in pre.testloader:
            images, labels = data
            outputs = self.tree(Variable(images).cuda())
            _, predicted = torch.max(outputs.data, 1)
            c = (predicted == labels.cuda()).squeeze()
            for i in range(n.batch_size):
                label = labels[i]
                class_correct[label] += c[i]
                class_total[label] += 1
        
        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                pre.classes[i], 100 * class_correct[i] / class_total[i]))
            indv_perfs.append(100 * class_correct[i] / class_total[i])
        v.visualize_barchart(indv_perfs)
            
        
        
    def _grow_tree(self):
        #find the leaf with lowest score, grow it
        tree = self.tree
        leaves = tree.leaves
        
        max_cost = leaves[0].average_loss
        max_leaf = leaves[0]
        
        for leaf in leaves:
            if leaf.average_loss > max_cost:
                max_cost = leaf.average_loss
                max_leaf = leaf
        
        tree.extend(max_leaf)
            
        
# dimension changes, conv layer input output size changes
# propagation_criteria
# growing_criteria
# growing frequency is important    


#write a method to see paths propagated

    

t = tree_trainer()
t.train_tree()
t.test_tree()