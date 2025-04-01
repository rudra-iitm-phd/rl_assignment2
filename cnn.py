import torch 
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
      def __init__(self, input_size, output_size, filters:list, padding_config:list, stride_config:list, dense_config:list, conv_activation:list, dense_activation:list, kernel_config:list, batch_size:int):
            super(CNN).__init__()

            self.filter_config = filters
            self.input_size = input_size
            self.input_channel = input_size[1]
            self.output_size = output_size
            self.padding_config = padding_config
            self.stride_config = stride_config
            self.dense_config = dense_config
            self.kernel_config = kernel_config
            self.batch_size = batch_size
            self.conv_activations = conv_activation
            self.dense_activation = dense_activation

            self.cnn = self.generate_cnn()
            self.dense = self.generate_dense()

            self.activations = {
                  'relu':F.relu,
                  'gelu':F.gelu,
                  'sigmoid':F.sigmoid,
                  'silu':F.silu,
                  'mish':F.mish,
                  'tanh':F.tanh,
                  'relu6':F.relu6,
                  'leaky_relu':F.leaky_relu
            }



      def generate_cnn(self) -> dict:

            architecture = dict()

            for i,j in enumerate(self.filter_config):

                  architecture[i] = dict()

                  if i == 0:
                        architecture[i]['conv'] = nn.Conv2d(in_channels = self.input_channel, out_channels = j, stride = self.stride_config[i], padding = self.padding_config[i], kernel_size = self.kernel_config[i])

                        architecture[i]['op_size'] = ((self.input_size[-1] - self.kernel_config[i] + 2*self.padding_config[i])//self.stride_config[i] + 1)

                  else:
                        architecture[i]['conv'] = nn.Conv2d(in_channels = self.filter_config[i-1], out_channels = j, stride = self.stride_config[i], padding = self.padding_config[i], kernel_size = self.kernel_config[i])

                        architecture[i]['op_size'] = ((architecture[i-1]['op_size'] - self.kernel_config[i] + 2*self.padding_config[i])//self.stride_config[i] + 1)


                  architecture[i]['pool'] = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0)
                  architecture[i]['op_size'] = architecture[i]['op_size']//2

            return architecture

      def generate_dense(self):

            architecture = dict()

            cnn_output_shape = self.filter_config[-1]*(self.cnn[len(self.filter_config)-1]['op_size']**2)

            for i,j in enumerate(self.dense_config):
                  
                  if i == 0:
                        architecture[i] = nn.Linear(cnn_output_shape, j) 
                  else:
                        architecture[i] = nn.Linear(self.dense_config[i-1], j)
                  
                  architecture[i+1] = nn.Linear(self.dense_config[-1], self.output_size)
                        
            return architecture

      def forward(self, x):

            for i,j in enumerate(self.cnn.keys()):
                  x = self.activations[self.conv_activations[i]](self.cnn[j]['pool'](self.cnn[j]['conv'](x)))
                  # print(x.shape)
            x = x.view(self.batch_size, -1)
            for i,j in enumerate(self.dense.keys()):
                  if i == len(self.dense.keys()) - 1:
                        x = F.softmax(self.dense[j](x), dim = -1)
                  else:
                        x = self.activations[self.dense_activation[i]](self.dense[j](x))

            return x

      def view_model_summary(self):
            print("Convolutions\n")
            for i in self.cnn.keys():
                  print(f'Block {i}')
                  print(f'Convolution : {self.cnn[i]["conv"]}')
                  print(f'Activation : {self.conv_activations[i]}')
                  print(f'Pooling :{self.cnn[i]["pool"]}')
                  print(f'Output size : {self.cnn[i]["op_size"]} X {self.cnn[i]["op_size"]} X {self.filter_config[i]}\n')
                  
            print("Dense\n")
            for i in self.dense.keys():
                  print(f'Linear : {self.dense[i]}')
                  if i == len(self.dense.keys()) - 1:
                        print(f'Activation : Softmax\n')
                  else:
                        print(f'Activation : {self.dense_activation[i]}\n')







