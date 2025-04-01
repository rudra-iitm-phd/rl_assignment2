from data import Data
import shared
from argument_parser import parser, genreate_filters, generate_activations
import matplotlib.pyplot as plt
from cnn import CNN

TRAIN_PATH = '../inaturalist_12K/train'
TEST_PATH = '../inaturalist_12K/val'


if __name__ == '__main__':

      args = parser.parse_args()

      config = args.__dict__

      d = Data(train_path=TRAIN_PATH, test_path=TEST_PATH, resize=(config['resize'],config['resize']), batch_size=config['batch_size'], train_test_split=0.8)
      train_dl, val_dl, test_dl = d.get_train_val_test_dataloaders()

      x_train, y_train = next(iter(train_dl))

      x_val, y_val = next(iter(val_dl))

      x_test, y_test = next(iter(test_dl))


      # d.show_sample(x_train[10], y_train[10])

      script = {
            "input_size":x_train.shape, 
            "output_size":10, 

            "filters":genreate_filters(config['filter_initial'], config['filter_automated'], config['n_convolutions']),

            # "filters":[8, 16, 64, 128, 256], 
            "kernel_config":config['kernel'], 
            "padding_config":config['padding'], 
            "stride_config":config['stride'], 
            "conv_activation":generate_activations(config['conv_activation'], config['n_convolutions']), 

            "dense_activation":generate_activations(config['dense_activation'], config['n_dense']),
            "dense_config":config['dense'], 

            "batch_size":config['batch_size']
      }

      model = CNN(x_train.shape, 10, script['filters'], script['padding_config'], script['stride_config'], script['dense_config'], script['conv_activation'], script['dense_activation'], script['kernel_config'], script['batch_size'])
      model.view_model_summary()

      
      print(model.forward(x_train).shape)