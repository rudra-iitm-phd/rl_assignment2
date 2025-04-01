from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import shared
import torch
import matplotlib.pyplot as plt

class Data:
      def __init__(self, train_path:str, test_path:str, resize:tuple, batch_size:int, train_test_split:float):

            self.transform = transforms.Compose([
                             transforms.Resize(resize),
                             transforms.ToTensor()
                        ])

            self.train_path = train_path
            self.test_path = test_path

            self.dataset = datasets.ImageFolder(train_path, transform=self.transform)
            self.test_dataset = datasets.ImageFolder(test_path, transform=self.transform)

            self.classes = self.dataset.classes

            shared.classes = {i:j for i,j in enumerate(self.classes)}

            self.train_size = int(train_test_split * len(self.dataset))
            self.val_size = len(self.dataset) - self.train_size

            self.train_dataset, self.val_dataset = random_split(self.dataset, [self.train_size, self.val_size])

            self.train_dl = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
            self.val_dl = DataLoader(self.val_dataset, batch_size = batch_size, shuffle = True)
            self.test_dl = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = True)


      def get_train_val_test_dataloaders(self):
            return self.train_dl, self.val_dl, self.test_dl

      def show_sample(self, x:torch.tensor, y:torch.tensor):

            x = x.permute(1, 2, 0)
            y = y.item()

            plt.figure()
            plt.imshow(x)
            plt.title(shared.classes[y])
            plt.show()






            

            






