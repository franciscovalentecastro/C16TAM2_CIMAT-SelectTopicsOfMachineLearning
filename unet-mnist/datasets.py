import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MNISTSegmentationDataset(Dataset):
    def __init__(self,mean,sigma):

        ## Load MNIST dataset 
        transform = transforms.Compose([
                                        transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                       ])

        trainset = torchvision.datasets.MNIST("", train = True,
                                                  transform = transform, download = True)

        testset = torchvision.datasets.MNIST("",  train = False,
                                                  transform = transform, download = True)

        self.mean = mean
        self.sigma = sigma
        self.samples = self.transform_dataset(trainset) + self.transform_dataset(testset)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def transform_dataset(self,dataset):
        output = [] 
        for i, data in enumerate(dataset, 0):
            inpt, labl = data

            ## Add normal noise
            noisy = inpt.clone()

            tmean = torch.ones(noisy.size()) * self.mean
            tsigma = torch.rand(noisy.size()) * self.sigma
            
            noisy += torch.normal(tmean,tsigma)

            ## Binarize image
            inpt = ((inpt > .2).type(torch.float) - .5) * 2

            output.append((noisy,inpt))

        return output