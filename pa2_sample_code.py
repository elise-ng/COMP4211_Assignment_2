'''

Data Split
Use train_dataset and eval_dataset as train / test sets

'''
from torchvision.datasets import EMNIST
from torch.utils.data import ConcatDataset, Subset
from torchvision.transforms import ToTensor, Compose
import numpy as np
    
# For convenience, show image at index in dataset
def show_image(dataset, index):
  import matplotlib.pyplot as plt
  plt.imshow(dataset[index][0][0], cmap=plt.get_cmap('gray'))

def get_datasets(split='balanced', save=False):
  download_folder = './data'
  
  transform = Compose([ToTensor()])

  dataset = ConcatDataset([EMNIST(root=download_folder, split=split, download=True, train=False, transform=transform),
                           EMNIST(root=download_folder, split=split, download=True, train=True, transform=transform)])
    
  # Ignore the code below with argument 'save'
  if save:
    random_seed = 4211 # do not change
    n_samples = len(dataset)
    eval_size = 0.2
    indices = list(range(n_samples))
    split = int(np.floor(eval_size * n_samples))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_indices, eval_indices = indices[split:], indices[:split]

    # cut to half
    train_indices = train_indices[:len(train_indices)//2]
    eval_indices = eval_indices[:len(eval_indices)//2]

    np.savez('train_test_split.npz', train=train_indices, test=eval_indices)
  
  # just use save=False for students
  # load train test split indices
  else:
    with np.load('./train_test_split.npz') as f:
      train_indices = f['train']
      eval_indices = f['test']

  train_dataset = Subset(dataset, indices=train_indices)
  eval_dataset = Subset(dataset, indices=eval_indices)
  
  return train_dataset, eval_dataset

# TODO
# 1. build your own CNN classifier with the given structure. DO NOT COPY OR USE ANY TRICK
# 2. load pretrained encoder from 'pretrained_encoder.pt' and build a CNN classifier on top of the encoder
# 3. load pretrained encoder from 'pretrained_encoder.pt' and build a Convolutional Autoencoder on top of the encoder (just need to implement decoder)
# *** Note that all the above tasks include implementation, training, analyzing, and reporting

# example main code
# each img has size (1, 28, 28) and each label is in {0, ..., 46}, a total of 47 classes
if __name__=='__main__':
  train_ds, eval_ds = get_datasets()
  
  img_index = 10
  show_image(train_ds, img_index)
  show_image(eval_ds, img_index)