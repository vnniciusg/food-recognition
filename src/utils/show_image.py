import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imageShow(loader):
    batch = next(iter(loader))
    images,labels = batch
    grid = torchvision.utils.make_grid(images, nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid,(1,2,0)))
    print('labels: ' , labels)
    plt.show()
  