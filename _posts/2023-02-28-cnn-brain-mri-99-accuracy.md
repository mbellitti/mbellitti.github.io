---
title:  "Transfer Learning CNN for Brain Tumor MRI"
excerpt_separator: "<!--more-->"
date: 2023-02-28
categories:
  - Blog
tags:
  - jupyter 
---


# Introduction
In this post I look at the [Brain Tumor MRI dataset](https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c), a collection of brain MRI images. 

To see the interactive plots and images check out [the version](https://www.kaggle.com/code/matteobellitti/cnn-brain-mri-99-accuracy/edit) on Kaggle. You can also run and edit the notebook.

The dataset does not use the [BIDS](https://github.com/bids-standard/) standard, each directory is a class and contains a collection of JPG images, organized like this:
```
base_dir:
    class_1:
        img11.jpg
        img12.jpg
    class_2:
        img21.jpg
        img21.jpg
```

The task is to classify each image in one of 15 classes (Normal, or one of 14 different kinds of tumors). The dataset contains images obtained with three different MRI techniques: T1, T1 Contrasted (T1C+), and T2. This dataset is unusual in how those are organized: in a clinical setting a neurologist would request a T1, a T1C+, and a T2 image of the same patient, but here we are given only one of the three per patient.

I will use PyTorch, and fine tune ResNet18 on these images.

Let's import the library we need and seed the random number generators for reproducibility:


```python
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
plt.style.use("ggplot")
%config InlineBackend.figure_formats = ['svg']

import os
from torchvision.io import read_image, ImageReadMode
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset,random_split
from torchvision.utils import make_grid
from torchvision import datasets
import torchvision.transforms as tr
from pathlib import Path

torch.manual_seed(42)
```

# Create DataSet class
We could use one of the base `DataSet` classes mentioned in the [docs](http://pytorch.org/vision/main/datasets.html#base-classes-for-custom-datasets) but I prefer writing my own to be more flexible. 


```python
class BrainDataset(Dataset):
    
    def __init__(self, base_dir, transforms = None):
        """
        Arguments:
            base_dir: str
                Directory where the dataset is stored
            transforms: iterable
                Collection of transforms to apply when loading an image
        """
        
        self.base_dir = Path(base_dir)
        self.transforms = transforms
        
        self.data_dicts = []
        for img in self.base_dir.glob("*oma*/*.jp*g"):
            # Get all images of sick patients, every tumor name contains "oma" 
            img_dict = {}
            img_dict["img_path"] = img
            img_dict["label"],img_dict["img_type"] = img.parent.name.split(" ")
            self.data_dicts.append(img_dict)
            
        for img in self.base_dir.glob("*NORMAL*/*.jp*g"):
            # Get all images of normal patients    
            img_dict = {}
            img_dict["img_path"] = img
            img_dict["label"],img_dict["img_type"] = "Normal", img.parent.name.split(" ")[1]
            self.data_dicts.append(img_dict)
        
        self.data = pd.DataFrame(self.data_dicts)
        self.labels = self.data["label"]
        
        self.encoder = LabelEncoder()
        self.y = self.encoder.fit_transform(self.labels)
        self.num_classes = len(self.encoder.classes_)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        image = read_image(str(self.data.loc[idx,"img_path"]),mode=ImageReadMode.RGB) # read image into a Tensor
        # image = Image.open(self.data.loc[idx,"img_path"]).convert("RGB") # read image into a PIL Image
        label = self.y[idx]
        
        if self.transforms is not None:
            return self.transforms(image), label
        else:
            return image, label
    
    
    def display(self,idxs):
        """
        Display a few images from the dataset.
        Arguments:
            idxs: iterable
                Collection of indices of the images that will be displayed
        Returns:
            fig, axs:
                Matplotlib Figure and List(Axes) where the images are rendered
                
        
        Example: 
            BrainDataset().display([1,2])
        """
        
        img_list = [F.to_pil_image(self[idx][0]) for idx in idxs]
            
        labels = [self.labels[idx] for idx in idxs]
        
        fig,axs = plt.subplots(ncols=len(idxs),nrows=1,squeeze=False,figsize=(10,len(idxs)*16))
        
        for i,ax in enumerate(axs.flat):
            ax.imshow(img_list[i],cmap='gray')
            ax.set_title(labels[i],size=8)
            ax.grid()
            ax.axis('off')
            
        return fig,axs
    
    def _ipython_display_(self):
        # Render the underlying DataFrame when asked to display self in a notebook
        display(self.data)
```


```python
base_dir = Path('/kaggle/input/brain-tumor-mri-images-44c')
```

# Load Dataset


```python
brain_data = BrainDataset(base_dir,transforms=tr.Resize(size=224))
```


```python
len(brain_data)
```

Ok, we have all of them. Let's take a look at this dataset


```python
brain_data
```

The images are all in grayscale, reasonably centered with a black frame around them.


```python
idxs = np.random.choice(len(brain_data),size=8)
brain_data.display(idxs);
```


```python
brain_data.data.describe()
```

Unfortunately some classes are much rarer than others, identifying Gangliomas, Germinomas, and Granulomas will be challenging.


```python
px.histogram(brain_data.data,x="label")
```

# Autocropping
Some images have a large black frame around them, which carries no information; training a model without removing them will probably not damage significantly the accuracy, but it is a waste of computational power. In a few cases there is a thin white frame as the edge of the image, and that might distract the neural network from the relevant features at the center of the image.

Here is an example of an image with a black frame that should be removed


```python
img = F.to_pil_image(brain_data[2000][0])
img
```

I tried using the `getbbox()` method from PIL, but it only removes the frame if the pixels are *exactly* black, and so it does not work in our case:


```python
img.crop(img.getbbox())
```

An approach that works in this specific case is the following: since all images are roughly centered, and the skull is very bright, take a horizontal line through the center of the image and look at the intensity of color (0 = black, 1 = white). 

When the intensity goes above some threshold (say, 50% of the max intensity) we found the edge of the skull, and we crop the image. In pictures:


```python
imgT = tr.ToTensor()(img)[:,:,]
C,H,W = imgT.shape

fig,ax = plt.subplots(1,2,figsize=(10,4))

ax[0].imshow(img)
ax[0].axhline(H//2,color="red")
ax[0].grid()
ax[0].axis("off");
       
ax[1].plot(torch.sum(imgT[:,H//2,:],axis=0))
ax[1].set_xlabel("Pixel index")
ax[1].set_ylabel("Intensity")
ax[1].axhline(0.5*torch.max(torch.sum(imgT,axis=0)),color="Black",label="50% Intensity",linestyle="--")
ax[1].legend();
# plt.plot(torch.sum(imgT[:,:,W//2],axis=0))
```

This is the simplest form of an edge detector. I implemented this as a PyTorch module, so it's easy to use.


```python
# to define a custom transform inherit from nn.Module and give it a forward method
class CropToContent(torch.nn.Module):

    def __init__(self,threshold=0.1,ignore_frame_pixels=5):
        super().__init__()
        self.threshold = threshold
        self.ignore_frame_pixels = ignore_frame_pixels if ignore_frame_pixels > 0 else None

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            Tensor: Cropped image.
        """
            
        C,H,W = img.shape
        skipH = self.ignore_frame_pixels
        skipT = -self.ignore_frame_pixels if self.ignore_frame_pixels is not None else None
        
        ymax = torch.max(torch.sum(img[:,H//2,skipH:skipT],axis=0))
        xmax = torch.max(torch.sum(img[:,skipH:skipT,W//2],axis=0))
           
        bottom, top = torch.nonzero(torch.sum(img[:,H//2,skipH:skipT],axis=0) > self.threshold*ymax)[[0,-1]]
        left, right = torch.nonzero(torch.sum(img[:,skipH:skipT,W//2],axis=0) > self.threshold*xmax)[[0,-1]]
        
        return F.center_crop(img,output_size=[(right-left).item(),(top-bottom).item()])
    
    def __repr__(self):
        return f"CropToContent(threshold={self.threshold})"
```

Let's see what it does to the image above:


```python
F.to_pil_image(CropToContent(threshold=0.1,ignore_frame_pixels=1)(F.to_tensor(img)))
```

Perfect. Let's reload the dataset including this transformation, and resizing the images to $224\times224$ so they're in the right shape to be fed into ResNet18.


```python
brain_data = BrainDataset(base_dir,transforms=tr.Compose([
    CropToContent(threshold=0.1,ignore_frame_pixels=5),
    tr.Resize([224,224]),
    ]
    ))
```


```python
F.to_pil_image(brain_data[2000][0])
```

This transformation distorts the image, but the tumor is still clearly recognizable so I do not expect to degrade performance too much.

# Transfer learning CNN  
The model I'm using is ResNet18, followed by a fully connected layer to make the output have the correct shape (15 classes).


```python
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

# Using pretrained weights:
resnet = resnet18(weights="IMAGENET1K_V1")

# Using resnet as a feature extractor, don't backprop through it
# for param in resnet.parameters():
    # param.requires_grad = False
```


```python
import pytorch_lightning as pl
from tqdm.notebook import tqdm
from torchmetrics.functional import accuracy
```

To simplify the training process I'm using PyTorch Lightning, so I need to define a `pl.LightningModule` that specifies what are the training step, validation step, and the optimizer.


```python
class TumorClassifier(pl.LightningModule):
    def __init__(self,transforms):
        super().__init__()
        
        self.model = torch.nn.Sequential(
                    transforms,
                    resnet,
                    torch.nn.Linear(
                        resnet.fc.out_features,
                        brain_data.num_classes)
                    )
        

    def forward(self, data):
        # not needed for training, but convenient for inference
        return self.model(data) 
    
    def training_step(self,batch,batch_idx):
        logits = self(batch[0])
        preds = torch.argmax(model(batch[0]),axis=1)
        
        loss = torch.nn.functional.cross_entropy(logits,batch[1])
        acc = accuracy(preds,batch[1],task="multiclass",num_classes=brain_data.num_classes) 
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss
    
    def validation_step(self,batch,batch_idx):
        logits = self(batch[0])
        preds = torch.argmax(model(batch[0]),axis=1)
        
        loss = torch.nn.functional.cross_entropy(logits,batch[1])
        acc = accuracy(preds,batch[1],task="multiclass",num_classes=brain_data.num_classes) 
        
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)
        
        return loss
    
    def test_step(self,batch,batch_idx):
        logits = self(batch[0])
        preds = torch.argmax(model(batch[0]),axis=1)
        
        loss = torch.nn.functional.cross_entropy(logits,batch[1])
        acc = accuracy(preds,batch[1],task="multiclass",num_classes=brain_data.num_classes) 
        
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adadelta(self.parameters())
```

## Splitting the data
We can either use `train_test_split` from `sklearn.model_selection`, which allows for stratification, or the `torch.utils.data` native `random.split`.

The advantage of the `torch` function is that it uses the `Dataset` interface, so we don't have to convert to/from pandas `DataFrame`s or `numpy` arrays. I'll test later how important stratification is. 


```python
val_set_size = 893
train_set, valid_set = random_split(brain_data,[len(brain_data)-val_set_size,val_set_size])
```

It looks like the `Subset` class does not inherit from the custom class we wrote, so it does not have the `display` method.


```python
from torch.utils.data import DataLoader
```


```python
train_loader = DataLoader(train_set,batch_size=32,shuffle=True,num_workers=4)
valid_loader = DataLoader(valid_set,batch_size=32,shuffle=False,num_workers=4)
```

# Augmentations
Since the dataset is small, it's best to do some data augmentation. I will use small random rotations (up to 15 degrees) and random left-right reflection. I talked to a neurologist, and they told me that some tumors grow down the median line of the brain, while others grow on the side. The latter appear with equal probability on the left and right, so using the left-right reflection is fine.

PyTorch comes with a collection of transforms that preprocess the data to the format accepted by ResNet:


```python
resnet_transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()
resnet_transforms
```

Let's add the augmentation transforms I just discussed


```python
# Including augmentations
transforms = torch.nn.Sequential(
    # tr.RandomHorizontalFlip(),
    # tr.RandomAutocontrast(1),
    # tr.RandomResizedCrop()
    #tr.RandomRotation(15),
     resnet_transforms,
)

model = TumorClassifier(transforms=transforms)
```

During training I monitored the performance using Weights&Biases.


```python
# from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger()
```

I trained the model on a single GPU and saved its parameters. Let's load it and measure its accuracy on the validation set


```python
model.load_state_dict(torch.load("/kaggle/input/cnn-parameters-mri/resnet_fully_trained.pth"))
```


```python
epochs = 64

# from pytorch_lightning.profilers import PyTorchProfiler

device = "cuda" if torch.cuda.is_available() else "cpu"

#trainer = pl.Trainer(accelerator=device,
                     #max_epochs=epochs,
                     # logger=wandb_logger,
                #)

# torch.set_float32_matmul_precision('medium')

# trainer.fit(model, 
#             train_dataloaders=train_loader,
#             val_dataloaders=valid_loader,
#            )
```


```python
# torch.save(model.state_dict(),"resnet_fully_trained.pth")
```


```python
# log_dicts = trainer.test(model,dataloaders=valid_loader)
```


```python
model.eval()

predictions = []
truth = []

with torch.inference_mode():
    
    for data,true_labels in valid_loader:
        pred_labels = torch.argmax(model(data),axis=1)
        
        predictions.append(pred_labels)
        truth.append(true_labels)
```


```python
from sklearn.metrics import classification_report,ConfusionMatrixDisplay

print(classification_report(torch.cat(truth),torch.cat(predictions),target_names=brain_data.encoder.classes_))

ConfusionMatrixDisplay.from_predictions(torch.cat(truth),torch.cat(predictions),display_labels=brain_data.encoder.classes_,xticks_rotation='vertical',cmap='Blues')
plt.grid(visible=False)
```

Despite some classes having very few examples (~10), the F1 score is still 0.99, which is excellent.

# Which examples are hard to classify?


```python
model.eval()

with torch.inference_mode():
    for data,label in valid_loader:
        pred_labels = torch.argmax(model(data),axis=1)
        
        idxs = torch.nonzero(pred_labels != label).T[0]
        
        for i in idxs:
            plt.figure()
            img = data[i]
            plt.imshow(F.to_pil_image(img))
            true = brain_data.encoder.classes_[label[i]]
            pred = brain_data.encoder.classes_[pred_labels[i]]
            plt.axis("off")
            plt.grid()
            plt.title(f"True: {true} Pred: {pred}")
```

It looks like the model misidentified a few cases as Meningiomas, which is unsurprising given that it's the most common class. There were two cases in which a tumor was misclassified as Normal, which in this application is highly undesirable. To improve on that we could define a custom loss function that highly penalizes this kind of misclassification.

# Ideas for future steps
The most important limitation of this dataset is that the three kinds of MRIs are not given for the same patient. If that was the case, I would love to try a trick: treat each image as a channel (RGB) , and feed the composite image to ResNet. In the current version of the model I'm converting a grayscale image to RGB, which is wasteful.

Furthermore, talking to that same neurologist, they told me that some kinds of tumors are easily identified because they behave differently when adding contrast (T1 vs. T1C+), I wonder if a CNN would learn this property.
