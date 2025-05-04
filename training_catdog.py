import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torchinfo import summary
from tqdm.auto import tqdm


if __name__ == "__main__":
    #Identify the data set
    image_path = "data/CatDog"
    train_dir = "data/CatDog/training"
    test_dir = "data/CatDog/testing"


    #transform the data
    IMAGE_WIDTH=128
    IMAGE_HEIGHT=128
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
    data_transform = transforms.Compose([
        # Resize the images to IMAGE_SIZE xIMAGE_SIZE 
        transforms.Resize(size=IMAGE_SIZE),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        # Turn the image into a torch.Tensor
        transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
    ])
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                  transform=data_transform, # transforms to perform on data (images)
                                  target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform)



    # Create DataLoaders for training and testing datasets
    NUM_WORKERS = os.cpu_count()
    train_dataloader = DataLoader(dataset=train_data, 
                                  batch_size=1, # how many samples per batch?
                                  num_workers=NUM_WORKERS,
                                  shuffle=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=1, 
                                num_workers=NUM_WORKERS, 
                                shuffle=False) # don't usually need to shuffle testing data
    
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)

    # new transforms for training and testing
    # Create training transform with TrivialAugment
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.TrivialAugmentWide(),
        transforms.ToTensor()])

    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()])
    
    train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data_augmented = datasets.ImageFolder(test_dir, transform=test_transform)

    BATCH_SIZE = 32
    torch.manual_seed(42)

    train_dataloader_augmented = DataLoader(train_data_augmented, 
                                            batch_size=BATCH_SIZE, 
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)

    test_dataloader_augmented = DataLoader(test_data_augmented, 
                                          batch_size=BATCH_SIZE, 
                                          shuffle=False, 
                                          num_workers=NUM_WORKERS)


    # Check if CUDA is available and set the device accordingly
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    
    class ImageClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layer_1 = nn.Sequential(
              nn.Conv2d(3, 64, 3, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(64),
              nn.MaxPool2d(2))
            self.conv_layer_2 = nn.Sequential(
              nn.Conv2d(64, 512, 3, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(512),
              nn.MaxPool2d(2))
            self.conv_layer_3 = nn.Sequential(
              nn.Conv2d(512, 512, kernel_size=3, padding=1),
              nn.ReLU(),
              nn.BatchNorm2d(512),
              nn.MaxPool2d(2)) 
            self.classifier = nn.Sequential(
              nn.Flatten(),
              nn.Linear(in_features=512*3*3, out_features=2))
        def forward(self, x: torch.Tensor):
            x = self.conv_layer_1(x)
            x = self.conv_layer_2(x)
            x = self.conv_layer_3(x)
            x = self.conv_layer_3(x)
            x = self.conv_layer_3(x)
            x = self.conv_layer_3(x)
            x = self.classifier(x)
            return x
    # Instantiate an object.
    model = ImageClassifier().to(device)
    summary(model, input_size=[1, 3, IMAGE_WIDTH ,IMAGE_HEIGHT]) 

    # Traing the model
    def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer):
      # Put model in train mode
      model.train()
      
      # Setup train loss and train accuracy values
      train_loss, train_acc = 0, 0
      
      # Loop through data loader data batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)
          
          # 1. Forward pass
          y_pred = model(X)

          # 2. Calculate  and accumulate loss
          loss = loss_fn(y_pred, y)
          train_loss += loss.item() 

          # 3. Optimizer zero grad
          optimizer.zero_grad()

          # 4. Loss backward
          loss.backward()

          # 5. Optimizer step
          optimizer.step()

          # Calculate and accumulate accuracy metric across all batches
          y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
          train_acc += (y_pred_class == y).sum().item()/len(y_pred)

      # Adjust metrics to get average loss and accuracy per batch 
      train_loss = train_loss / len(dataloader)
      train_acc = train_acc / len(dataloader)
      return train_loss, train_acc

    
    NUM_EPOCHS = 25
    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    for epoch in tqdm(range(NUM_EPOCHS)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader_augmented,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer)
      
    #model prediction
    img_batch, label_batch = next(iter(test_data_augmented))

    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    model.eval()
    with torch.inference_mode():
      pred = model(img_single.to(device))

    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")
    