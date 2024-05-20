"""Code template for training a model on the ETHMugs dataset."""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from eth_mugs_dataset import ETHMugsDataset
from utils import IMAGE_SIZE, compute_iou


def build_model():  # TODO: Add your model definition here
    """Build the model."""


def train(
    ckpt_dir: str,
    train_data_root: str,
    val_data_root: str,
):
    """Train function."""
    # Logging and validation settings
    log_frequency = 10
    val_batch_size = 1
    val_frequency = 10

    # TODO: Set your own values for the hyperparameters
    num_epochs = 50
    # lr = 1e-4
    train_batch_size = 8
    shuffle = True
    # val_batch_size = 1
    # ...

    print(f"[INFO]: Number of training epochs: {num_epochs}")
    # print(f"[INFO]: Image scale: {image_scale}")
    # print(f"[INFO]: Learning rate: {lr}")
    # print(f"[INFO]: Training batch size: {train_batch_size}")

    #set data root
    train_data_root = ".\datasets\public_test_images_378_252"

    # Choose Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # TODO: Define your Dataset and DataLoader
    # ETHMugsDataset
    train_dataset = ETHMugsDataset(train_data_root, "train")
    val_dataset = ETHMugsDataset(val_data_root, "val")
    # Data loaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, train_batch_size, shuffle
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, val_batch_size, shuffle
    )
    # train_dataset = ...
    # train_dataloader = ...
    # val_dataset = ...
    # val_dataloader = ...

    # TODO: Define you own model
    # model = build_model(...)
    # model.to(device)
    
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()

            # First convolutional layer with 32 filters
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

            # Second convolutional layer with 64 filters
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

            # Max pooling layer
            self.pool = nn.MaxPool2d(2, 2)
            


            # Fully connected layers
            self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 64 filters, 7x7 spatial size after max pooling
            self.fc2 = nn.Linear(128, num_classes)  # Output layer with 10 classes
        
        def forward(self, x):
            # Forward pass through the network
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = self.fc2(self.fc1(x))
            return x
    
    class UNet(nn.Module):
        def __init__(self, input_channels=3, num_classes=2):
            super(UNet, self).__init__()
            
            # Encoder
            self.enc1 = self.conv_block(input_channels, 64)
            self.enc2 = self.conv_block(64, 128)
            self.enc3 = self.conv_block(128, 256)
            self.enc4 = self.conv_block(256, 512)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Bottleneck
            self.bottleneck = self.conv_block(512, 1024)
            
            # Decoder
            self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.dec4 = self.conv_block(1024, 512)
            self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec3 = self.conv_block(512, 256)
            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2 = self.conv_block(256, 128)
            self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec1 = self.conv_block(128, 64)
            
            # Output layer
            self.conv_final = nn.Conv2d(64, num_classes, kernel_size=1)

        def conv_block(self, in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            return block

        def forward(self, x):
            # Encoder
            enc1 = self.enc1(x.float())
            enc2 = self.enc2(self.pool(enc1))
            enc3 = self.enc3(self.pool(enc2))
            enc4 = self.enc4(self.pool(enc3))
            
            # Bottleneck
            bottleneck = self.bottleneck(self.pool(enc4))
            
            # Decoder
            dec4 = self.upconv4(bottleneck)
            dec4 = torch.cat((dec4, enc4[:, :, :dec4.size(2), :dec4.size(3)]), dim=1)  # Adjusted tensor dimensions
            dec4 = self.dec4(dec4)
            
            dec3 = self.upconv3(dec4)
            dec3 = torch.cat((dec3, enc3[:, :, :dec3.size(2), :dec3.size(3)]), dim=1)
            dec3 = self.dec3(dec3)
            
            dec2 = self.upconv2(dec3)
            dec2 = torch.cat((dec2, enc2[:, :, :dec2.size(2), :dec2.size(3)]), dim=1)  # Adjusted tensor dimensions
            dec2 = self.dec2(dec2)
            
            dec1 = self.upconv1(dec2)
            dec1 = torch.cat((dec1, enc1[:, :, :dec1.size(2), :dec1.size(3)]), dim=1)  # Adjusted tensor dimensions
            dec1 = self.dec1(dec1)
            
            # Output
            return self.conv_final(dec1)
        
    class UNetSmall(nn.Module):
        def __init__(self, input_channels=3, num_classes=2):
            super(UNetSmall, self).__init__()
            
            # Encoder
            self.enc1 = self.conv_block(input_channels, 64)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Bottleneck
            self.bottleneck = self.conv_block(64, 128)
            
            # Decoder
            self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec1 = self.conv_block(128, 64)
            
            # Output layer
            self.conv_final = nn.Conv2d(64, num_classes, kernel_size=1)

        def conv_block(self, in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            return block

        def forward(self, x):
            # Encoder
            enc1 = self.enc1(x.float())
            
            # Bottleneck
            bottleneck = self.bottleneck(self.pool(enc1))
            
            # Decoder
            dec1 = self.upconv1(bottleneck)
            dec1 = torch.cat((dec1, enc1[:, :, :dec1.size(2), :dec1.size(3)]), dim=1)  # Adjusted tensor dimensions
            dec1 = self.dec1(dec1)
            
            # Output
            return self.conv_final(dec1)
    
    # Instantiate the model
    input_channels = 3  # e.g., RGB images
    num_classes = 2  # e.g., binary segmentation
    model = UNetSmall(input_channels=input_channels, num_classes=num_classes)
    model.to(device)
    
    # Print the model architecture
    print(model)

    # TODO: Define Loss function
    # criterion = ...
    def dice_loss(pred, target, smooth=1.):
        pred = pred.contiguous()
        target = target.contiguous()    
        
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
        return 1 - loss.mean()

    class CombinedLoss(nn.Module):
        def __init__(self):
            super(CombinedLoss, self).__init__()
            self.cross_entropy_loss = nn.CrossEntropyLoss()

        def forward(self, inputs, targets):
            ce_loss = self.cross_entropy_loss(inputs, targets)
            dice = dice_loss(inputs, targets)
            return ce_loss + dice
    
    criterion = nn.BCELoss()

    # TODO: Define Optimizer
    # optimizer = ...
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # TODO: Define Learning rate scheduler if needed
    # lr_scheduler = ...

    # TODO: Write the training loop!
    print("[INFO]: Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for image, gt_mask in train_dataloader:
            image = image.to(device)
            gt_mask = gt_mask.squeeze(1).to(device)  # Remove singleton dimension
            gt_mask = gt_mask.long()  # Convert gt_mask to Long type

            optimizer.zero_grad()

            # Forward pass
            # output = model(image ...)
            outputs = model(image)

            # loss = criterion(output ...)
            loss = criterion(outputs, gt_mask)  # Adjusted tensor dimensions

            # Backward pass
            loss.backward()
            optimizer.step()
            # lr_scheduler.step()

            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader):.4f}')

        # Save model
        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_epoch.pth"))

        if epoch % val_frequency == 0:
            model.eval()

            val_iou = 0.0
            with torch.no_grad():
                for val_image, val_gt_mask in val_dataloader:
                    val_image = val_image.to(device)
                    val_gt_mask = val_gt_mask.to(device)

                    # Forward pass
                    # output = model(image ...)
                    val_outputs = model(val_image)

                    # val_iou += compute_iou(...)
                    val_iou += compute_iou(val_outputs, val_gt_mask)


                val_iou /= len(val_dataloader)

                val_iou *= 100

                print(f"[INFO]: Validation IoU: {val_iou.item():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SML Project 2.")
    parser.add_argument(
        "-d",
        "--data_root",
        default="./datasets",
        help="Path to the datasets folder.",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="Path to save the model checkpoints to.",
    )
    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d-%H-%M-%S")
    ckpt_dir = os.path.join(args.ckpt_dir, dt_string)
    os.makedirs(ckpt_dir, exist_ok=True)
    print("[INFO]: Model checkpoints will be saved to:", ckpt_dir)

    # Set data root
    train_data_root = os.path.join(args.data_root, "train_images_378_252")
    print(f"[INFO]: Train data root: {train_data_root}")

    val_data_root = os.path.join(args.data_root, "public_test_images_378_252")
    print(f"[INFO]: Validation data root: {val_data_root}")

    train(ckpt_dir, train_data_root, val_data_root)
