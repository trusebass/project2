"""Code template for training a model on the ETHMugs dataset."""

import argparse
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from eth_mugs_dataset import ETHMugsDataset
from utils import IMAGE_SIZE, compute_iou


def build_model(model_name):  # TODO: Add your model definition here
    """Build the model."""
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
        def __init__(self, input_channels=3, num_classes=1):
            super(UNet, self).__init__()
            
            # Encoder
            self.enc1 = self.conv_block(input_channels, 32)
            self.enc2 = self.conv_block(32, 64)
            self.enc3 = self.conv_block(64, 128)
            self.enc4 = self.conv_block(128, 256)
            #self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            
            # Bottleneck
            self.bottleneck = self.conv_block(256, 512)
            
            # Decoder
            self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec4 = self.conv_block(512, 256)
            self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3 = self.conv_block(256, 128)
            self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = self.conv_block(128, 64)
            self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.dec1 = self.conv_block(64, 32)
            
            # Output layer 
            self.conv_final = nn.Conv2d(32, num_classes, kernel_size=1)

        def conv_block(self, in_channels, out_channels):
            block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                #nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            return block

        def forward(self, x):
            # Encoder
            enc1 = self.enc1(x)
            enc2 = self.enc2(F.max_pool2d(enc1, 2))
            enc3 = self.enc3(F.max_pool2d(enc2, 2))
            enc4 = self.enc4(F.max_pool2d(enc3, 2))
            
            # Bottleneck
            bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
            
            # Decoder
            dec4 = self.upconv4(bottleneck)
            dec4 = self.center_crop_and_concat(enc4, dec4)
            dec4 = self.dec4(dec4)
            
            dec3 = self.upconv3(dec4)
            dec3 = self.center_crop_and_concat(enc3, dec3)
            dec3 = self.dec3(dec3)
            
            dec2 = self.upconv2(dec3)
            dec2 = self.center_crop_and_concat(enc2, dec2)
            dec2 = self.dec2(dec2)
            
            dec1 = self.upconv1(dec2)
            dec1 = self.center_crop_and_concat(enc1, dec1)
            dec1 = self.dec1(dec1)
            
            # Output
            #return self.conv_final(dec1)
            return torch.sigmoid(self.final(dec1))
        
        def center_crop_and_concat(self, enc, dec):
            # Find the target size that matches the decoder size
            target_size = dec.size()[2:]
            enc_size = enc.size()[2:]

            # Calculate the crop coordinates
            crop_start = [(enc_size[i] - target_size[i]) // 2 for i in range(len(target_size))]
            crop_end = [crop_start[i] + target_size[i] for i in range(len(target_size))]

            # Crop the encoder output
            enc_cropped = enc[:, :, crop_start[0]:crop_end[0], crop_start[1]:crop_end[1]]

            return torch.cat((enc_cropped, dec), dim=1)
        
    class UNetSmall(nn.Module):
        def __init__(self, input_channels=3, num_classes=1):
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

    if model_name == "CNNModel":
        return CNNModel()
    elif model_name == "UNet":
        return UNet()
    elif model_name == "UNetSmall":
        return UNetSmall()

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
    lr = 1e-4
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
    
    
    
    # Instantiate the model
    input_channels = 3  # e.g., RGB images
    num_classes = 1  # e.g., binary segmentation
    #model = UNetSmall(input_channels=input_channels, num_classes=num_classes)
    model = build_model("UNetSmall")
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
    
    class DiceLoss(torch.nn.Module):
        def __init__(self):
            super(DiceLoss, self).__init__()

        def forward(self, inputs, targets, smooth=1):
            # comment out if your model contains a sigmoid or equivalent activation layer
            inputs = torch.sigmoid(inputs)

            # flatten label and prediction tensors
            inputs = inputs.view(-1)
            targets = targets.view(-1)

            intersection = (inputs * targets).sum()
            dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

            return 1 - dice
        
    
    
    
    criterion = DiceLoss()

    # TODO: Define Optimizer
    # optimizer = ...
    optimizer = optim.Adam(model.parameters(), lr)

    # TODO: Define Learning rate scheduler if needed
    # lr_scheduler = ...
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # TODO: Write the training loop!
    best_val_iou = 0.0  # Track the best validation IoU
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
            
            # Stellen Sie sicher, dass die Ausgaben und Ziele die gleiche Form haben
            #if outputs.shape != gt_mask.shape:
                #outputs = nn.functinoal.interpolate(outputs, size=gt_mask.shape[2:], mode="bilinear", align_corners=False)
            outputs = torch.squeeze(outputs)
            # loss = criterion(output ...)
            loss = criterion()

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
                
                if val_iou > best_val_iou:
                    best_val_iou = val_iou
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"best_model.pth"))
                    print(f"[INFO]: Best model saved with IoU: {val_iou.item():.2f}")


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
