import torch
import torchvision
import numpy as np
from dataset import ProstateDataset
#from torchmetrics import PrecisionRecallCurve
#from sklearn.metrics import auc, precision_recall_curve
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=0,#4,
    pin_memory=True,
):
    train_ds = ProstateDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=0,#num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = ProstateDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=0,#num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

#
def predict_at_threshold(prob, threshold):
    return np.where(prob >= threshold, 1., 0.)

def precision_at_threshold(Y, prob, threshold):
    Y_hat = predict_at_threshold(prob, threshold)
    return np.sum((Y_hat == 1) & (Y == 1)) / np.sum(Y_hat)

def recall_at_threshold(Y, prob, threshold):
    Y_hat = predict_at_threshold(prob, threshold)
    return np.sum((Y_hat == 1) & (Y == 1)) / np.sum(Y)

def precision_recall_curve(Y, prob):
    unique_thresh = np.unique(prob)
    precision = [precision_at_threshold(Y, prob, t) for t in unique_thresh]
    recall = [recall_at_threshold(Y, prob, t) for t in unique_thresh]
    return precision, recall, unique_thresh
#

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    
    c = 0
    TP = 0
    FP = 0
    FN = 0
    
    with torch.no_grad():
        for x, y in loader:
            print(c)
            c +=1

            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            #
            TP += (preds * y).sum()    
            FP += ((1-y) * preds).sum()
            FN += (y * (1-preds)).sum()
            #

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    with open("Output.txt", "a") as text_file:
        #print(f"Stopped early at: {epoch} with {acc} and {acc_before}", file=text_file)
        print(
            f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}", file=text_file)
        print(f"Dice score: {dice_score/len(loader)}", file=text_file)
        print(f"TP: {TP}, FP: {FP}, FN: {FN}" , file=text_file)
        
    model.train()

    return num_correct/num_pixels*100

def get_test_vectors(loader, model, device="cuda"):
    y_ = []
    preds_ = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            y_.append(y)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            preds_.append(preds)

    return y_, preds_



def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()