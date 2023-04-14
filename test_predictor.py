import torch
import numpy as np
import models
import datasets
import utils

from tqdm import tqdm
import time
from tensorboardX import SummaryWriter


def snapshot(
        x: np.ndarray,
        y: np.ndarray,
    ):
    import matplotlib.pyplot as plt
    import matplotlib
    
    cmap = matplotlib.colormaps["plasma"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    pidx = np.nonzero(y)[0]
    nidx = np.nonzero(~y)[0]
    ax.scatter(x[pidx, 0], x[pidx, 1], x[pidx, 2], color=cmap(norm(y[pidx])), label=1)
    ax.scatter(x[nidx, 0], x[nidx, 1], x[nidx, 2], color=cmap(norm(y[nidx])), label=0)
    ax.title.set_text("snapshot")
    ax.legend(loc="upper right")
    
    return fig


if __name__ == "__main__":
    test_loader = torch.utils.data.DataLoader(
        datasets.train_data.MatchingFCGF(
            "./data",
            16
        ),
        num_workers=2,
        batch_size=8,
        shuffle=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier = models.inlier_proposal.mapper.Mapper(64, 3, [128, 256, 512, 128, 64])
    classifier.load_state_dict(torch.load("log/Mapper/2023-04-12_21:09:09/weights/100.pth"))
    classifier.to(device)
    classifier.eval()
    predictor = models.inlier_proposal.predictor.Predictor(3, 1, [32, 64, 32])
    predictor.load_state_dict(torch.load("log/Predictor/2023-04-13_17:17:49/weights/100.pth"))
    predictor.to(device)
    predictor.eval()
    
    corr_totl = []
    for iter, (matches, labels) in tqdm(enumerate(test_loader), total=len(test_loader), ncols=100, desc=f"{utils.gren('test')}"):
        matches = matches.to(device)
        labels = labels.to(device)

        manifold_coords = classifier(matches.transpose(1, 2))
        confscores = predictor(manifold_coords).transpose(1, 2)
        
        confscores = confscores.reshape(1, -1).sigmoid()
        labels = labels.reshape(1, -1).int()
        pred_valid = (confscores > 0.5).int()
        
        corr_totl.append((pred_valid == labels).int().sum().item() / labels.shape[1])
    
    utils.log_info("mPR:", np.array(corr_totl).mean())
