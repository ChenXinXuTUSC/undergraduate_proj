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
    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    log_dir = f"./log/{timestamp}"
    tfxw = SummaryWriter(log_dir=log_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = models.mapper.Mapper(64, 3, [128, 256, 512, 128, 64])
    classifier.to(device)
    train_loader = torch.utils.data.DataLoader(
        datasets.train_data.MatchingFCGF(
            "./data",
            16
        ),
        batch_size=8,
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(classifier.parameters(), 1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    lossfn = models.metric.contrastive.ContrastiveLoss()
    for epoch in range(1, 101):
        for iter, (matches, labels) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc=f"{utils.redd('train')}"):
            matches = matches.to(device)
            labels = labels.to(device)
            matches = matches.transpose(1, 2)
            output = classifier(matches).transpose(1, 2).reshape(-1, 3)
            labels = labels.reshape(-1, 1)
            loss = lossfn(output, labels, centeralized=False)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iter % 10 == 0:
                tqdm.write(str(loss.item()))
                tfxw.add_scalar(tag="train/loss", scalar_value=loss.item(), global_step=epoch*len(train_loader) + iter)
                tfxw.add_figure(
                    tag="train/snapshot",
                    figure=snapshot(
                        output.detach().cpu().numpy(),
                        labels.cpu().numpy(), 
                        log_dir, "snapshot"
                    ), global_step=epoch*len(train_loader) + iter
                )
        if epoch % 10 == 0:
            try:
                torch.save(classifier.state_dict(), f"{log_dir}/weights/{epoch:03d}.pth")
            except FileNotFoundError:
                import os
                os.makedirs(f"{log_dir}/weights", mode=0o755)
                torch.save(classifier.state_dict(), f"{log_dir}/weights/{epoch:03d}.pth")
        scheduler.step()
