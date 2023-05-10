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
        d: int,
        title: str="snapshot"
    ):
    main_axes = utils.principle_K_components(x, d)
    x = np.dot(x, main_axes)
    import matplotlib.pyplot as plt
    import matplotlib
    
    cmap = matplotlib.colormaps["plasma"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax = fig.add_subplot(1, 1, 1)
    pidx = np.nonzero(y)[0]
    nidx = np.nonzero(~y)[0]
    if d == 2:
        pposes = (x[pidx, 0], x[pidx, 1])
        nposes = (x[nidx, 0], x[nidx, 1])
    elif d == 3:
        pposes = (x[pidx, 0], x[pidx, 1], x[pidx, 2])
        nposes = (x[nidx, 0], x[nidx, 1], x[nidx, 2])
    ax.scatter(*pposes, color=cmap(norm(y[pidx])), label=1)
    ax.scatter(*nposes, color=cmap(norm(y[nidx])), label=0)
    ax.title.set_text(title)
    ax.legend(loc="upper right")
    
    return fig

def save_state_dict(state, out_dir:str, out_name: str):
    try:
        torch.save(state, f"{out_dir}/{out_name}.pth")
    except FileNotFoundError:
        import os
        os.makedirs(out_dir, mode=0o755)
        torch.save(state, f"{out_dir}/{out_name}.pth")

if __name__ == "__main__":
    dataset_name = "ModelNet40"
    train_loader = torch.utils.data.DataLoader(
        datasets.train_data.MatchingFeats(
            f"./data/matches_{dataset_name}",
            64,
            postive_ratio=0.1,
            filter_strs=["toilet", "night_stand", "radio", "sofa", "dresser"]
        ),
        num_workers=2,
        batch_size=8,
        shuffle=True
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier = models.inlier_proposal.mapper.Mapper.conf_init(f"models/conf/mapper_{dataset_name}.yaml")
    classifier.to(device)
    classifier.eval()
    predictor = models.inlier_proposal.predicter.Predicter.conf_init(f"models/conf/predicter_{dataset_name}.yaml")
    predictor.to(device)
    predictor.train()
    
    optimizer = torch.optim.Adam(predictor.parameters(), 1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    lossfn = models.metric.bce.BalancedLoss()
    
    
    timestamp = time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())
    log_dir = f"./log/{dataset_name}/Predicter/i{predictor.in_channels}o{predictor.out_channels}/{timestamp}"
    tfxw = SummaryWriter(log_dir=log_dir)
    num_epochs = 100
    log_freq = 10
    save_freq = 25
    best_avg_loss = None
    for epoch in range(1, num_epochs + 1):
        loss_totl = 0.0
        for iter, (matches, labels) in tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, desc=f"{utils.redd(f'train epoch {epoch:3d}/{num_epochs}')}"):
            matches = matches.to(device).float()
            labels = labels.to(device)

            manifold_coords = classifier(matches.transpose(1, 2))
            confscores = predictor(manifold_coords).transpose(1, 2) # [batch, channel, num] -> [batch, num, channel]
            # lossfn contains a sigmoid layer to convert logits to probabilities
            loss = lossfn(confscores.reshape(-1, 1), labels.reshape(-1, 1))
            
            optimizer.zero_grad()
            loss.backward()
            loss_totl += loss.item()
            optimizer.step()
            
            if iter % log_freq == 0:
                tqdm.write(f"{utils.redd('loss')}: {loss.item():.4f}")
                tfxw.add_scalar(tag="train/loss", scalar_value=loss.item(), global_step=epoch*len(train_loader) + iter)
                
                with torch.no_grad():
                    x = manifold_coords.detach().transpose(1, 2).reshape(-1, classifier.num_feats()).cpu().numpy()
                    y_pred = (confscores.detach().reshape(-1, 1).sigmoid().cpu().numpy() > 0.8).astype(np.int32)
                    y_gdth = labels.detach().reshape(-1, 1).cpu().numpy()
                
                tfxw.add_figure(
                    tag="train/pred",
                    figure=snapshot(x, y_pred, 2), global_step=epoch*len(train_loader) + iter
                )
                tfxw.add_figure(
                    tag="train/gdth",
                    figure=snapshot(x, y_gdth, 2), global_step=epoch*len(train_loader) + iter
                )
        if epoch % save_freq == 0:
            save_state_dict(predictor.state_dict(), out_dir=f"{log_dir}/weight", out_name=f"{epoch:03d}")
        epoch_avg_loss = loss_totl / len(train_loader)
        if best_avg_loss is None or epoch_avg_loss < best_avg_loss:
            best_avg_loss = epoch_avg_loss
            save_state_dict(predictor.state_dict(), out_dir=f"{log_dir}/weight", out_name="best")
        scheduler.step()
