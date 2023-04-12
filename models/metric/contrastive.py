import torch

class ContrastiveLoss:
    def __call__(
        self,
        manifold_coords: torch.Tensor,
        labels: torch.Tensor,
        centeralized: bool=False
    ):
        '''compute contrastive loss
        perform correlation computation between samples
        |(x1,x1) (x1,x2) (x1,x3) ... (x1,xn)|
        |(x2,x1) (x2,x2) (x2,x3) ... (x2,xn)|
        |   .       .       .           .   |
        |   .       .       .           .   |
        |(xn,x1) (xn,x2) (xn,x3) ... (xn,xn)|
        '''
        # this makes each sample compare to every other
        # sample in the same batch, if t hey  have  the
        # same label, mask at this position  should  be
        # true.
        
        # (n,c)=>(n,1,c) and (n,c)=>(1,n,c), operations
        # between (1,n,c) and  (n,1,c)  will  boradcast
        # automatically.
        num_samples = len(labels)
        eq_mask = labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1) # torch.BoolTensor
        mutual1 = manifold_coords.unsqueeze(dim=0).repeat(num_samples, 1, 1)
        mutual2 = manifold_coords.unsqueeze(dim=1).repeat(1, num_samples, 1)

        mutual_dists = torch.norm((mutual1 - mutual2), dim=-1, p=2, keepdim=True)

        loss_same = torch.sum(mutual_dists[eq_mask] ** 2)
        loss_diff = torch.sum(torch.clamp(1.0 - mutual_dists[~eq_mask], min=0) ** 2)
        
        # loss is contributed by distance of every sample
        # don't forget to redistribute it.
        ctaloss  = (loss_same + loss_diff) / (num_samples * (num_samples - 1))
        if centeralized:
            distloss = (manifold_coords ** 2).sum(dim=1).sqrt().mean()   # distant converge
            centloss = ((manifold_coords.mean(dim=1)) ** 2).sum().sqrt() # center to origin
            ctaloss = ctaloss + distloss * 1e-2 + centloss * 1e-3
        return ctaloss
