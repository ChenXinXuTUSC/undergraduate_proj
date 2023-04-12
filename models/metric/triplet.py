import torch

class TripletLoss:
    def __init__(self, margin: float=1.0) -> None:
        self.margin = margin
    
    def euclidean_distmat(self, batch: torch.Tensor):
        '''efficiently compute Euclidean distance matrix 
        
        params
        ----------
        * batch: input tensor of shape(batch_size, feature_dims)

        return
        ----------
        * Distance matrix of shape (batch_size, batch_size) 
        '''
        eps = 1e-8
        
        # step1: compute self cartesian product
        self_product = torch.mm(batch, batch.T)
        
        # step2: extract the squared euclidean distance of each
        #        sample from diagonal
        squared_diag = torch.diag(self_product)
        
        # step3: compute squared euclidean dists using the formula
        #        (a - b)^2 = a^2 - 2ab + b^2
        distmat = squared_diag.unsqueeze(dim=0) - 2*self_product + squared_diag.unsqueeze(dim=1)
        
        # get rid of negative distances due to numerical instabilities
        distmat = torch.nn.functional.relu(distmat)
        
        # step4: take the squared root of distance matrix and handle
        #        the numerical instabilities
        mask = (distmat == 0.0).float()
        
        # use the zero-mask to set those zero values to epsilon
        distmat = distmat + eps * mask
        
        distmat = torch.sqrt(distmat)
        
        # undo the trick for numerical instabilities
        # do not use *= operator, as it's an inplace operation
        # which will break the backward chain on sqrt function
        distmat = distmat * (1.0 - mask)
        
        return distmat

    def valid_triplet_mask(self, labels: torch.Tensor):
        '''efficiently compute valid triplet mask
        
        params
        ----------
        * labels: labels of samples in shape(batch_size, label_dims)
        
        return
        ----------
        * mask: valid triplet mask in shape(batch_size, batch_size, 
            batch_size).
            A triplet is valid only if  labels[i]  ==  labels[j]  &&
            labels[j] != labels[k] and indices 'i', 'j', 'k' are not
            the same value.
        '''
        
        # step1: mask of unique indices
        indices_eql = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
        indices_neq = torch.logical_not(indices_eql)
        
        i_neq_j = indices_neq.unsqueeze(dim=2)
        i_neq_k = indices_neq.unsqueeze(dim=1)
        j_neq_k = indices_neq.unsqueeze(dim=0)
        
        indices_unq = torch.logical_and(i_neq_j, torch.logical_and(i_neq_k, j_neq_k))
        
        # step2: mask of valid triplet(labels[i],labels[j],labels[k])
        labels_eql = (labels.unsqueeze(dim=0) == labels.unsqueeze(dim=1)).squeeze()
        li_eql_lj = labels_eql.unsqueeze(dim=2)
        li_eql_lk = labels_eql.unsqueeze(dim=1)
        labels_vld = torch.logical_and(li_eql_lj, torch.logical_not(li_eql_lk))
        
        return torch.logical_and(indices_unq, labels_vld)

    def __call__(self, manifold_coords: torch.Tensor, labels: torch.Tensor, centeralized: bool=False):
        '''triplet loss proposed in FaceNet CVPR 2015
        
        params
        ----------
        * manifold_coords: mapping results of the original input features
        * labels: labels of each sample in the batch
        
        return
        ----------
        * triplet loss
        '''
        eps = 1e-8
        # step1: get distance matrix
        distmat = self.euclidean_distmat(manifold_coords)
        
        # step2: compute triplet loss for all possible combinations
        #        a - anchor sample
        #        p - positive sample
        #        n - negative sample
        ap_dist = distmat.unsqueeze(dim=2)
        an_dist = distmat.unsqueeze(dim=1)
        trploss = ap_dist - an_dist + self.margin
        
        # step3: filter out invalid triplet by setting their values
        #        to zero
        valid_mask = self.valid_triplet_mask(labels)
        trploss = trploss * valid_mask
        trploss = torch.nn.functional.relu(trploss)
        
        # step4: compute scalar loss value  by  averaging  positive
        #        values
        num_positive_losses = (trploss > 0.0).float().sum()
        trploss = trploss.sum() / (num_positive_losses + eps)
        if centeralized:
            distloss = (manifold_coords ** 2).sum(dim=1).sqrt().mean()   # distant converge
            centloss = ((manifold_coords.mean(dim=1)) ** 2).sum().sqrt() # center to origin
            trploss = trploss + distloss * 1e-2 + centloss * 1e-3
        
        return trploss
