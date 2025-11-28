import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#https://discuss.pytorch.org/t/manual-implementation-of-unrolled-3d-convolutions/91021/4

def alternativeConv(X, K, Xcon, Kcon, 
                    COut       = None,
                    kernelSize = (3,3,3),
                    pad        = (1,1,1),
                    stride     = (1,1,1) ):
    def unfold3d(tensor, kernelSize, pad, stride): 
        B, C, _, _, _ = tensor.shape
        # Input shape: (B, C, D, H, W)
        tensor = F.pad(tensor,
                       (pad[2], pad[2],
                        pad[1], pad[1],
                        pad[0], pad[0])
                      )
        tensor = (tensor
                  .unfold(2, size=kernelSize[0], step=stride[0])
                  .unfold(3, size=kernelSize[1], step=stride[1])
                  .unfold(4, size=kernelSize[2], step=stride[2])
                  .permute(0, 2, 3, 4, 1, 5, 6, 7)
                  .reshape(B, -1, C * np.prod(kernelSize))
                  .transpose(1, 2)
                 )
        return tensor
    
    bestFit = np.array([ 7, 17, 22, 10])
    coords = np.array([16, 22,  8])
    endCoords = np.array([19, 23, 13])
    
    B,CIn,H,W,D = X.shape
    outShape = ( (np.array([H,W,D]) - np.array(kernelSize) + 2 * np.array(pad)) / np.array(stride) ) + 1
    outShape = outShape.astype(np.int32)
    X = unfold3d(X, kernelSize, pad, stride)
    
    Xcon = unfold3d(Xcon, kernelSize, pad, stride)
    Kcon = Kcon.view(1, -1, 1)
    Xcon = Xcon * Kcon
    Ksize = Kcon.size(1)//3
    Xcon = Xcon[:, 0:Ksize, :] + Xcon[:, Ksize:Ksize*2, :] + Xcon[:, Ksize*2:Ksize*3, :]
    X = X * Xcon

    # Mask the windows with the kernel
    xShape = X.shape
    K = K.view(1, -1, 1)
    X2 = X.permute(1, 0, 2).reshape(K.size(1), -1)
    # For now we have to loop MIGHT BE SLOW
    a, b = torch.unique(X2, dim=1, return_inverse=True)
    c = torch.Tensor([torch.unique(a[a[:, i] != 0, i]).numel() for i in range(a.size(1))]).cuda()
    a = c[b]
    X = a.reshape(1, xShape[0], xShape[2]).permute(1, 0, 2)
    Y = X.view(B, COut, *outShape)

    K = K.view(COut, -1)
    #K = torch.randn(COut, CIn, *kernelSize).cuda() 
    #K = K.view(COut, -1)
    #Y = torch.matmul(K, X).view(B, COut, *outShape)
    return Y