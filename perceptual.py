import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class LPIPF(nn.Module):
    """
    LPIPF - Learned Perceptual Image Patch Features
    Based on LPIPS [Zhang et al, 2018]
    Extract a reasonable-size perceptual feature that can be used for searching
    """
    def __init__(self, net='vgg', eval_mode=True, pooling=True, normalize=False, layer_res=None):
        super(LPIPF, self).__init__()
        # Using composition instead of inheritance because of path issues...
        self.ps = lpips.LPIPS(net=net, eval_mode=eval_mode)
        assert self.ps.lpips
        assert pooling or not layer_res
        self.pooling = pooling
        self.normalize = normalize
        if layer_res:
            self.layer_res = {net: layer_res}
        else:
            self.layer_res = {'vgg': [8, 4, 2, 2, 1],
                              # 'vgg': [16, 8, 4, 4, 2],
                              # 'vgg': [32, 16, 8, 8, 4],
                              # 'vgg': [64, 32, 16, 16, 8],
                              # 'vgg': [128, 64, 32, 16, 8], # No Pooling
                              'alex': [7, 3, 3, 1, 1],
                              # 'alex': [15, 7, 7, 7, 7],
                              # 'alex': [31, 15, 7, 7, 7],  # No pooling
                              }

    def forward(self, inp):
        if self.normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            inp = 2 * inp - 1

        inp_input = self.ps.scaling_layer(inp) if self.ps.version=='0.1' else inp
        outs = self.ps.net.forward(inp_input)
        feats = []
        for kk in range(self.ps.L):
            h, w = outs[kk].shape[2], outs[kk].shape[3]
            assert h == w
            out_res = self.layer_res[self.ps.pnet_type][kk] if self.pooling else h
            if out_res > 0:
                # Reduce spatial size
                feats_kk = outs[kk] if h == out_res else F.adaptive_avg_pool2d(outs[kk], out_res)
                # Normalize and apply the learned weights
                feats_kk = self.ps.lins[kk].model[-1].weight.data.pow(0.5) * lpips.normalize_tensor(feats_kk)
                # Flatten all spatial dimensions and divide by resolution due to HW normalization in eq. 1
                feats.append(feats_kk.reshape(feats_kk.shape[0], -1) / out_res)
        # Return the concatenated feature
        return torch.cat(feats, dim=1)

    @staticmethod
    def to_np_uint16(x):
        assert x.min().item() >= 0 and x.max().item() <= 1
        return (x.cpu().numpy()*65535).astype(np.uint16)

    @staticmethod
    def to_np_float32(x):
        return x.astype(np.float32) / 65535.0


if __name__ == '__main__':
    """
    Unit testing - compare LPIPF to the original LPIPS
    """
    import argparse
    import faiss
    from torch.utils.data import DataLoader
    from data import get_dataset, get_samples
    from utils import mosaic, visualize_nns
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from embed import vgg_representation
    from imageio import imwrite

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['shapes', 'celeba', 'ffhq', 'cifar10'], default='shapes')
    parser.add_argument('--resolution', help='resized image height(=width) after cropping', type=int, default=64)
    parser.add_argument('--n_neighbors', type=int, default=10)
    parser.add_argument('--n_train', type=int, default=None)
    parser.add_argument('--n_test', type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(1)
    np.random.seed(1)

    dataset = get_dataset(args.dataset, resolution=args.resolution, size_limit=args.n_train)
    test_i = np.random.choice(len(dataset), args.n_test)


    print('Trying kNNs in image-space first')
    X = get_samples(dataset)
    res = faiss.StandardGpuResources()
    nbrs = faiss.GpuIndexFlatL2(res, X.shape[1])
    nbrs.add(X)
    D, I = nbrs.search(X[test_i], args.n_neighbors+1)
    I_Img = I.T
    print(I_Img)

    # print('First using the original LPIPS')
    loss_fn = lpips.LPIPS(net='vgg', eval_mode=True).cuda()
    test_img = torch.stack([dataset[i][0] for i in test_i]).cuda()
    loader = DataLoader(dataset, batch_size=1, num_workers=8)
    D_LPIPS = []
    for x_batch in tqdm(loader):
        x_batch = x_batch[0] if isinstance(x_batch, list) else x_batch
        d = loss_fn(test_img, x_batch.cuda(), normalize=True)
        D_LPIPS.append(d.detach().cpu().numpy().squeeze())

    D_LPIPS = np.stack(D_LPIPS)
    print(D_LPIPS.shape)
    I_LPIPS = np.argsort(D_LPIPS, axis=0)[:args.n_neighbors+1]
    print(test_i)
    print(I_LPIPS)

    print('Now finding LPIPF NNs...')
    X_LPIPF = vgg_representation(dataset)
    # res = faiss.StandardGpuResources()
    nbrs = faiss.GpuIndexFlatL2(res, X_LPIPF.shape[1])
    nbrs.add(X_LPIPF)
    D, I = nbrs.search(X_LPIPF[test_i], args.n_neighbors+1)
    I_LPIPF = I.T
    print(I_LPIPF)

    print('Measuring similarity...')
    intersections = []
    for i in range(args.n_test):
        intersections.append(len(np.intersect1d(I_LPIPS[1:, i], I_LPIPF[1:, i])))
    print(intersections)
    print(np.mean(intersections))

    img = visualize_nns(dataset, test_i, [I_Img[1:].T, I_LPIPS[1:].T, I_LPIPF[1:].T], name='Orig, LPIPS, LPIPF')
    imwrite('figures/nns.jpg', img)
    plt.show()

