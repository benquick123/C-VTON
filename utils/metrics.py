from math import exp

import numpy as np
import torch
import torchvision
from scipy import linalg
from scipy.stats import entropy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.models.inception import inception_v3


def ssim(img1, img2, window_size = 11, size_average = True):
    
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(img1, img2, window, window_size, channel, size_average = True):
        mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
        mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    # assuming its a Dataset if not a Tensor
    if not isinstance(img1, torch.Tensor):
        img1 = torch.stack([s["image"]["I"] for s in iter(img1)], dim=0)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.stack([s["image"]["I"] for s in iter(img2)], dim=0)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval();
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=-1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader):
        if isinstance(batch, list):
            batch = batch[0]
            
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def fid(img1, img2, batch_size=50, device="cuda", dims=2048):
    
    class FIDInceptionA(torchvision.models.inception.InceptionA):
        """InceptionA block patched for FID computation"""
        def __init__(self, in_channels, pool_features):
            super(FIDInceptionA, self).__init__(in_channels, pool_features)

        def forward(self, x):
            branch1x1 = self.branch1x1(x)

            branch5x5 = self.branch5x5_1(x)
            branch5x5 = self.branch5x5_2(branch5x5)

            branch3x3dbl = self.branch3x3dbl_1(x)
            branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
            branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

            # Patch: Tensorflow's average pool does not use the padded zero's in
            # its average calculation
            branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                    count_include_pad=False)
            branch_pool = self.branch_pool(branch_pool)

            outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
            return torch.cat(outputs, 1)

    class FIDInceptionC(torchvision.models.inception.InceptionC):
        """InceptionC block patched for FID computation"""
        def __init__(self, in_channels, channels_7x7):
            super(FIDInceptionC, self).__init__(in_channels, channels_7x7)

        def forward(self, x):
            branch1x1 = self.branch1x1(x)

            branch7x7 = self.branch7x7_1(x)
            branch7x7 = self.branch7x7_2(branch7x7)
            branch7x7 = self.branch7x7_3(branch7x7)

            branch7x7dbl = self.branch7x7dbl_1(x)
            branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
            branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
            branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
            branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

            # Patch: Tensorflow's average pool does not use the padded zero's in
            # its average calculation
            branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                    count_include_pad=False)
            branch_pool = self.branch_pool(branch_pool)

            outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
            return torch.cat(outputs, 1)

    class FIDInceptionE_1(torchvision.models.inception.InceptionE):
        """First InceptionE block patched for FID computation"""
        def __init__(self, in_channels):
            super(FIDInceptionE_1, self).__init__(in_channels)

        def forward(self, x):
            branch1x1 = self.branch1x1(x)

            branch3x3 = self.branch3x3_1(x)
            branch3x3 = [
                self.branch3x3_2a(branch3x3),
                self.branch3x3_2b(branch3x3),
            ]
            branch3x3 = torch.cat(branch3x3, 1)

            branch3x3dbl = self.branch3x3dbl_1(x)
            branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
            branch3x3dbl = [
                self.branch3x3dbl_3a(branch3x3dbl),
                self.branch3x3dbl_3b(branch3x3dbl),
            ]
            branch3x3dbl = torch.cat(branch3x3dbl, 1)

            # Patch: Tensorflow's average pool does not use the padded zero's in
            # its average calculation
            branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1,
                                    count_include_pad=False)
            branch_pool = self.branch_pool(branch_pool)

            outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
            return torch.cat(outputs, 1)

    class FIDInceptionE_2(torchvision.models.inception.InceptionE):
        """Second InceptionE block patched for FID computation"""
        def __init__(self, in_channels):
            super(FIDInceptionE_2, self).__init__(in_channels)

        def forward(self, x):
            branch1x1 = self.branch1x1(x)

            branch3x3 = self.branch3x3_1(x)
            branch3x3 = [
                self.branch3x3_2a(branch3x3),
                self.branch3x3_2b(branch3x3),
            ]
            branch3x3 = torch.cat(branch3x3, 1)

            branch3x3dbl = self.branch3x3dbl_1(x)
            branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
            branch3x3dbl = [
                self.branch3x3dbl_3a(branch3x3dbl),
                self.branch3x3dbl_3b(branch3x3dbl),
            ]
            branch3x3dbl = torch.cat(branch3x3dbl, 1)

            # Patch: The FID Inception model uses max pooling instead of average
            # pooling. This is likely an error in this specific Inception
            # implementation, as other Inception models use average pooling here
            # (which matches the description in the paper).
            branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
            branch_pool = self.branch_pool(branch_pool)

            outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
            return torch.cat(outputs, 1)
    
    def _inception_v3(*args, **kwargs):
        """Wraps `torchvision.models.inception_v3`
        Skips default weight inititialization if supported by torchvision version.
        See https://github.com/mseitzer/pytorch-fid/issues/28.
        """
        try:
            version = tuple(map(int, torchvision.__version__.split('.')[:2]))
        except ValueError:
            # Just a caution against weird version strings
            version = (0,)

        if version >= (0, 6):
            kwargs['init_weights'] = False

        return torchvision.models.inception_v3(*args, **kwargs)
    
    def fid_inception_v3():
        """Build pretrained Inception model for FID computation
        The Inception model for FID computation uses a different set of weights
        and has a slightly different structure than torchvision's Inception.
        This method first constructs torchvision's Inception and then patches the
        necessary parts that are different in the FID Inception model.
        """
        inception = _inception_v3(num_classes=1008,
                                aux_logits=False,
                                pretrained=False)
        inception.Mixed_5b = FIDInceptionA(192, pool_features=32)
        inception.Mixed_5c = FIDInceptionA(256, pool_features=64)
        inception.Mixed_5d = FIDInceptionA(288, pool_features=64)
        inception.Mixed_6b = FIDInceptionC(768, channels_7x7=128)
        inception.Mixed_6c = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6d = FIDInceptionC(768, channels_7x7=160)
        inception.Mixed_6e = FIDInceptionC(768, channels_7x7=192)
        inception.Mixed_7b = FIDInceptionE_1(1280)
        inception.Mixed_7c = FIDInceptionE_2(2048)

        state_dict = load_state_dict_from_url("https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth", progress=True)
        inception.load_state_dict(state_dict)
        return inception
    
    class InceptionV3(nn.Module):
        """Pretrained InceptionV3 network returning feature maps"""

        # Index of default block of inception to return,
        # corresponds to output of final average pooling
        DEFAULT_BLOCK_INDEX = 3

        # Maps feature dimensionality to their output blocks indices
        BLOCK_INDEX_BY_DIM = {
            64: 0,   # First max pooling features
            192: 1,  # Second max pooling featurs
            768: 2,  # Pre-aux classifier features
            2048: 3  # Final average pooling features
        }

        def __init__(self,
                    output_blocks=[DEFAULT_BLOCK_INDEX],
                    resize_input=True,
                    normalize_input=True,
                    requires_grad=False,
                    use_fid_inception=True):
            """Build pretrained InceptionV3
            Parameters
            ----------
            output_blocks : list of int
                Indices of blocks to return features of. Possible values are:
                    - 0: corresponds to output of first max pooling
                    - 1: corresponds to output of second max pooling
                    - 2: corresponds to output which is fed to aux classifier
                    - 3: corresponds to output of final average pooling
            resize_input : bool
                If true, bilinearly resizes input to width and height 299 before
                feeding input to model. As the network without fully connected
                layers is fully convolutional, it should be able to handle inputs
                of arbitrary size, so resizing might not be strictly needed
            normalize_input : bool
                If true, scales the input from range (0, 1) to the range the
                pretrained Inception network expects, namely (-1, 1)
            requires_grad : bool
                If true, parameters of the model require gradients. Possibly useful
                for finetuning the network
            use_fid_inception : bool
                If true, uses the pretrained Inception model used in Tensorflow's
                FID implementation. If false, uses the pretrained Inception model
                available in torchvision. The FID Inception model has different
                weights and a slightly different structure from torchvision's
                Inception model. If you want to compute FID scores, you are
                strongly advised to set this parameter to true to get comparable
                results.
            """
            super(InceptionV3, self).__init__()

            self.resize_input = resize_input
            self.normalize_input = normalize_input
            self.output_blocks = sorted(output_blocks)
            self.last_needed_block = max(output_blocks)

            assert self.last_needed_block <= 3, \
                'Last possible output block index is 3'

            self.blocks = nn.ModuleList()

            inception = fid_inception_v3()

            # Block 0: input to maxpool1
            block0 = [
                inception.Conv2d_1a_3x3,
                inception.Conv2d_2a_3x3,
                inception.Conv2d_2b_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block0))

            # Block 1: maxpool1 to maxpool2
            if self.last_needed_block >= 1:
                block1 = [
                    inception.Conv2d_3b_1x1,
                    inception.Conv2d_4a_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2)
                ]
                self.blocks.append(nn.Sequential(*block1))

            # Block 2: maxpool2 to aux classifier
            if self.last_needed_block >= 2:
                block2 = [
                    inception.Mixed_5b,
                    inception.Mixed_5c,
                    inception.Mixed_5d,
                    inception.Mixed_6a,
                    inception.Mixed_6b,
                    inception.Mixed_6c,
                    inception.Mixed_6d,
                    inception.Mixed_6e,
                ]
                self.blocks.append(nn.Sequential(*block2))

            # Block 3: aux classifier to final avgpool
            if self.last_needed_block >= 3:
                block3 = [
                    inception.Mixed_7a,
                    inception.Mixed_7b,
                    inception.Mixed_7c,
                    nn.AdaptiveAvgPool2d(output_size=(1, 1))
                ]
                self.blocks.append(nn.Sequential(*block3))

            for param in self.parameters():
                param.requires_grad = requires_grad

        def forward(self, inp):
            """Get Inception feature maps
            Parameters
            ----------
            inp : torch.autograd.Variable
                Input tensor of shape Bx3xHxW. Values are expected to be in
                range (0, 1)
            Returns
            -------
            List of torch.autograd.Variable, corresponding to the selected output
            block, sorted ascending by index
            """
            outp = []
            x = inp

            if self.resize_input:
                x = F.interpolate(x,
                                size=(299, 299),
                                mode='bilinear',
                                align_corners=False)

            if self.normalize_input:
                x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

            for idx, block in enumerate(self.blocks):
                x = block(x)
                if idx in self.output_blocks:
                    outp.append(x)

                if idx == self.last_needed_block:
                    break

            return outp
    
    def calculate_activation_statistics(img, model, batch_size=50, dims=2048, device='cuda'):
        """Calculation of the statistics used by the FID.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : The images numpy array is split into batches with
                        batch size batch_size. A reasonable batch size
                        depends on the hardware.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        Returns:
        -- mu    : The mean over samples of the activations of the pool_3 layer of
                the inception model.
        -- sigma : The covariance matrix of the activations of the pool_3 layer of
                the inception model.
        """
        act = get_activations(img, model, batch_size, dims, device)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def get_activations(img, model, batch_size=50, dims=2048, device='cuda'):
        """Calculates the activations of the pool_3 layer for all images.
        Params:
        -- files       : List of image files paths
        -- model       : Instance of inception model
        -- batch_size  : Batch size of images for the model to process at once.
                        Make sure that the number of samples is a multiple of
                        the batch size, otherwise some samples are ignored. This
                        behavior is retained to match the original FID score
                        implementation.
        -- dims        : Dimensionality of features returned by Inception
        -- device      : Device to run calculations
        Returns:
        -- A numpy array of dimension (num images, dims) that contains the
        activations of the given tensor when feeding inception with the
        query tensor.
        """
        model.eval()

        dl = torch.utils.data.DataLoader(img, batch_size=batch_size, drop_last=False, num_workers=0)

        pred_arr = np.empty((len(img), dims))

        start_idx = 0

        for batch in dl:
            if isinstance(batch, list):
                batch = batch[0]
            batch = batch.to(device)

            with torch.no_grad():
                pred = model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()

            pred_arr[start_idx:start_idx + pred.shape[0]] = pred

            start_idx = start_idx + pred.shape[0]

        return pred_arr

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = calculate_activation_statistics(img1, model, batch_size, dims, device)
    m2, s2 = calculate_activation_statistics(img2, model, batch_size, dims, device)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value