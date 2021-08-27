import torch
from .utils import spatial_argmax


class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 64, 128]):
        super().__init__()

        conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 7, 2, 3), torch.nn.ReLU(True)]
        upconv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.ConvTranspose2d(h, c, 4, 2, 1),
                                     torch.nn.ReLU(True)]

        h, _conv, _upconv = 3, [], []
        for c in channels:
            _conv += conv_block(c, h)
            h = c

        for c in channels[:-3:-1]:
            _upconv += upconv_block(c, h)
            h = c

        _upconv += [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, 1, 1, 1, 0)]

        self._conv = torch.nn.Sequential(*_conv)
        self._upconv = torch.nn.Sequential(*_upconv)
        
        self._mean = torch.FloatTensor([0.4446, 0.5542, 0.6197])
        self._std = torch.FloatTensor([0.0011, 0.0019, 0.0021])


    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        img = (img - self._mean[None, :, None, None].to(img.device)) / self._std[None, :, None, None].to(img.device)
        h = self._conv(img)
        x = self._upconv(h)
        return (1 + spatial_argmax(x.squeeze(1))) * torch.as_tensor([img.size(3) - 1, img.size(2) - 1]).float().to(
            img.device)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r
