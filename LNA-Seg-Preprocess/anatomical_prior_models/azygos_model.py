from anatomical_prior_models.azygos_network import Dense_UNet, preprocess_vessel_normalization
from anatomical_prior_models.azygos_config import config
from anatomical_prior_models.utils import InnerTransformer, sliding_window_inference

import torch
import numpy as np

class AzygosModel(object):
    def __init__(self):
        self.config = config
        self.device = self.config['device']
        self.net = Dense_UNet(
            n_channels=self.config['in_channels'],
            n_classes=self.config['out_channels']
        )
        self.net = self.net.to(self.device)

        self.net = torch.nn.DataParallel(self.net, device_ids=list(
            range(torch.cuda.device_count()))).to(self.device)
        checkpoint = torch.load(self.config['weight_path'], map_location='cpu')
        self.net.load_state_dict(checkpoint['state_dict'])

    @torch.no_grad()
    def predict(self, image: np.ndarray):
        self.net.eval()
        image = preprocess_vessel_normalization(image)
        image = InnerTransformer.ToTensor(image)
        image = InnerTransformer.AddChannel(image)
        image = InnerTransformer.AddChannel(image)
        image = image.to(self.device)

        pred = sliding_window_inference(
            inputs=image,
            roi_size=self.config['roi_size'],
            sw_batch_size=self.config['sw_batch_size'],
            predictor=self.net,
            overlap=self.config['overlap']
        )

        pred = InnerTransformer.SqueezeDim(pred)
        pred = InnerTransformer.SqueezeDim(pred)
        pred = InnerTransformer.ToNumpy(pred)
        pred = pred.round()

        torch.cuda.empty_cache()
        return pred


