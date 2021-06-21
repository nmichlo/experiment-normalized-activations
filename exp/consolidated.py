from flash.core.classification import ClassificationTask
from pytorch_lightning.callbacks import ProgressBar
from torch import nn
from torch import optim
from torch.nn import functional as F

from exp.data import make_image_data_module
from exp.data import MnistDataModule
from exp.nn.swish import Swish
from exp.nn.weights import init_weights
from exp.util.pl_utils import pl_quick_train

# from torchvision.models import vgg16
#    from nfnets import replace_conv, WSConv2d, ScaledStdConv2d
#     model = vgg16()
#     # replace_conv(model, WSConv2d)  # This repo's original implementation
#     replace_conv(model, ScaledStdConv2d)  # From timm


if __name__ == '__main__':

    # create the model
    model = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        Swish(),
        nn.Conv2d(32, 16, kernel_size=3, padding=1),
        Swish(),
        nn.AvgPool2d(kernel_size=2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        Swish(),
        nn.Conv2d(32, 8, kernel_size=3, padding=1),
        Swish(),
        nn.AvgPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(7*7*8, 128),
        nn.Dropout(),
        Swish(),
        nn.Linear(128, 10),
    )

    init_weights(model)

    # create the task
    classifier = ClassificationTask(
        model,
        loss_fn=F.cross_entropy,
        optimizer=optim.Adam,
        learning_rate=1e-3,
    )

    # make the data
    data = make_image_data_module(dataset='mnist', batch_size=128, normalise=True, return_labels=True)

    # train the network
    pl_quick_train(classifier, data, wandb_enabled=True, wandb_project='weights-test-2')
