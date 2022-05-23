from .example_model import ResNet18, Lenet

def build_model(cfg):

    #model = ResNet18(cfg.MODEL.NUM_CLASSES)
    model = Lenet()

    return model