from .example_model import ResNet18, Lenet, MobileResNet100

def build_model(cfg):

    #model = ResNet18(cfg.MODEL.NUM_CLASSES)
    #model = Lenet()
    model = MobileResNet100(cfg)

    return model