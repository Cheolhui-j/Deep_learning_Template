import logging

from ignite.engine import Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall

def output_transform(output):
    thresh = 0.75
    
    gt_label = []
    pred_label = []

    index = 0

    while index < output[0].size(0):
        
        if output[1][index] == output[1][index+1]:
            gt_label.append(1) 
        else :
            gt_label.append(0)  

        if torch.sum(output[0][index] * output[0][index+1]) > thresh:
            pred_label.append(1) 
        else: 
            pred_label.append(0)

        index += 2

    return pred_label, gt_label

def inference(
    cfg, 
    model, 
    val_loader
):
    device = cfg.Test.device

    logger = logging.getLogger("Model Inference")
    logger.info("Start inference")
    evaluator = create_supervised_evaluator(model, metrics={'accuracy':Accuracy(output_transform=output_transform), 
        'precision':Precision(output_transform=output_transform), 'recall':Recall(output_transform=output_transform), 'loss':Loss(loss_fn)}, device=device)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_evaluation_results(engine):
        metrics = engine.state.metircs
        acc = metrics['acc']
        precision = metrics['precision']
        recall = metrics['recall']
        logger.info('Evaluation Results - Accuracy: {:.3f}, Precision: {:.3f}, Recall: {:.3f}'.format(acc))

    evaluator.run(val_loader)