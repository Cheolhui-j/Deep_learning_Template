import logging
import torch

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.metrics.precision import Precision
from ignite.metrics.recall import Recall
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.contrib.handlers.tensorboard_logger import *

from eval import ROC

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

def do_train(
    cfg, 
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    loss_fn
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    scheduler = LRScheduler(scheduler)
    
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)

    # evaluator = create_supervised_evaluator(model, metrics={'accuracy':Accuracy(), 'loss':Loss(loss_fn)}, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy':Accuracy(output_transform=output_transform), 
        'precision':Precision(output_transform=output_transform), 'recall':Recall(output_transform=output_transform), 'roc' : ROC(), 'loss':Loss(loss_fn)}, device=device)
    
    checkpointer = ModelCheckpoint(output_dir, 'mnist', n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'optimizer': optimizer, 'lr_scheduler': scheduler})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)
    
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')
        
#     tb_logger = TensorboardLogger(log_dir="output/tb_logs")
    
#     tb_logger.attach_output_handler(
#         trainer,
#         event_name=Events.ITERATION_COMPLETED,
#         tag="Training",
#         output_transform=lambda loss: {"loss": loss}
#     )

#     tb_logger.attach_output_handler(
#         evaluator,
#         event_name=Events.EPOCH_COMPLETED,
#         tag="validation",
#         metric_names=["accuracy", "precision", "recall"],
#         global_step_transform=global_step_from_engine(trainer)
#     )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] LR : {} Loss: {:.2f}"
                        .format(engine.state.epoch, iter, len(train_loader), optimizer.param_groups[0]['lr'], engine.state.metrics['avg_loss']))
                        
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['loss']
        logger.info("Training Results - Epoch: {} Avg accuracy: {} Avg Loss: {}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss))

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            acc = metrics['accuracy']
            precision = metrics['precision']
            recall = metrics['recall']
            roc = metrics["roc"]
            logger.info("Validation Results - Epoch: {} Accuracy: {} Precision: {} recall: {} ROC : {}"
                        .format(engine.state.epoch, acc, precision, recall, roc)
                        )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)


