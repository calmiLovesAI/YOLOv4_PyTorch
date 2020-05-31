import torch
import torch.optim as optim
import time

from core.procedure import PostProcessing
from core.yolo_v4 import YOLOv4
from configuration import Config
from data.dataset import YoloDataset, GroundTruth
from torchvision import transforms
from torch.utils.data import DataLoader
from data.transform import Rescale, ToTensor
from core.loss import YoloLoss
from utils.metrics import MeanMetric


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # dataset
    dataset = YoloDataset(transform=transforms.Compose([
        Rescale(output_size=Config.input_size),
        ToTensor()
    ]))
    dataloader = DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=True)
    steps_per_epoch = len(dataset) // Config.batch_size

    # model
    yolo_v4 = YOLOv4()
    yolo_v4.to(device)
    yolo_v4.train()

    # loss
    gt = GroundTruth()
    loss_object = YoloLoss()

    # optimizer
    optimizer = optim.AdamW(params=yolo_v4.parameters(), lr=1e-5)

    giou_mean = MeanMetric()
    conf_mean = MeanMetric()
    prob_mean = MeanMetric()
    total_loss_mean = MeanMetric()

    for epoch in range(Config.epochs):
        for step, batch_data in enumerate(dataloader):
            step_start_time = time.time()

            batch_images, batch_labels = batch_data["image"], batch_data["label"]
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = yolo_v4(batch_images)
            preds = PostProcessing.training_procedure(outputs)
            target = gt(labels=batch_labels)
            giou_loss, conf_loss, prob_loss = loss_object(y_pred=preds, y_true=target, yolo_outputs=outputs)
            total_loss = giou_loss + conf_loss + prob_loss

            total_loss_mean.update(total_loss)
            giou_mean.update(giou_loss)
            conf_mean.update(conf_loss)
            prob_mean.update(prob_loss)

            total_loss.backward()
            optimizer.step()

            step_end_time = time.time()

            print("Epoch: {}/{}, step: {}/{}, total_loss: {}, "
                  "giou_loss: {}, conf_loss: {}, prob_loss: {}, time_cost: {:.3f}s".format(epoch,
                                                                                           Config.epochs,
                                                                                           step,
                                                                                           steps_per_epoch,
                                                                                           total_loss_mean.result(),
                                                                                           giou_mean.result(),
                                                                                           conf_mean.result(),
                                                                                           prob_mean.result(),
                                                                                           step_end_time - step_start_time))

        total_loss_mean.reset()
        giou_mean.reset()
        conf_mean.reset()
        prob_mean.reset()

    torch.save(yolo_v4.state_dict(), Config.save_model_dir + "saved_model.pth")





