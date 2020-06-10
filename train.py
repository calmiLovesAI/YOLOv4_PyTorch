import torch
import torch.optim as optim
import time


from core.yolo_v4 import YOLOv4
from configuration import Config
from data.dataset import YoloDataset, GroundTruth
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.transform import Rescale, ToTensor
from core.loss import YoloLoss
from utils.metrics import MeanMetric
from detect import detect_multiple_pictures


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # dataset
    dataset = YoloDataset(transform=transforms.Compose([
        Rescale(output_size=Config.input_size),
        ToTensor()
    ]))
    dataloader = DataLoader(dataset=dataset, batch_size=Config.batch_size, shuffle=False)
    steps_per_epoch = len(dataset) // Config.batch_size

    # model
    yolo_v4 = YOLOv4()

    load_weights_from_epoch = Config.load_weights_from_epoch
    if Config.load_weights_before_training:
        yolo_v4.load_state_dict(torch.load(Config.save_model_dir + "epoch-{}.pth".format(load_weights_from_epoch)))
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1

    yolo_v4.to(device)
    yolo_v4.train()

    # loss
    gt = GroundTruth(device=device)
    loss_object = YoloLoss(device=device)

    # optimizer
    optimizer = optim.AdamW(params=yolo_v4.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[500, 2000], gamma=0.1)

    ciou_mean = MeanMetric()
    conf_mean = MeanMetric()
    prob_mean = MeanMetric()
    total_loss_mean = MeanMetric()

    # tensorboard --logdir=runs
    writer = SummaryWriter()

    for epoch in range(load_weights_from_epoch + 1, Config.epochs):
        for step, batch_data in enumerate(dataloader):
            step_start_time = time.time()

            batch_images, batch_labels = batch_data["image"], batch_data["label"]
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = yolo_v4(batch_images)
            target = gt(labels=batch_labels)
            ciou_loss, conf_loss, prob_loss = loss_object(y_pred=outputs, y_true=target)
            total_loss = ciou_loss + conf_loss + prob_loss

            total_loss_mean.update(total_loss.item())
            ciou_mean.update(ciou_loss.item())
            conf_mean.update(conf_loss.item())
            prob_mean.update(prob_loss.item())

            total_loss.backward()
            optimizer.step()

            step_end_time = time.time()

            print("Epoch: {}/{}, step: {}/{}, total_loss: {}, "
                  "ciou_loss: {}, conf_loss: {}, prob_loss: {}, time_cost: {:.3f}s".format(epoch,
                                                                                           Config.epochs,
                                                                                           step,
                                                                                           steps_per_epoch,
                                                                                           total_loss_mean.result(),
                                                                                           ciou_mean.result(),
                                                                                           conf_mean.result(),
                                                                                           prob_mean.result(),
                                                                                           step_end_time - step_start_time))
        writer.add_scalar("total_loss", total_loss_mean.result(), epoch)
        writer.add_scalar("ciou_loss", ciou_mean.result(), epoch)
        writer.add_scalar("conf_loss", conf_mean.result(), epoch)
        writer.add_scalar("prob_loss", prob_mean.result(), epoch)

        total_loss_mean.reset()
        ciou_mean.reset()
        conf_mean.reset()
        prob_mean.reset()

        scheduler.step()

        if epoch % Config.save_frequency == 0:
            torch.save(yolo_v4.state_dict(), Config.save_model_dir + "epoch-{}.pth".format(epoch))

        if Config.test_images_during_training:
            detect_multiple_pictures(model=yolo_v4, pictures=Config.test_images_dir_list, epoch=epoch, device=device)

    writer.flush()
    writer.close()

    torch.save(yolo_v4.state_dict(), Config.save_model_dir + "saved_model.pth")





