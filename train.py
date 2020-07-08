import torch
import torch.optim as optim
import time


from core.yolo_v4 import YOLOv4
from configuration import Config
from data.dataset import YoloDataset, GroundTruth
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from data.transform import Rescale, ToTensor, ColorTransform
from core.loss import YoloLoss
from utils.metrics import MeanMetric
from detect import detect_multiple_pictures



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)

    # dataset
    train_dataset = YoloDataset(annotation_dir=Config.train_txt, transform=transforms.Compose([
        ColorTransform(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        Rescale(output_size=Config.input_size),
        ToTensor()
    ]))
    valid_dataset = YoloDataset(annotation_dir=Config.valid_txt, transform=transforms.Compose([
        Rescale(output_size=Config.input_size),
        ToTensor()
    ]))

    train_loader = DataLoader(dataset=train_dataset, batch_size=Config.batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=Config.batch_size, shuffle=False)
    steps_per_epoch = len(train_dataset) // Config.batch_size

    # model
    yolo_v4 = YOLOv4()

    load_weights_from_epoch = Config.load_weights_from_epoch
    if Config.load_weights_before_training:
        yolo_v4.load_state_dict(torch.load(Config.save_model_dir + "epoch-{}.pth".format(load_weights_from_epoch)))
        print("Successfully load weights!")
    else:
        load_weights_from_epoch = -1

    yolo_v4.to(device)

    # loss
    gt = GroundTruth(device=device)
    loss_object = YoloLoss(device=device)

    # optimizer
    optimizer = optim.AdamW(params=yolo_v4.parameters(), lr=Config.initial_learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=Config.MultiStepLR_milestones, gamma=0.1)

    ciou_mean = MeanMetric()
    conf_mean = MeanMetric()
    prob_mean = MeanMetric()
    total_loss_mean = MeanMetric()

    # tensorboard --logdir=runs
    writer = SummaryWriter()

    for epoch in range(load_weights_from_epoch + 1, Config.epochs):
        # train
        yolo_v4.train()
        for step, train_data in enumerate(train_loader):
            step_start_time = time.time()

            train_images, train_labels = train_data["image"], train_data["label"]
            train_images, train_labels = train_images.to(device), train_labels.to(device)

            optimizer.zero_grad()
            outputs = yolo_v4(train_images)
            target = gt(labels=train_labels)
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

        # validation
        if len(valid_dataset):
            yolo_v4.eval()
            valid_loss = 0
            length = 0
            with torch.no_grad():
                for i, valid_data in enumerate(valid_loader):
                    length += 1
                    valid_images, valid_labels = valid_data["image"], valid_data["label"]
                    valid_images, valid_labels = valid_images.to(device), valid_labels.to(device)

                    outputs = yolo_v4(valid_images)
                    target = gt(labels=valid_labels)
                    ciou_loss, conf_loss, prob_loss = loss_object(y_pred=outputs, y_true=target)
                    valid_loss += ciou_loss.item() + conf_loss.item() + prob_loss.item()
            print("Epoch: {}/{}, valid set: loss: {}".format(epoch,
                                                             Config.epochs,
                                                             valid_loss / length))

        if epoch % Config.save_frequency == 0:
            torch.save(yolo_v4.state_dict(), Config.save_model_dir + "epoch-{}.pth".format(epoch))

        if Config.test_images_during_training:
            detect_multiple_pictures(model=yolo_v4, pictures=Config.test_images_dir_list, epoch=epoch, device=device)

    writer.flush()
    writer.close()

    torch.save(yolo_v4.state_dict(), Config.save_model_dir + "saved_model.pth")

    torch.save(yolo_v4, Config.save_model_dir + "entire_model.pth")
