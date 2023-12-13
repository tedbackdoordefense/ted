import os
import numpy as np
import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18, PreActResNet34, VGG
from attack_dataloader import get_dataloader
from networks.models import Generator, NetC_MNIST
from torch.utils.tensorboard import SummaryWriter
from utils import progress_bar
import wandb


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def create_bd(victim_inputs, victim_labels, netG, netM, opt):

    bd_labels = create_labels_bd(victim_labels, opt)
    patterns = create_patterns(netG, victim_inputs)
    masks_output = create_masks_output(netM, victim_inputs)
    bd_inputs = apply_masks_to_inputs(victim_inputs, patterns, masks_output)

    return bd_inputs, bd_labels, patterns, masks_output


def filter_victim_inputs_and_targets(inputs, labels, opt):

    victim_inputs = [input for input, label in zip(
        inputs, labels) if label == opt.victim_label]
    victim_labels = [label for label in labels if label == opt.victim_label]

    if not victim_inputs:
        return torch.empty(0, *inputs.shape[1:], device=inputs.device, dtype=torch.float), victim_labels

    return torch.stack(victim_inputs), torch.stack(victim_labels)


def filter_non_victim_inputs_and_targets(inputs, labels, opt):
    non_victim_inputs = [input for input, label in zip(
        inputs, labels) if label != opt.victim_label]
    non_victim_labels = [
        label for label in labels if label != opt.victim_label]
    if not non_victim_inputs:
        return torch.empty(0, *inputs.shape[1:], device=inputs.device, dtype=torch.float), non_victim_labels
    return torch.stack(non_victim_inputs), non_victim_labels


def create_labels_bd(victim_labels, opt):
    if opt.attack_mode == "SSDT":
        bd_targets = torch.tensor([opt.target_label for _ in victim_labels])
    else:
        raise Exception(
            "{} attack mode is not implemented".format(opt.attack_mode))

    return bd_targets.to(opt.device)


def create_patterns(netG, inputs):
    patterns = netG(inputs)
    return netG.normalize_pattern(patterns)


def create_masks_output(netM, inputs):
    return netM.threshold(netM(inputs))


def apply_masks_to_inputs(inputs, patterns, masks_output):
    return inputs + (patterns - inputs) * masks_output


def create_cross(inputs1, inputs2, netG, netM, opt):
    patterns2 = netG(inputs2)
    patterns2 = netG.normalize_pattern(patterns2)
    masks_output = netM.threshold(netM(inputs2))
    inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
    return inputs_cross, patterns2, masks_output


def train_step(
        netC, netG, netM, optimizerC, optimizerG, schedulerC, schedulerG, train_dl1, train_dl2, epoch, opt, tf_writer
):
    netC.train()
    netG.train()
    print(" Training:")
    total = 0
    total_cross = 0
    total_bd = 0
    total_clean = 0

    total_correct_clean = 0
    total_cross_correct = 0
    total_bd_correct = 0

    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    criterion_div = nn.MSELoss(reduction="none")

    for batch_idx, (inputs1, labels1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
        optimizerC.zero_grad()

        inputs1, labels1 = inputs1.to(opt.device), labels1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]

        victim_inputs1, victim_labels1 = filter_victim_inputs_and_targets(
            inputs1, labels1, opt)

        if len(victim_inputs1) > 0:
            inputs_bd, targets_bd, patterns1, masks1 = create_bd(
                victim_inputs1, victim_labels1, netG, netM, opt)
            num_bd = inputs_bd.shape[0]
            num_cross = inputs_bd.shape[0]

            inputs_cross, patterns2, masks2 = create_cross(
                inputs1[num_bd: num_bd + num_cross], inputs2[num_bd: num_bd +
                                                             num_cross], netG, netM, opt
            )

            total_inputs = torch.cat(
                (inputs_bd, inputs_cross, inputs1[num_bd + num_cross:]), 0)
            total_targets = torch.cat((targets_bd, labels1[num_bd:])).long()

            preds = netC(total_inputs)
            loss_ce = criterion(preds, total_targets)

            # Calculating diversity loss
            distance_images = criterion_div(
                inputs1[:num_bd], inputs2[num_bd: num_bd + num_bd])
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(patterns1, patterns2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + opt.EPSILON)
            loss_div = torch.mean(loss_div) * opt.lambda_div

            total_loss = loss_ce + loss_div
            total_loss.backward()
            optimizerC.step()
            optimizerG.step()

            total += bs
            total_bd += num_bd
            total_cross += num_cross
            total_clean += bs - num_bd - num_cross

            total_correct_clean += torch.sum(
                torch.argmax(preds[num_bd + num_cross:],
                             dim=1) == total_targets[num_bd + num_cross:]
            )
            total_cross_correct += torch.sum(
                torch.argmax(preds[num_bd: num_bd + num_cross],
                             dim=1) == total_targets[num_bd: num_bd + num_cross]
            )
            bd_predict = torch.argmax(preds[:num_bd], dim=1)
            total_bd_correct += torch.sum(bd_predict == targets_bd)
            total_loss += loss_ce.detach() * bs
            avg_loss = total_loss / total

            acc_clean = total_correct_clean * 100.0 / total_clean
            acc_bd = total_bd_correct * 100.0 / total_bd
            acc_cross = total_cross_correct * 100.0 / total_cross
            batch_acc_bd = torch.sum(torch.argmax(
                preds[:num_bd], dim=1) == targets_bd) / num_bd * 100.0

            infor_string = "Clean Acc: {:.3f} | BD Acc: {:.3f} | Cross Acc: {:3f} | Batch BD Acc: {:.3f}".format(
                acc_clean, acc_bd, acc_cross, batch_acc_bd
            )

            progress_bar(batch_idx, len(train_dl1), infor_string)

            # Saving images for debugging

            if batch_idx == len(train_dl1) - 2:
                dir_temps = os.path.join(opt.temps, opt.dataset)
                if not os.path.exists(dir_temps):
                    os.makedirs(dir_temps)
                images = netG.denormalize_pattern(
                    torch.cat((inputs1[:num_bd], patterns1, inputs_bd), dim=2))
                file_name = "{}_{}_images.png".format(
                    opt.dataset, opt.attack_mode)
                file_path = os.path.join(dir_temps, file_name)
                torchvision.utils.save_image(
                    images, file_path, normalize=True, pad_value=1)

    if not epoch % 1:
        # Save figures (tfboard)
        tf_writer.add_scalars(
            "Accuracy/lambda_div_{}/".format(opt.lambda_div),
            {"Clean": acc_clean, "BD": acc_bd, "Cross": acc_cross},
            epoch,
        )
        wandb.log({"TrainClean": acc_clean, "TrainBD": acc_bd,
                  "TrainCross": acc_cross}, step=epoch)
        tf_writer.add_scalars(
            "Loss/lambda_div_{}".format(opt.lambda_div), {"CE": loss_ce, "Div": loss_div}, epoch)
        wandb.log({"TrainCE": loss_ce, "TrainDiv": loss_div}, step=epoch)

    schedulerC.step()
    schedulerG.step()


def eval(
        netC,
        netG,
        netM,
        optimizerC,
        optimizerG,
        schedulerC,
        schedulerG,
        test_dl1, test_dl2,
        epoch,
        best_acc_clean,
        best_acc_bd,
        best_acc_cross,
        opt,
):
    netC.eval()
    print(" Eval:")
    total = 0.0

    total_correct_clean = 0.0
    total_correct_cross = 0.0

    total_victim = 0.0
    total_correct_bd = 0.0

    total_non_victim = 0.0
    total_correct_nvt = 0.0

    for batch_idx, (inputs1, labels1), (inputs2, labels2) in zip(range(len(test_dl1)), test_dl1, test_dl2):

        with torch.no_grad():
            inputs1, labels1 = inputs1.to(opt.device), labels1.to(opt.device)
            inputs2, labels2 = inputs2.to(opt.device), labels2.to(opt.device)
            bs = inputs1.shape[0]

            victim_inputs1, victim_labels1 = filter_victim_inputs_and_targets(
                inputs1, labels1, opt)

            if len(victim_inputs1) > 0:
                inputs_bd, labels_bd, patterns1, masks1 = create_bd(
                    victim_inputs1, victim_labels1, netG, netM, opt)

                total_victim += len(victim_labels1)
                preds_bd = netC(inputs_bd)
                preds_bd_label = torch.argmax(preds_bd, 1)
                correct_bd = torch.sum(preds_bd_label == labels_bd)
                total_correct_bd += correct_bd

            preds_clean = netC(inputs1)
            correct_clean = torch.sum(torch.argmax(preds_clean, 1) == labels1)
            total_correct_clean += correct_clean

            inputs_cross, _, _ = create_cross(
                inputs1, inputs2, netG, netM, opt)
            preds_cross = netC(inputs_cross)
            correct_cross = torch.sum(torch.argmax(preds_cross, 1) == labels1)
            total_correct_cross += correct_cross

            non_victim_inputs1, non_victim_labels1 = filter_non_victim_inputs_and_targets(
                inputs1, labels1, opt)
            if len(non_victim_labels1) > 0:
                total_non_victim += len(non_victim_labels1)
                inputs_nvt, targets_nvt, _, _ = create_bd(
                    non_victim_inputs1, non_victim_labels1, netG, netM, opt)
                preds_nvt = netC(inputs_nvt)
                preds_nvt_label = torch.argmax(preds_nvt, 1)
                correct_nvt = torch.sum(preds_nvt_label == torch.tensor(
                    non_victim_labels1).to(opt.device))
                total_correct_nvt += correct_nvt

            total += bs
            avg_acc_clean = total_correct_clean * 100.0 / total
            avg_acc_cross = total_correct_cross * 100.0 / total
            avg_acc_bd = total_correct_bd * 100.0 / total_victim
            avg_acc_nvt = total_correct_nvt * 100.0 / total_non_victim
            batch_acc_bd = correct_bd / len(victim_labels1) * 100.0

            infor_string = "Clean Acc: {:.3f} | BD Acc: {:.3f} | Cross Acc: {:.3f} | NVT Acc : {:.3f} | Batch BD Acc : {:.3f}".format(
                avg_acc_clean, avg_acc_bd, avg_acc_cross, avg_acc_nvt, batch_acc_bd
            )
            progress_bar(batch_idx, len(test_dl1), infor_string)

        print("Clean Acc: {:.3f} | BD Acc: {:.3f} | Cross Acc: {:.3f} | NVT Acc : {:.3f} ".format(
            avg_acc_clean, avg_acc_bd, avg_acc_cross, avg_acc_nvt,
        ))
        wandb.log({
            "EvalCleanAcc": avg_acc_clean,
            "EvalBDAcc": avg_acc_bd,
            "EvalCrossAcc": avg_acc_cross,
            "EvalNVTAcc": avg_acc_nvt
        }, step=epoch)

    if avg_acc_clean + avg_acc_bd + avg_acc_cross > best_acc_clean + best_acc_bd + best_acc_cross:
        print(" Saving...")
        best_acc_clean = avg_acc_clean
        best_acc_bd = avg_acc_bd
        best_acc_cross = avg_acc_cross
        state_dict = {
            "netC": netC.state_dict(),
            "netG": netG.state_dict(),
            "netM": netM.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "schedulerG": schedulerG.state_dict(),
            "best_acc_clean": best_acc_clean,
            "best_acc_bd": best_acc_bd,
            "best_acc_cross": best_acc_cross,
            "epoch": epoch,
            "opt": opt,
        }
        ckpt_folder = os.path.join(
            opt.checkpoints, opt.dataset, opt.attack_mode, 'target_' + str(opt.target_label))
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(
            ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
        torch.save(state_dict, ckpt_path)
        wandb.save(ckpt_path)

    print(
        " Result: Best Clean Accuracy: {:.3f} - Best BD Accuracy: {:.3f} - Best Cross Accuracy: {:.3f} - Clean Accuracy: {:.3f}".format(
            best_acc_clean, best_acc_bd, best_acc_cross, avg_acc_clean
        )
    )
    return best_acc_clean, best_acc_bd, best_acc_cross, epoch


# -------------------------------------------------------------------------------------
def train_mask_step(netM, optimizerM, schedulerM, train_dl1, train_dl2, epoch, opt, tf_writer):
    netM.train()
    print(" Training:")
    total = 0

    total_loss = 0
    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(train_dl1)), train_dl1, train_dl2):
        optimizerM.zero_grad()

        inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)

        bs = inputs1.shape[0]
        masks1 = netM(inputs1)
        masks1, masks2 = netM.threshold(
            netM(inputs1)), netM.threshold(netM(inputs2))

        # Calculating diversity loss
        distance_images = criterion_div(inputs1, inputs2)
        distance_images = torch.mean(distance_images, dim=(1, 2, 3))
        distance_images = torch.sqrt(distance_images)

        distance_patterns = criterion_div(masks1, masks2)
        distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
        distance_patterns = torch.sqrt(distance_patterns)

        loss_div = distance_images / (distance_patterns + opt.EPSILON)
        loss_div = torch.mean(loss_div) * opt.lambda_div

        loss_norm = torch.mean(F.relu(masks1 - opt.mask_density))

        total_loss = opt.lambda_norm * loss_norm + opt.lambda_div * loss_div
        total_loss.backward()
        optimizerM.step()
        infor_string = "Mask loss: {:.4f} - Norm: {:.3f} | Diversity: {:.3f}".format(
            total_loss, loss_norm, loss_div)
        progress_bar(batch_idx, len(train_dl1), infor_string)

        # Saving images for debugging
        if batch_idx == len(train_dl1) - 2:
            dir_temps = os.path.join(opt.temps, opt.dataset, "masks")
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            path_masks = os.path.join(
                dir_temps, "{}_{}_masks.png".format(opt.dataset, opt.attack_mode))
            torchvision.utils.save_image(masks1, path_masks, pad_value=1)

    if not epoch % 10:
        tf_writer.add_scalars(
            "Loss/lambda_norm_{}".format(opt.lambda_norm), {
                "MaskNorm": loss_norm, "MaskDiv": loss_div}, epoch
        )

    schedulerM.step()


def eval_mask(netM, optimizerM, schedulerM, test_dl1, test_dl2, epoch, opt):
    netM.eval()
    print(" Eval:")
    total = 0.0

    criterion_div = nn.MSELoss(reduction="none")
    for batch_idx, (inputs1, targets1), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        with torch.no_grad():
            inputs1, targets1 = inputs1.to(opt.device), targets1.to(opt.device)
            inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
            bs = inputs1.shape[0]
            masks1, masks2 = netM.threshold(
                netM(inputs1)), netM.threshold(netM(inputs2))

            # Calculating diversity loss
            distance_images = criterion_div(inputs1, inputs2)
            distance_images = torch.mean(distance_images, dim=(1, 2, 3))
            distance_images = torch.sqrt(distance_images)

            distance_patterns = criterion_div(masks1, masks2)
            distance_patterns = torch.mean(distance_patterns, dim=(1, 2, 3))
            distance_patterns = torch.sqrt(distance_patterns)

            loss_div = distance_images / (distance_patterns + opt.EPSILON)
            loss_div = torch.mean(loss_div) * opt.lambda_div

            loss_norm = torch.mean(F.relu(masks1 - opt.mask_density))

            infor_string = "Norm: {:.3f} | Diversity: {:.3f}".format(
                loss_norm, loss_div)
            progress_bar(batch_idx, len(test_dl1), infor_string)

    state_dict = {
        "netM": netM.state_dict(),
        "optimizerM": optimizerM.state_dict(),
        "schedulerM": schedulerM.state_dict(),
        "epoch": epoch,
        "opt": opt,
    }
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset,
                               opt.attack_mode, 'target_' + str(opt.target_label), "mask")
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt_path = os.path.join(
        ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    torch.save(state_dict, ckpt_path)

    # Save to Weights & Biases
    wandb.save(ckpt_path)

    return epoch


# -------------------------------------------------------------------------------------


def train(opt):
    # Prepare model related things
    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
    elif opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43).to(opt.device)
    elif opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    elif opt.dataset == "imagenet":
        netC = VGG("VGG16").to(opt.device)
    else:
        raise Exception("Invalid dataset")

    netG = Generator(opt).to(opt.device)
    optimizerC = torch.optim.SGD(
        netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    optimizerG = torch.optim.Adam(
        netG.parameters(), opt.lr_G, betas=(0.5, 0.9))
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(
        optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(
        optimizerG, opt.schedulerG_milestones, opt.schedulerG_lambda)

    netM = Generator(opt, out_channels=1).to(opt.device)
    optimizerM = torch.optim.Adam(
        netM.parameters(), opt.lr_M, betas=(0.5, 0.9))
    schedulerM = torch.optim.lr_scheduler.MultiStepLR(
        optimizerM, opt.schedulerM_milestones, opt.schedulerM_lambda)

    # For tensorboard
    log_dir = os.path.join(opt.checkpoints, opt.dataset,
                           opt.attack_mode, 'target_' + str(opt.target_label))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tf_writer = SummaryWriter(log_dir=log_dir)

    # Continue training
    ckpt_folder = os.path.join(
        opt.checkpoints, opt.dataset, opt.attack_mode, 'target_' + str(opt.target_label))
    ckpt_path = os.path.join(
        ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    ckpt_path_mask = os.path.join(
        ckpt_folder, "mask", "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path)
        netC.load_state_dict(state_dict["netC"])
        netG.load_state_dict(state_dict["netG"])
        netM.load_state_dict(state_dict["netM"])
        epoch = state_dict["epoch"] + 1
        optimizerC.load_state_dict(state_dict["optimizerC"])
        optimizerG.load_state_dict(state_dict["optimizerG"])
        schedulerC.load_state_dict(state_dict["schedulerC"])
        schedulerG.load_state_dict(state_dict["schedulerG"])
        best_acc_clean = state_dict["best_acc_clean"]
        best_acc_bd = state_dict["best_acc_bd"]
        best_acc_cross = state_dict["best_acc_cross"]
        # opt = state_dict["opt"]
        print("Continue training")
    elif os.path.exists(ckpt_path_mask):
        state_dict = torch.load(ckpt_path_mask)
        netM.load_state_dict(state_dict["netM"])
        optimizerM.load_state_dict(state_dict["optimizerM"])
        schedulerM.load_state_dict(state_dict["schedulerM"])
        # opt = state_dict["opt"]
        best_acc_clean = 0.0
        best_acc_bd = 0.0
        best_acc_cross = 0.0
        epoch = state_dict["epoch"] + 1
        print("Continue training ---")
    else:
        # Prepare mask
        best_acc_clean = 0.0
        best_acc_bd = 0.0
        best_acc_cross = 0.0
        epoch = 1

        print("Training from scratch")

    # Prepare dataset
    train_dl1 = get_dataloader(opt, train=True)
    train_dl2 = get_dataloader(opt, train=True)
    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False)

    if epoch < 25:
        netM.train()
        for i in range(25):
            print(
                "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}  - lambda_norm: {}:".format(
                    epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div, opt.lambda_norm
                )
            )
            train_mask_step(netM, optimizerM, schedulerM,
                            train_dl1, train_dl2, epoch, opt, tf_writer)
            epoch = eval_mask(netM, optimizerM, schedulerM,
                              test_dl1, test_dl2, epoch, opt)
            epoch += 1

    netM.eval()
    netM.requires_grad_(False)

    for i in range(opt.n_iters):
        print(
            "Epoch {} - {} - {} | mask_density: {} - lambda_div: {}:".format(
                epoch, opt.dataset, opt.attack_mode, opt.mask_density, opt.lambda_div
            )
        )
        train_step(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            train_dl1,
            train_dl2,
            epoch,
            opt,
            tf_writer,
        )

        best_acc_clean, best_acc_bd, best_acc_cross, epoch = eval(
            netC,
            netG,
            netM,
            optimizerC,
            optimizerG,
            schedulerC,
            schedulerG,
            test_dl1,
            test_dl2,
            epoch,
            best_acc_clean,
            best_acc_bd,
            best_acc_cross,
            opt,
        )
        epoch += 1
        if epoch > opt.n_iters:
            break


def main(k):
    opt = config.get_arguments().parse_args()
    use_cuda = torch.cuda.is_available()
    opt.device = torch.device("cuda" if use_cuda else "cpu")

    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "imagenet":
        opt.num_classes = 100
    else:
        raise Exception("Invalid Dataset")

    opt.target_label = k
    if k == opt.num_classes - 1:
        opt.victim_label = 0
    else:
        opt.victim_label = k + 1

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "imagenet":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")
    train(opt)


if __name__ == "__main__":

    dataset_ranges = {
        "gtsrb": 43,
        "cifar10": 10,
        "mnist": 10,
        "imagenet": 100,
    }

    opt = config.get_arguments().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    for key, value in dataset_ranges.items():
        if opt.dataset == key:
            for k in range(value):
                # Train Debug
                if k == 0:
                    wandb.init(project="SSDT", name=f"{opt.dataset}_k_{k}", config=opt)
                    main(k)
                    wandb.finish()
                else:
                    break
