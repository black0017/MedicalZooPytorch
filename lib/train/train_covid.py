import torch
import torch.nn as nn

from lib.utils.covid_utils import MetricTracker, accuracy


def train(args, model, trainloader, optimizer, epoch, writer):
    model.train()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'accuracy']
    train_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='train')
    train_metrics.reset()

    for batch_idx, input_tensors in enumerate(trainloader):
        optimizer.zero_grad()
        input_data, target = input_tensors
        if (args.cuda):
            input_data = input_data.cuda()
            target = target.cuda()

        output = model(input_data)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        correct, total, acc = accuracy(output, target)

        num_samples = batch_idx * args.batchSz + 1
        train_metrics.update_all_metrics(batch_idx + 1, {'loss': loss.item(), 'accuracy': acc},
                                         writer_step=(epoch - 1) * len(trainloader) + batch_idx)
        partial_epoch = epoch + batch_idx / len(trainloader) - 1
        # train_metrics.display_terminal(partial_epoch,epoch,'train')
    train_metrics.display_terminal(partial_epoch, epoch, 'train', summary=True)
    return train_metrics


def validation(args, model, testloader, epoch, writer):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='mean')

    metric_ftns = ['loss', 'correct', 'accuracy']
    val_metrics = MetricTracker(*[m for m in metric_ftns], writer=writer, mode='val')
    val_metrics.reset()
    confusion_matrix = torch.zeros(args.classes, args.classes)
    with torch.no_grad():
        for batch_idx, input_tensors in enumerate(testloader):

            input_data, target = input_tensors
            if (args.cuda):
                input_data = input_data.cuda()
                target = target.cuda()

            output = model(input_data)

            loss = criterion(output, target)

            correct, total, acc = accuracy(output, target)
            num_samples = batch_idx * args.batchSz + 1
            _, preds = torch.max(output, 1)
            for t, p in zip(target.cpu().view(-1), preds.cpu().view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            val_metrics.update_all_metrics(batch_idx + 1, {'loss': loss.item(), 'accuracy': acc},
                                           writer_step=(epoch - 1) * len(testloader) + batch_idx)

            # val_metrics.display_terminal(num_samples/len(testloader),epoch,'VAL')
    val_metrics.display_terminal(num_samples / len(testloader), epoch, 'VAL', summary=True)
    print('Confusion Matrix\n{}'.format(confusion_matrix.cpu().numpy()))
    return val_metrics, confusion_matrix
