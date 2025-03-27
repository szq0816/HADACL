import math
import sys
from typing import Iterable
from graph_adjacency import *
from sklearn.cluster import KMeans
import utils
from utils import adjust_learning_config, SmoothedValue, MetricLogger


def train_one_epoch(model: torch.nn.Module,
                    data_loader_train: Iterable, data_loader_test: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    state_logger=None,
                    args=None):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50
    if args.print_this_epoch:
        data_loader = enumerate(metric_logger.log_every(data_loader_train, print_freq, header))
    else:
        data_loader = enumerate(data_loader_train)

    model.train(True)
    optimizer.zero_grad()


    for data_iter_step, (ids, samples, mask, _) in data_loader:
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        # print(smooth_epoch)
        lr = adjust_learning_config(optimizer, smooth_epoch, args)
        mmt = args.momentum
        # adj1, _, adj2, _ = get_miss_adjacency(samples[0], samples[1], mask, samples.shape[0], topk=10)
        # adj = [adj1.to(device, non_blocking=True), adj2.to(device, non_blocking=True)]

        for i in range(args.n_views):
            samples[i] = samples[i].to(device, non_blocking=True)

        with torch.autocast('cuda', enabled=False):
            loss = model(samples, mmt, mask, epoch < args.start_rectify_epoch, args)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.print_this_epoch:
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)

    # gather the stats from all processes
    if args.print_this_epoch or epoch == 0:
        # print("Averaged stats:", metric_logger)
        eval_result = evaluate(model, data_loader_test, device, epoch, args)
    else:
        eval_result = None
    return eval_result


def evaluate(model: torch.nn.Module, data_loader_test: Iterable,
             device: torch.device, epoch: int,
             args=None):
    model.eval()
    extracter = model.extract_feature
    with torch.no_grad():
        features_all = torch.zeros(args.n_views, args.n_sample, args.embed_dim).to(device)
        labels_all = torch.zeros(args.n_sample, dtype=torch.long).to(device)
        for indexs, samples, mask, labels in data_loader_test:
            for i in range(args.n_views):
                samples[i] = samples[i].to(device, non_blocking=True)

            labels = labels.to(device, non_blocking=True)
            features = extracter(samples, mask)

            for i in range(args.n_views):
                features_all[i][indexs] = features[i]

            labels_all[indexs] = labels

        features_cat = features_all.permute(1, 0, 2).reshape(args.n_sample, -1)
        features_cat = torch.nn.functional.normalize(features_cat, dim=-1).cpu().numpy()
        kmeans_label = KMeans(n_clusters=args.n_classes, random_state=0).fit_predict(features_cat)

        # if args.dataset == 'aloideep3v':
        #     if epoch+1 == args.epochs:
        #     # if epoch == 0 or epoch+1 == 300 or epoch+1 == args.epochs:
        #         import scipy.io as scio
        #         scio.savemat('./data/cluster/' + str(args.dataset) + '_' + str(args.missing_rate) + '_' + str(epoch) + '.mat',
        #                      {'Z': features_cat})
        #
        # elif args.dataset == 'NoisyMNIST':
        #     if epoch == 0 or epoch+1 == 60 or epoch+1 == args.epochs:
        #         import scipy.io as scio
        #         scio.savemat('./data/cluster/' + args.dataset + '_' + str(epoch) + '.mat',
        #                      {'Z': features_cat})
        #
        # elif args.dataset == 'Scene15':
        #     if epoch == 0 or epoch+1 == 150 or epoch+1 == args.epochs:
        #         import scipy.io as scio
        #         scio.savemat('./data/cluster/' + args.dataset + '_' + str(epoch) + '.mat',
        #                      {'Z': features_cat})
        #
        # elif args.dataset == 'MNISTUSPS':
        #     if epoch == 0 or epoch+1 == 250 or epoch+1 == args.epochs:
        #         import scipy.io as scio
        #         scio.savemat('./data/cluster/' + args.dataset + '_' + str(epoch) + '.mat',
        #                      {'Z': features_cat})
        #
        # elif args.dataset == 'YouTubeFace50_4Views':
        #     if epoch == 0 or epoch+1 == 100 or epoch+1 == args.epochs:
        #         import scipy.io as scio
        #         scio.savemat('./data/cluster/' + args.dataset + '_' + str(epoch) + '.mat',
        #                      {'Z': features_cat})

    nmi, ari, f, acc = utils.evaluate(np.asarray(labels_all.cpu()), kmeans_label)
    result = {'nmi': nmi, 'ari': ari, 'f': f, 'acc': acc}
    return result
