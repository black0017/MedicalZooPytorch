from .viz import *
from lib.utils.general import prepare_input


def visualize_offline(args, epoch, model, full_volume, affine, writer):
    model.eval()
    test_loss = 0

    classes, slices, height, width = 4, 144, 192, 256

    predictions = torch.tensor([]).cpu()
    segment_map = torch.tensor([]).cpu()
    for batch_idx, input_tuple in enumerate(full_volume):
        with torch.no_grad():
            t1_path, t2_path, seg_path = input_tuple

            img_t1, img_t2, sub_segment_map = torch.tensor(np.load(t1_path), dtype=torch.float32)[None, None], \
                                              torch.tensor(np.load(t2_path), dtype=torch.float32)[None, None], \
                                              torch.tensor(
                                                  np.load(seg_path), dtype=torch.float32)[None]

            input_tensor, sub_segment_map = prepare_input(args, (img_t1, img_t2, sub_segment_map))
            input_tensor.requires_grad = False

            predicted = model(input_tensor).cpu()
            predictions = torch.cat((predictions, predicted))
            segment_map = torch.cat((segment_map, sub_segment_map.cpu()))

    predictions = predictions.view(-1, classes, slices, height, width).detach()
    segment_map = segment_map.view(-1, slices, height, width).detach()
    save_path_2d_fig = args.save + '/' + 'epoch__' + str(epoch).zfill(4) + '.png'

    create_2d_views(predictions, segment_map, epoch, writer, save_path_2d_fig)

    # TODO test save
    save_path = args.save + '/Pred_volume_epoch_' + str(epoch)
    save_3d_vol(predictions, affine, save_path)

    return test_loss


def visualize_no_overlap(args, full_volume, affine, model, epoch, dim, writer):
    """
    this function will produce NON-overlaping  sub-volumes prediction
    that produces full 3d medical image
    compare some slices with ground truth
    :param full_volume: t1, t2, segment
    :param dim: (d1,d2,d3))
    :return: 3d reconstructed volume
    """
    print(full_volume[0].shape)
    _, slices, height, width = full_volume[0].shape

    ## TODO generalize function - currently in CPU due to memory problems
    args.cuda = False
    classes = args.classes
    model = model.eval()
    if not args.cuda:
        model = model.cpu()

    input_tensor, segment_map = create_3d_subvol(args, full_volume, dim)

    sub_volumes = input_tensor.shape[0]
    predictions = torch.tensor([]).cpu()

    # TODO generalize
    for i in range(sub_volumes):
        predicted = model(input_tensor).cpu()
        predictions = torch.cat((predictions, predicted))

    predictions = predictions.view(-1, classes, slices, height, width).detach()

    print('Inference complete, shape ==', predictions.shape)

    save_path_2d_fig = args.save + '/' + 'epoch__' + str(epoch).zfill(4) + '.png'
    create_2d_views(predictions, segment_map, epoch, writer, save_path_2d_fig)

    # TODO test save
    save_path = args.save + '/Pred_volume_epoch_' + str(epoch)
    save_3d_vol(predictions, affine, save_path)
