import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data


from Algorithm.OCR.ocr.dataset import RawDataset, AlignCollate
from Algorithm.OCR.ocr.model import Model
from Algorithm.OCR.ocr.utils import CTCLabelConverter, AttnLabelConverter


def ocr():
    parser = argparse.ArgumentParser()
    # todo image_folder 需要识别的图片的路径
    parser.add_argument('--image_folder', default='./Algorithm/OCR/ocr/demo_image',
                        help='path to image_folder which contains text images')
    # todo workers 加载图片时的线程数
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
    # todo batch_size 训练时 一次训练图片的次数
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    # todo saved_model 测试的时候用   已经训练好的模型的路径
    parser.add_argument('--saved_model', default='./Algorithm/OCR/ocr/NoPoint-model/iter_23000.pth',
                        help="path to saved_model to evaluation")
    """ Data processing """
    # todo  以下的参数设置   训练和测试的时候要一致
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    # todo character 标签类别
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet',
                        help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = False
    cudnn.deterministic = False
    opt.num_gpu = torch.cuda.device_count()
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    # print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
    #       opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
    #       opt.SequenceModeling, opt.Prediction)

    # model = torch.nn.DataParallel(model)
    # if torch.cuda.is_available():
    #     model = model

    # load model
    # print('loading pretrained model from %s' % opt.saved_model)
    # todo  加载预训练模型
    pretrain = torch.load(opt.saved_model,map_location='cpu')
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    try:
        model.load_state_dict(pretrain)
    except:
        for k, v in pretrain.items():
            name = k[7:]
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    # print(model.parameters())
    # model.load_state_dict(weights)

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(    # todo 加载要识别的图片
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    myRes = []
    for image_tensors, image_path_list in demo_loader:
        batch_size = image_tensors.size(0)
        with torch.no_grad():
            image = image_tensors

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        # print('-' * 80)
        # print('image_path\tpredicted_labels')
        # print('-' * 80)

        for img_name, pred in zip(image_path_list, preds_str):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])
            myRes.append(pred)
    return myRes


