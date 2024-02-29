import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image


# =============================================================================
# from models.modeling_ViT_fixFG_CAM import VisionTransformer, CONFIGS
# =============================================================================
from models.modeling_ViT_selfdata_CAM_li import VisionTransformer, CONFIGS
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils_cover import get_loader
from utils.dist_util import get_world_size

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_false', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    
    parser.add_argument('--aug_smooth',action='store_true',  default=False,
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true', default=False,
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    
    parser.add_argument(
        '--target_category',
        type=int,
        default=None,
        help='label')
    
# =============================================================================
#     parser.add_argument('--model',type=float, default=0.5,
#                         help='0.3,0.5,0.7')
# =============================================================================

    parser.add_argument('--model',type=float, default=500,
                        help='500,700,1000')
    
    
    parser.add_argument('--pretrained_model',type=str, default=None,
                        help='')
    
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=37, width=37):
    # print('reshape_transform',tensor.shape)     # [1, 192, 14, 14] ---> [1,768,37,37]
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == '__main__':
    """ python vit_gradcam.py -image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM}
        
    print('--aug_smooth',args.aug_smooth)
    
    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
        
    model_type = 'ViT-B_16'
    img_size = 448
    num_classes = 555
    smoothing_value = 0.0
    pretrained_dir = '/data/kb/tanyuanyong/TransFG-master/data/vit_model/ViT-B_16.npz'

    
# =============================================================================
#     if args.model == 0.7:    
#         pretrained_model = '/data/kb/tanyuanyong/TransFG-master/output/ViTcover_fixFG_e2_7_checkpoint.bin'
#     else:
#         raise Exception("model error  0.5  0.7")
#     # pretrained_model = None
# =============================================================================
# =============================================================================
#     if args.model == 0.5:    
#         pretrained_model = '/data/kb/tanyuanyong/TransFG-master/output/82.2VIT_coverdog_1e2_checkpoint.bin'
#     else:
#         raise Exception("model error  0.5  0.7")
# =============================================================================
    
# =============================================================================
#     if args.model == 500:
#         pretrained_model = '/data/kb/tanyuanyong/TransFG-master/output/SEV_VIT/SEV_VIT_selfdataV1.9_3e2_500_checkpoint.bin'
#     elif args.model == 700:
#         pretrained_model = '/data/kb/tanyuanyong/TransFG-master/output/SEV_VIT/SEV_VIT_selfdataV1.9_3e2_700_checkpoint.bin'
#     elif args.model == 1000:
#         pretrained_model = '/data/kb/tanyuanyong/TransFG-master/output/SEV_VIT/SEV_VIT_selfdataV1.9_3e2_1000_checkpoint.bin'
#     else:
#         raise Exception("model error  ")
# =============================================================================

    pretrained_model = args.pretrained_model
    # pretrained_model = None
    
    print('model',args.model)
    print(pretrained_model)
    config = CONFIGS[model_type]
    config.split = 'overlap'
    config.slide_step = 12


    model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes,                                                   smoothing_value=smoothing_value)

    model.load_from(np.load(pretrained_dir))
    
    if pretrained_model is not None:
        pretrained_model = torch.load(pretrained_model)['model']
        model.load_state_dict(pretrained_model)



    model.eval()
    # print(model)
    if args.use_cuda:
        model = model.cuda()

    target_layers = target_layer = model.transformer.encoder.layer[-1].attention_norm




    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layer=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)

    rgb_img = cv2.imread(args.image_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (448, 448))
    rgb_img = np.float32(rgb_img) / 255
    '''
    input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    '''
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = None

    # AblationCAM and ScoreCAM have batched implementations.
    # You can override the internal batch size for faster computation.
    cam.batch_size = 32

    grayscale_cam = cam(input_tensor=input_tensor,
                        target_category=target_category,
                        eigen_smooth=args.eigen_smooth,
                        aug_smooth=args.aug_smooth)

    # Here grayscale_cam has only one image in the batch
    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam)
    cv2.imwrite(f'{args.method}_cam.jpg', cam_image)
