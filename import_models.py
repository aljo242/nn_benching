import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch

from models_dir.p_s_s.models import duc_hdc, fcn8s, fcn16s, fcn32s, gcn, psp_net, seg_net, u_net



def import_models(download):
    
    """
    Commented out models do not have pre-trained variants available.

    This function will load and cache all of these models if they are not already cached.

    On CRC we can just run this initially as its own python script to download and cache the models.

    Afterwards, this function will be implemented into some other testing code that will call it in 
    a similar fashion to what is shown below.
    """

    print("Loading or checking if models are cached...\n")
    print("This may take a while.  Progress will be shown if models are being downloaded.\n ")

    ############################################################################################
    # Image Classification

    alexnet = models.alexnet(pretrained=download, progress=True)

    squeezenet1_0 = models.squeezenet1_0(pretrained=download, progress=True)
    squeezenet1_1 = models.squeezenet1_1(pretrained=download, progress=True)

    vgg16 = models.vgg16_bn(pretrained=download, progress=True)
    vgg19 = models.vgg19_bn(pretrained=download, progress=True)

    resnet18 = models.resnet18(pretrained=download, progress=True)
    resnet34 = models.resnet34(pretrained=download, progress=True)
    resnet50 = models.resnet50(pretrained=download, progress=True)
    resnet101 = models.resnet101(pretrained=download, progress=True)
    resnet152 = models.resnet152(pretrained=download, progress=True)

    densenet121 = models.densenet121(pretrained=download, progress=True, memory_efficient=False)
    densenet161 = models.densenet161(pretrained=download, progress=True, memory_efficient=False)
    densenet201 = models.densenet201(pretrained=download, progress=True, memory_efficient=False)
    densenet121_efficient = models.densenet121(pretrained=download, progress=True, memory_efficient=True)
    densenet161_efficient = models.densenet161(pretrained=download, progress=True, memory_efficient=True)
    densenet201_efficient = models.densenet201(pretrained=download, progress=True, memory_efficient=True)

    googlenet = models.googlenet(pretrained=download, progress=True)

    shufflenet_v2_1 = models.shufflenet_v2_x1_0(pretrained=download, progress=True)
    shufflenet_v2_0_5 = models.shufflenet_v2_x0_5(pretrained=download, progress=True)
    #shufflenet_v2_1_5 = models.shufflenet_v2_x1_5(pretrained=download, progress=True)
    #shufflenet_v2_2 = models.shufflenet_v2_x2_0(pretrained=download, progress=True)

    mobilenet_v2 = models.mobilenet_v2(pretrained=download, progress=True)

    resnext50_32x4d = models.resnext50_32x4d(pretrained=download, progress=True)
    resnext101_32x8d = models.resnext101_32x8d(pretrained=download, progress=True)

    wide_resnet50_2 = models.wide_resnet50_2(pretrained=download, progress=True)
    wide_resnet101_2 = models.wide_resnet101_2(pretrained=download, progress=True)

    mnasnet1_0 = models.mnasnet1_0(pretrained=download, progress=True)
    mnasnet0_5 = models.mnasnet0_5(pretrained=download, progress=True)
    #mnasnet0_75 = models.mnasnet0_75(pretrained=download, progress=True)
    #mnasnet1_3 = models.mnasnet1_3(pretrained=download, progress=True)

    ###########################################################################################
    # Video Classification
    resnet_3d = models.video.r3d_18(pretrained=download, progress=True)
    resnet_mixed_conv = models.video.mc3_18(pretrained=download, progress=True)
    resnet_2_1D = models.video.r2plus1d_18(pretrained=download, progress=True)

    ###########################################################################################
    # Object Detection

    fasterrcnn_resnet50 = models.detection.fasterrcnn_resnet50_fpn(pretrained=download, progress=True, num_classes=91, pretrained_backbone=True)
    maskcnn_resnet50 = models.detection.maskrcnn_resnet50_fpn(pretrained=download, progress=True, num_classes=91, pretrained_backbone=True)
    keypointrcnn_resnet50 = models.detection.keypointrcnn_resnet50_fpn(pretrained=download, progress=True, num_classes=2, num_keypoints=17, pretrained_backbone=True)

    ###########################################################################################
    # Semantic Segmentation

    fcn_resnet50 = models.segmentation.fcn_resnet50(pretrained=download, progress=True, num_classes=21, aux_loss=None)
    fcn_resnet101 = models.segmentation.fcn_resnet101(pretrained=download, progress=True, num_classes=21, aux_loss=None)

    deeplabv3_resnet50 = models.segmentation.deeplabv3_resnet50(pretrained=download, progress=True, num_classes=21, aux_loss=None)
    deeplabv3_resnet101 = models.segmentation.deeplabv3_resnet101(pretrained=download, progress=True, num_classes=21, aux_loss=None)

    ###########################################################################################
    # Generative Adversarial Networks

    
    ###########################################################################################



    checking_input = True
    while (checking_input):
        model_type = int(input("Choose the type of model you want:\n1 (Image Classification)\n2 (Video Classification)\n3 (Object Detection)\n4 (Semantic Segmentation)\n5 (GAN)\nInput: "))
        print(model_type)

        # Convolutional Neual Networks
        if model_type == 1:
            models_dict = {
                "alexnet" : alexnet,
                "squeezenet1_0" : squeezenet1_0,
                "squeezenet1_1": squeezenet1_1,
                "vgg16" : vgg16,
                "vgg19" : vgg19,
                "resnet18" : resnet18,
                "resnet34" : resnet34,
                "resnet50" : resnet50,
                "resnet101" : resnet101,
                "resnet152" : resnet152,
                "densenet121" : densenet121,
                "densenet161" : densenet161,
                "densenet201" : densenet201,
                "densenet121_efficient": densenet121_efficient,
                "densenet161_efficient": densenet161_efficient,
                "densenet201_efficient": densenet201_efficient,
                "googlenet" : googlenet,
                "shufflenet_v2_1" : shufflenet_v2_1,
                "shufflenet_v2_0_5": shufflenet_v2_0_5,
                #"shufflenet_v2_1_5": shufflenet_v2_1_5,
                #"shufflenet_v2_2": shufflenet_v2_2,
                "mobilenet_v2" : mobilenet_v2,
                "resnext50_32x4d" : resnext50_32x4d,
                "resnext101_32x4d": resnext101_32x8d,
                "wide_resnet50_2" : wide_resnet50_2,
                "wide_resnet101_2" : wide_resnet101_2,
                "mnasnet1_0" : mnasnet1_0,
                #"mnasnet1_3": mnasnet1_3,
                "mnasnet0_5": mnasnet0_5,
                #"mnasnet0_75": mnasnet0_75,
            }
            checking_input = False

            return models_dict

        # Video Classification
        elif model_type == 2:
            checking_input = False

            models_dict = {
                "resnet_3d" : resnet_3d,
                "resnet_mixed_conv" : resnet_mixed_conv,
                "resnet_2_1D" : resnet_2_1D
            }
            return models_dict

        # Object Detection
        elif model_type == 3:
            checking_input = False

            models_dict = {
                "fasterrcnn_resnet50" : fasterrcnn_resnet50l,
                "maskcnn_resnet50" : maskcnn_resnet50,
                "keypointrcnn_resnet50" : keypointrcnn_resnet50
            }
            return models_dict

        # Semantic Segmentation
        elif model_type == 4:
            checking_input = False

            models_dict = {
                "fcn_resnet50" : fcn_resnet50,
                "fcn_resnet101" : fcn_resnet101,
                "deeplabv3_resnet50" : deeplabv3_resnet50,
                "deeplabv3_resnet101" : deeplabv3_resnet101
            }
            return models_dict

                # Generative Adversarial Networks
        elif model_type == 5:
            checking_input = False

            models_dict = {

            }
            return models_dict



        else:
            print("You did not choose a valid input...")