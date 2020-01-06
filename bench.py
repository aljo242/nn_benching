import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import torch.onnx
from torchsummary import summary
from thop import profile

import time
import os
from _utils import split_indices, get_default_device, DeviceDataLoader, to_device, fit, evaluate, accuracy, predict_image
import logging
import statistics



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
    return models_dict


def select_model(models_dict):
    print("List of Models to be Selected...")
    print(f"There are {len(models_dict)} options:")
    for key in models_dict.keys():
        print(key)
    blocked = True

    model_name = input("Select a model from above: ")
    for key in models_dict.keys():
        if model_name == key:
            print(f"Selecting {model_name} to benchmark...")
            model = models_dict[model_name]
            return [model, model_name]
    print(f"Model ({model_name}) did not match any given above. Try again...")
    exit()


def get_ImageNet(transform):
    #dataset = datasets.ImageNet(root='data/ImageNet/', download=True)
    #test_dataset = datasets.ImageNet(root='data/ImageNet', train=False, transform=transform)
    cwd = os.getcwd()
    print(cwd)
    test_dir =  cwd + "/data"
    print(test_dir)

    print("Checking if files need to be renamed...")
    for root, subdirs, files in os.walk(test_dir):
        for file in files:
            if os.path.splitext(file)[1] in ( '.JPEG', ".JPG"):
                og = os.path.join(root, file)
                print(og)
                newname = file.replace(".JPEG", ".jpeg")
                newfile = os.path.join(root, newname)
                print(newfile)
                os.rename(og, newfile)

    test = datasets.ImageFolder(test_dir, transform)
    return test

    return [dataset, test_dataset]


if __name__ == "__main__":
    [device, device_name] = get_default_device() 
    if device_name == None:
        device_name = 'CPU'
    cpu = torch.device('cpu') 
    print(str(device))
    torch.backends.cudnn.benchmark = True

    BATCH_SIZE = 1
    SHUFFLE = True
    NUM_WORKERS = 1
    crop_size = 224
    download = False # flag to download pretrained weights

    transform = transforms.Compose([
    	transforms.RandomResizedCrop(crop_size),
    	transforms.RandomHorizontalFlip(),
    	transforms.ToTensor(),
    	transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                         std = [ 0.229, 0.224, 0.225 ]),
    ])

    """
    Use the transforms on the images just to replicate findings reported on pytorch website.

    All models trained on Imagenet (3, 224, 224).  This will be their default input shapes.
    EXCEPT for inception_v3 as noted above.  Will need to be reshaped.
    """
    models_dict = import_models(download)
    [model, model_name]  = select_model(models_dict)
    logger_name = "logs/" + model_name + '_' + device_name +'.log'
    logging.basicConfig(filename=logger_name,filemode='w', format='%(message)s')
    logging.warning("Beginning Log:...\n")


    to_device(model, device, True)
    # send back to cpu host
    #to_device(model, cpu, True)
    test_dataset = get_ImageNet(transform)
    test_loader = DataLoader(test_dataset, 
    	batch_size = BATCH_SIZE, shuffle = SHUFFLE, num_workers = NUM_WORKERS)
    test_loader = DeviceDataLoader(test_loader, device)

    model.eval()
    counter = 0
    times = []
    iterations = 2

    for i in range(iterations):
        for xb, yb in test_loader:
            counter += 1
            print(f"Test #{counter}")
            if counter != 1:
                start_time = time.process_time()
                out = model(xb)
                times.append((time.process_time() - start_time))
            elif counter == 1: 
                out = model(xb)

    summary(model, input_size = (3, 224, 224))
    profile_input = torch.randn(1, 3, 224, 224)
    to_device(profile_input, cpu, True)
    to_device(model, cpu, True)
    flops, params  = profile(model, inputs =(profile_input,))
    print(f"\n\n# of FLOPs: {flops}\n# of Params: {params}")
    logging.warning(f"Model is: {model_name}")
    logging.warning(f"# of FLOPs: {flops}\n# of Params: {params}\n")

    inf_mean = statistics.mean(times)*1000          # convert to ms
    inf_stdev = statistics.stdev(times)*1000        # convert to ms
    print(f"MEAN IS: {(inf_mean)} ms\n")
    print(f"STDEV IS: {(inf_stdev)} ms\n\n\n")
    logging.warning(f"MEAN IS: {(inf_mean)} ms\n") 
    logging.warning(f"STDEV IS: {(inf_stdev)} ms\n\n\n")

    onnx_model_name = "onnx/" + model_name + ".onnx"
    if not os.path.exists(onnx_model_name):
        print(f"Saving Model to onnx format... {onnx_model_name}\n")
        logging.warning(f"Saving Model to onnx format... {onnx_model_name}\n")
        torch.onnx.export(model, profile_input, onnx_model_name, verbose = False, export_params = True, opset_version=10
        , input_names = ['input'], output_names = ['output'])
        print(f"successfully Saved!\n")
        logging.warning(f"Successfully Saved!\n")

    #check parameter counting
    parameters = 0
    for p in model.parameters():
        parameters += int(p.numel())

    if parameters != params:
        print("There was an issue with counting the # of parameters...")
        logging.warning("There was an issue with counting the # of parameters...")

