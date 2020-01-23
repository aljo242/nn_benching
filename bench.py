import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import torch.onnx
from torchsummary import summary
from thop import profile
from import_models import import_models
import platform, socket, sys, psutil

import time
import os
from _utils import split_indices, get_default_device, DeviceDataLoader, to_device, fit, evaluate, accuracy, predict_image, printCPUInfo, select_device
import logging
import statistics



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
    test_dir =  cwd + "/data"
    print(f"Checking for directory: {test_dir}")    
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

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

    [device, device_name] = select_device()
    cpu = torch.device('cpu') 
    cpu_name = printCPUInfo()
    if device_name is None:
        device_name = cpu_name
    print(f"CPU: {str(device_name)}")
    print(f"Computing with: {str(device)}")
    torch.backends.cudnn.benchmark = True


    BATCH_SIZE = 1
    SHUFFLE = True
    NUM_WORKERS = 1
    crop_size = 224
    download = True # flag to download pretrained weights

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
    cwd = os.getcwd()
    log_dir =  cwd + "/logs"
    print(f"Checking for directory: {log_dir}")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
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

    #summary(model, input_size = (3, 224, 224))
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
    cwd = os.getcwd()
    onnx_dir =  cwd + "/onnx"
    print(f"Checking for directory: {onnx_dir}")    
    if not os.path.exists(onnx_dir):
        os.mkdir(onnx_dir)

    if not os.path.exists(onnx_model_name):
        print(f"Saving Model to onnx format... {onnx_model_name}\n")
        logging.warning(f"Saving Model to onnx format... {onnx_model_name}\n")
        torch.onnx.export(model, profile_input, onnx_model_name, verbose = False, export_params = True, opset_version=11
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

