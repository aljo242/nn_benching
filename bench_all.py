import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch
import torch.onnx
from torchsummary import summary
from thop import profile
from import_models import import_all
from pathlib import Path

import time
import os
from _utils import DeviceDataLoader, to_device, printCPUInfo, select_device
import logging
import statistics

from modelstats import get_critical_path



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

import numpy as np

def get_ImageNet(transform):
    # dataset = datasets.ImageNet(root='data/ImageNet/', download=True)
    # test_dataset = datasets.ImageNet(root='data/ImageNet', train=False, transform=transform)
    cwd = os.getcwd()
    print(cwd)
    test_dir =  cwd + "/data"
    print(test_dir)

    print("Checking if files need to be renamed...")
    for root, _, files in os.walk(test_dir):
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


if __name__ == "__main__":

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    [device, device_name] = select_device()
    cpu_name = printCPUInfo()
    device = torch.device('cpu')
    device_name = None
    if device_name == None:
        device_name = cpu_name.replace(' ', '_')
    cpu = torch.device('cpu')
    print(f"Computing with: {str(device_name)}", flush=True)
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
	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    """
    Use the transforms on the images just to replicate findings reported on pytorch website.

    All models trained on Imagenet (3, 224, 224).  This will be their default input shapes.
    EXCEPT for inception_v3 as noted above.  Will need to be reshaped.
    """
    test_dataset = get_ImageNet(transform)
    test_loader = DataLoader(test_dataset,
       batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
    models_dict = import_all(download)

    # print(f"LENGHT OF THE DICT IS: {len(models_dict)}")
    Path("logs").mkdir(parents=True, exist_ok=True)
    print("Device: ", device_name)
    print("model_name,flops,params,cdl,avg_lat,med_lat,inf_mean,inf_stdev")
    logger_name = "logs/" + 'all_tests_' + device_name +'.log'
    logging.basicConfig(filename=logger_name, filemode='w', format='%(message)s')
    logging.disable(logging.INFO)
    logging.info("Beginning Log:...\n")
    logging.warning("model_name,flops,params,cdl,avg_lat,med_lat,inf_mean,inf_stdev")

    for model_name in models_dict:
        model = models_dict[model_name]
    #[model, model_name]  = select_model(models_dict)
        print(f"Testing: {model_name}\n", flush=True)
        logging.info(f"Testing: {model_name}\n")

        device_loader = DeviceDataLoader(test_loader, device)
        to_device(model, device, True)
        # send back to cpu host
        #to_device(model, cpu, True)

        model.eval()
        counter = 0
        times = []
        iterations = 1

        for i in range(iterations):
            for xb, yb in device_loader:
                counter += 1
                print(f"Test {counter}", flush=True, end='\r')
                if counter != 1:
                    start_time = time.perf_counter()
                    out = model(xb)
                    times.append((time.perf_counter() - start_time))
                elif counter == 1:
                    out = model(xb)

                if counter > 20:
                    break

        #   summary(model, input_size = (3, 224, 224))
        try:
            profile_input = torch.randn(1, 3, 224, 224)
            to_device(profile_input, cpu, True)
            to_device(model, cpu, True)
            flops, params = profile(model, inputs=(profile_input,))
            logging.info(f"\n\nModel: {model_name}")
            print(f"# FLOPs: {flops}\n# Params: {params}")
            logging.info(f"# FLOPs: {flops}\n# Params: {params}\n")
        except RuntimeError:
            print(f"Could not compute model statistics {model_name}")

        inf_mean = statistics.mean(times)*1000          # convert to ms
        inf_stdev = statistics.stdev(times)*1000        # convert to ms

        critical_path, latencies, sorted_latencies = get_critical_path(model)
        lat_arr = np.array(sorted_latencies)

        cdl = np.max(lat_arr)
        avg_lat = np.average(lat_arr)
        med_lat = np.median(lat_arr)

        print(f"# CDL: {np.max(lat_arr)}")
        print(f"# Avg Node Latency {np.average(lat_arr)}")
        print(f"# Median Node Latency {np.median(lat_arr)}")
        logging.info(f"# CDL: {np.max(lat_arr)}")
        logging.info(f"# Avg Node Latency: {np.average(lat_arr)}")
        logging.info(f"# Median Node Latency: {np.average(lat_arr)}")


        #onnx_model_name = "onnx/" + model_name + ".onnx"
        #if not os.path.exists(onnx_model_name):
        #    print(f"Saving Model to onnx format... {onnx_model_name}\n")
        #   logging.info(f"Saving Model to onnx format... {onnx_model_name}\n")
        #   torch.onnx.export(model, profile_input, onnx_model_name, verbose = False, export_params = True, opset_version=11
        #        , input_names = ['input'], output_names = ['output'])
        #    print(f"successfully Saved!\n")
        #    logging.info(f"Successfully Saved!\n")

        #check parameter counting
        parameters = 0
        for p in model.parameters():
            parameters += int(p.numel())
        if parameters != params:
            print("There was an issue with counting the # of parameters...")
            logging.info("There was an issue with counting the # of parameters...")


        print(f"# Inference Mean: {(inf_mean)}")
        print(f"# Inference Stdev: {(inf_stdev)}\n\n\n")
        logging.info(f"# Inference Mean: {(inf_mean)}")
        logging.info(f"# Inference Stdev: {(inf_stdev)}\n")
        logging.info(f"# Inference Stdev: {(inf_stdev)}\n")
        print(f"{model_name},{flops},{params},{cdl},{avg_lat},{med_lat},{inf_mean},{inf_stdev}")
        logging.warning(f"{model_name},{flops},{params},{cdl},{avg_lat},{med_lat},{inf_mean},{inf_stdev}")
