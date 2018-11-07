from network_init import get_model
from io_utils import *
import tensorflow as tf
from forward import forward_model
from train import train_model

tf.set_random_seed(0)

if __name__ == "__main__":
    outputChannels = 16
    savePrefix = "all"
    outputPrefix = ""
    # 0=car, 1=person, 2=rider, 3=motorcycle, 4=bicycle, 5=truck, 6=bus, 7=train
    train = True

    if train:
        batchSize = 2
        learningRate = 5e-6 # usually i use 5e-6
        wd = 1e-6

        #modelWeightPaths = [""]

        # modelWeightPaths = ["../model/depth/depth_unified_CR_CR_pretrain_060.mat","../model/direction/direction_unified_CR_unified_CR_pretrain_060.mat"]
        modelWeightPaths = ["../model/01.mat"]
	initialIteration = 1

        trainFeeder = Batch_Feeder(dataset="cityscapes",
                                           train=train,
                                           batchSize=batchSize,
                                           flip=True, keepEmpty=False, shuffle=True)

        trainFeeder.set_paths(idList=read_ids('../cityscapes/splits/train/list.txt'),
                         imageDir="../cityscapes/leftImg8bit/train",
                         gtDir="../cityscapes/unified/iGTFull/train",
                         ssDir="../cityscapes/gtFine/train")

        valFeeder = Batch_Feeder(dataset="cityscapes",
                                         train=train,
                                         batchSize=batchSize, shuffle=False)

        valFeeder.set_paths(idList=read_ids('../cityscapes/splits/val/list.txt'),
                         imageDir="../cityscapes/leftImg8bit/val",
                         gtDir="../cityscapes/unified/iGTFull/val",
                         ssDir="../cityscapes/gtFine/val")

        model = get_model(wd=wd, modelWeightPaths=modelWeightPaths)

        train_model(model=model, outputChannels=outputChannels,
                    learningRate=learningRate,
                    trainFeeder=trainFeeder,
                    valFeeder=valFeeder,
                    modelSavePath="../model",
                    savePrefix=savePrefix,
                    initialIteration=initialIteration)

    else:
        batchSize = 1
        #modelWeightPaths = ["../model/depth/depth_unified_CR_CR_pretrain_060.mat","../model/direction/direction_unified_CR_unified_CR_pretrain_060.mat"]
	modelWeightPaths = ["../model/Q_935.mat"]
        model = get_model(modelWeightPaths=modelWeightPaths)

        feeder = Batch_Feeder(dataset="cityscapes",
                                      train=train,
                                      batchSize=batchSize)

        feeder.set_paths(idList=read_ids('../example/sample_list.txt'),
                         imageDir="../example/inputImages",
                            ssDir="../example/PSPNet")

        forward_model(model, feeder=feeder,
                      outputSavePath="../example/output")
