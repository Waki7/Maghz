# TODO WE NEED TO MOVE ALL THE CONSTANTS TO THE ACTUAL FILE THEY ARE RELEVANT IN, UNLESS IT IS REFERENCED A LOT WHICH IT SHOULDN'T BE A CONSTANT THEN


class Directories:
    LOG_DIR = 'logs'


class Networks:
    CONFIG_FILENAME = 'config.yaml'
    WEIGHTS_FILENAME = 'model.pth'

    ENCODER_WEIGHTS = 'networks/blueprint_weights'
    OPTIMIZER_FILENAME = 'optimizer.pth'


class Names:
    WEIGHTS_NAME = 'weights.pth'


class DataPaths:
    # BSR
    BSR_processed_folder = 'C:\\Users\\Ceyer\\Documents\\Projects\\Data\\Processed'
    BSR_image_folder = 'C:\\Users\\Ceyer\\Documents\\Projects\\FewShotSegmentation\\Data\\BSR\\BSDS500\\data\\images\\'
    BSR_label_folder = 'C:\\Users\\Ceyer\\Documents\\Projects\\FewShotSegmentation\\Data\\BSR\\BSDS500\\data\\groundTruth\\'