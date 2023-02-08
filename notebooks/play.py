from mgz.datasets.image_datasets.spaceship_dataset import Spaceship
from mgz.model_vc.manager import Manager
from mgz.model_vc.model_node import ModelNode
from mgz.model_vc.model_index import Indexer
from mgz.models.mobile_net import MobileNetV2
import spaces as sp
from mgz.models.default_layers import *

if __name__ == '__main__':
    space_ds = Spaceship()
    net = MobileNetV2()

    network_in_space: sp.GenericBox = net.in_space
    in_layer, in_layer_out_space = get_default_in_layer(space_ds.input_space,
                                                        network_in_space)

    net.set_in_layer(in_layer)
    encoded_out_space = net.out_space(in_layer_out_space)
    predictor = get_default_pred_layer(encoded_out_space, space_ds.target_space)
    net.set_predictor(predictor)

    print(out_spaces)
    print(exit(2))
    net.set_predictor(space_ds.target_space)

    node = ModelNode(net)
    index = Indexer('../index_dir/')
    manager = Manager(index)
    print(index.to_json())
    net.ad
    model: ModelNode = manager.query_by_dataset(space_ds)
    model.model_cls
    # print(sp[0])
