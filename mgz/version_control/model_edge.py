from __future__ import annotations

import json
import os
from enum import Enum

import mgz.log_utils as log_utils
import mgz.version_control as vc
from mgz.ds import BaseDataset
from mgz.model_running.base_routine import TrainState
from mgz.typing import *


class ModelsToStore(str, Enum):
    LATEST_AND_BEST_VAL = 'latest_and_best_val'
    ALL = 'all'


class ModelTransitionEdge:
    '''
    Represents a model "training". This is a single edge in the model graph.
    Will track the stats of the model's training. When a new Node is created,
    the stats will be populated from here.
    '''

    def __init__(self, orig_model: vc.ModelNode, loss_fn,
                 optimizer: torch.optim.Optimizer, ds: BaseDataset,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 models_to_store: ModelsToStore = ModelsToStore.LATEST_AND_BEST_VAL):
        '''
        Please only use this from model_node begin_transition
        '''
        self.parent: vc.ModelNode = orig_model
        orig_model.edges_out.append(self)

        self.child: vc.ModelNode = None
        self.optimizer = optimizer
        self.loss_fn: Callable[
            [FloatTensorT['B,NClasses'], LongTensorT['B']], FloatTensorT[
                '1']] = loss_fn
        self.scheduler = scheduler
        self.ds = ds

        # Temporary Data
        self.train_state = TrainState()
        self.training_metrics: Dict[vc.Metrics, float] = {}

        run_identifer = "data_{}".format(
            self.ds.to_json(as_summary_str=True))
        root_path = os.path.join(
            vc.CACHED_INDEXER.path_from_id(self.parent.model_id),
            run_identifer)
        i = 0
        new_path = root_path + '_' + str(i)
        while os.path.exists(new_path):
            i += 1
            new_path = root_path + '_' + str(i)
        self.root_path = root_path + '_' + str(i)
        self.run_identifer = run_identifer + '_' + str(i)

        self.exp_tracker = log_utils.ExperimentLogger(self.root_path)
        self.models_to_store = models_to_store

    def to_json(self, as_str=False) -> Union[dict, str]:
        obj_dict = {
            'parent': self.parent.model_id,
            'child': self.child.model_id,
            'train_state': self.train_state.to_json(),
            'training_metrics': self.training_metrics
        }
        return json.dumps(obj_dict, indent=4,
                          separators=(',', ': ')) if as_str else obj_dict

    def record_metric(self, metric: vc.Metrics, val: float):
        """
        Records metric for model training transition, it will also be
        recorded to tensorboard.
        """
        self.training_metrics[metric] = val
        self.exp_tracker.add_scalar(metric, val, track_mean=True)

    def record_metrics(self, metric: vc.Metrics, vals: List[float]):
        """
        Records metric for model training transition, it will also be
        recorded to tensorboard.
        """
        self.training_metrics[metric] = np.mean(vals)
        self.exp_tracker.add_scalars(metric, vals, track_mean=True)

    def record_validation(self, vals: List[float]):
        self.record_metrics(vc.Metrics.VAL_ACC_ALL, vals)
        self.record_metric(vc.Metrics.VAL_ACC_MEAN, np.mean(vals))

    def complete_model_transition(self) -> vc.ModelNode:
        """
        Basically this means let's stop the transitioning of the model,
        we want to record it as a new node and save it along with some
        metrics. The naming convention will indicate something about how the
        model was created. It should be a unique identifier.
        """

        # start with a non-unique summary string
        summary_string = ""

        # Determine if we should store the best model
        store_model = False
        if self.models_to_store == ModelsToStore.ALL:
            store_model = True
            summary_string += '_steps_{}'.format(self.train_state.step)
        if self.models_to_store == ModelsToStore.LATEST_AND_BEST_VAL:
            latest_mean_val_acc = self.training_metrics[vc.Metrics.VAL_ACC_MEAN]
            best_val_acc = self.exp_tracker.get_max_scalar(
                vc.Metrics.VAL_ACC_MEAN)
            if latest_mean_val_acc > best_val_acc:
                store_model = True
                summary_string = "BEST_" + summary_string

        # Make the string unique depending on what we plan on storing
        new_model_id = self.parent.model_id + '/' + self.run_identifer + '/' + summary_string
        new_model = vc.ModelNode(self.parent.model, self.parent.tokenizer,
                                 new_model_id, metrics=self.training_metrics)
        self.child = new_model
        model_dir: DirPath = vc.CACHED_INDEXER.save_to_index(new_model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if store_model:
            new_model.store_model_node(model_dir)

        with open(
                os.path.join(model_dir, 'edge_info.json').replace("\\", "/"),
                'w') as f:
            f.write(self.to_json(as_str=True))
        return new_model
