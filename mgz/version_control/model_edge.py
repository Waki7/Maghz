from __future__ import annotations

import json
import os

import mgz.model_running.run_ops as run_ops
import mgz.version_control as vc
from mgz.ds import BaseDataset
from mgz.typing import *


class ModelTransitionEdge:
    '''
    Represents a model "training". This is a single edge in the model graph.
    Will track the stats of the model's training. When a new Node is created,
    the stats will be populated from here.
    '''

    def __init__(self, orig_model: vc.ModelNode, loss_fn,
                 optimizer: torch.optim.Optimizer, ds: BaseDataset,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None):
        '''
        Please only use this from model_node begin_transition
        '''
        self.parent: vc.ModelNode = orig_model
        self.parent.transitioning = True
        orig_model.edges_out.append(self)

        self.child: vc.ModelNode = None
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.ds = ds

        # Temporary Data
        self.train_state = run_ops.TrainState()
        self.training_metrics: Dict[vc.Metrics, float] = {}

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
        self.training_metrics[metric] = val

    def complete_model_transition(self) -> vc.ModelNode:
        """
        Basically this means let's stop the transitioning of the model,
        we want to record it as a new node and save it along with some
        metrics. The naming convention will indicate something about how the
        model was created. It should be a unique identifier.
        """
        self.parent.end_transition()
        assert vc.Metrics.VAL_ACC in self.training_metrics, \
            "Must have validation accuracy to complete model transition"
        summary_string = "train_step_{}_data_{}_valacc_{}".format(
            self.train_state.step, self.ds.to_json(as_summary_str=True),
            self.training_metrics[vc.Metrics.VAL_ACC])
        new_model_id = self.parent.model_id + '/' + summary_string
        new_model = vc.ModelNode(self.parent.model, self.parent.tokenizer,
                                 new_model_id, metrics=self.training_metrics)
        self.child = new_model
        model_dir: DirPath = vc.CACHED_INDEXER.save_to_index(new_model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        new_model.store_model_node()
        with open(
                os.path.join(model_dir, 'edge_info.json').replace("\\", "/"),
                'w') as f:
            f.write(self.to_json(as_str=True))
        return new_model
