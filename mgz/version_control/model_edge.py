from __future__ import annotations

import json
import os
import time
from enum import Enum

import mgz.log_utils as log_utils
import mgz.version_control as vc
from mgz.ds import BaseDataset
from mgz.model_running.base_routine import TrainState
from mgz.typing import *


class ModelsToStore(str, Enum):
    """
    Enum to specify which models to store during training. Options: -
    LATEST_AND_BEST_VAL: Stores the latest and the best model based on
    validation accuracy. - ALL: Stores all models.
    """
    LATEST_AND_BEST_VAL = 'latest_and_best_val'
    ALL = 'all'


class ModelTransitionEdge:
    """
    Represents a single transition or step in the model training process.
    This class tracks the training statistics and manages the transition from
    one model state to another.
    """

    def __init__(self, orig_model: vc.ModelNode, loss_fn,
                 optimizer: torch.optim.Optimizer, ds: BaseDataset,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 models_to_store: ModelsToStore = ModelsToStore.LATEST_AND_BEST_VAL,
                 ):
        """
        Initializes a model transition edge.

        Parameters: - orig_model (vc.ModelNode): The original model before
        the transition. - loss_fn (Callable): The loss function used for
        training. - optimizer (torch.optim.Optimizer): The optimizer used for
        training. - ds (BaseDataset): The dataset used for training. -
        scheduler (torch.optim.lr_scheduler.LRScheduler, optional): The
        learning rate scheduler. - models_to_store (ModelsToStore): Enum
        indicating which models to store.
        """
        self.parent: vc.ModelNode = orig_model
        orig_model.edges_out.append(self)

        self.child: Optional[vc.ModelNode] = None
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.ds = ds

        # Temporary data used during training
        self.train_state = TrainState()
        self.training_metrics: Dict[vc.Metrics, float] = {}

        # Set up paths for logging and saving models
        run_identifer = "data_{}".format(self.ds.to_json(as_summary_str=True))
        root_path = os.path.join(
            vc.CACHED_INDEXER.path_from_id(self.parent.model_id), run_identifer)
        i = 0
        new_path = root_path + '_' + str(i)
        while os.path.exists(new_path):
            i += 1
            new_path = root_path + '_' + str(i)
        self.root_path = new_path
        self.run_identifer = run_identifer + '_' + str(i)

        # Experiment logger for tracking the training process
        self.exp_tracker = log_utils.ExperimentLogger(self.root_path)
        self.models_to_store = models_to_store
        self.best_val_acc: float = 0.0
        self.timer = None

    def start_timer(self):
        """
        Starts a timer to measure training duration.
        """
        self.timer = time.time()

    def print_train_step_info(self):
        """
        Prints information about the current training step, including loss
        and learning rate.
        """
        lr = self.optimizer.param_groups[0]["lr"]
        print((
                  "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f | "
                  "Tasks / Sec: %7.1f | Learning Rate: %6.1e")
              % (self.train_state.step, self.train_state.accum_step,
                 self.training_metrics[vc.Metrics.TRAIN_LOSS_MEAN],
                 self.train_state.step / (time.time() - self.timer), lr))

    def to_json(self, as_str=False) -> Union[dict, str]:
        """
        Converts the model transition edge to a JSON object or string.

        Parameters: - as_str (bool): If True, returns a JSON string;
        otherwise, returns a dictionary.

        Returns: - Union[dict, str]: The JSON representation of the model
        transition edge.
        """
        obj_dict = {
            'parent': self.parent.model_id,
            'child': self.child.model_id if self.child else None,
            'train_state': self.train_state.to_json(),
            'training_metrics': self.training_metrics
        }
        return json.dumps(obj_dict, indent=4,
                          separators=(',', ': ')) if as_str else obj_dict

    def record_metric(self, metric: vc.Metrics, val: float):
        """
        Records a single metric value for the training transition.

        Parameters:
        - metric (vc.Metrics): The metric to record.
        - val (float): The value of the metric.
        """
        self.training_metrics[metric] = val
        self.exp_tracker.add_scalar(metric, val, track_mean=True)

    def record_metrics(self, metric: vc.Metrics, vals: List[float]):
        """
        Records multiple metric values for the training transition.

        Parameters:
        - metric (vc.Metrics): The metric to record.
        - vals (List[float]): A list of metric values.
        """
        self.training_metrics[metric] = np.mean(vals)
        self.exp_tracker.add_scalars(metric, vals, track_mean=True)

    def log_metric(self, metric: Union[str, vc.Metrics], val: float):
        """
        Records a single metric value for the training transition.

        Parameters:
        - metric (vc.Metrics): The metric to record.
        - val (float): The value of the metric.
        """
        self.exp_tracker.add_scalar(metric, val, track_mean=False, log=True)

    def record_validation(self, vals: List[float]):
        """
        Records validation accuracy metrics.

        Parameters:
        - vals (List[float]): A list of validation accuracy values.
        """
        if len(vals) == 0:
            return
        self.record_metrics(vc.Metrics.VAL_ACC_ALL, vals)
        self.record_metric(vc.Metrics.VAL_ACC_MEAN, np.mean(vals))

    def store_with_identifier(self, identifier: str,
                              extra_config: dict = None) -> vc.ModelNode:
        # Create a unique identifier for the new model
        new_model_id = os.path.join(self.parent.model_id, self.run_identifer,
                                    identifier)

        # Save the new model and its training data
        model_dir: DirPath = vc.CACHED_INDEXER.path_from_id(new_model_id)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.parent.store_model_node(model_dir)

        with open(os.path.join(model_dir, 'edge_info.json').replace("\\", "/"),
                  'w') as f:
            info_dict = self.to_json(as_str=False)
            if extra_config is not None:
                info_dict.update(extra_config)
            f.write(info_dict)

    def complete_model_transition(self) -> vc.ModelNode:
        """
        Completes the model transition, creating a new model node with the
        updated metrics.

        Returns:
        - vc.ModelNode: The new model node resulting from the transition.
        """
        # Construct summary string for naming the new model
        identifiers = []
        store_model = False
        # Determine whether to store the model based on the specified policy
        if self.models_to_store == ModelsToStore.ALL:
            store_model = True
            identifiers.append('steps_{}'.format(self.train_state.step))
        elif self.models_to_store == ModelsToStore.LATEST_AND_BEST_VAL:
            latest_mean_val_acc = self.training_metrics.get(
                vc.Metrics.VAL_ACC_MEAN, 0)

            store_model = True
            if latest_mean_val_acc >= self.best_val_acc:
                identifiers.append("BEST")
            else:
                identifiers.append("LATEST")

            self.best_val_acc = max(self.best_val_acc, latest_mean_val_acc)
        summary_dir = '_'.join(identifiers)

        # Create a unique identifier for the new model
        new_model_id = os.path.join(self.parent.model_id, self.run_identifer,
                                    summary_dir)
        new_model = vc.ModelNode(self.parent.model, self.parent.tokenizer,
                                 new_model_id, metrics=self.training_metrics,
                                 quantization_config=self.parent.quantization_config)
        self.child = new_model

        # Save the new model and its training data
        model_dir: DirPath = vc.CACHED_INDEXER.save_to_index(new_model)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if store_model:
            new_model.store_model_node(model_dir)

        with open(os.path.join(model_dir, 'edge_info.json').replace("\\", "/"),
                  'w') as f:
            f.write(self.to_json(as_str=True))
        return new_model
