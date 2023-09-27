from __future__ import annotations

from enum import Enum

from torch.utils.tensorboard import SummaryWriter

from mgz.typing import *

if TYPE_CHECKING:
    from mgz.version_control.model_edge import Metrics


class ExperimentLogger:
    def __init__(self):
        self.results_path = ''
        self.writer = SummaryWriter()
        self.progress_values_sum: Dict[Metrics, List[float]] = {}
        self.progress_values_mean: Dict[Metrics, List[float]] = {}
        self.counts: Dict[Metrics, int] = {}

    def reset_buffers(self, reset_count):
        # reset progress buffer after every progress update
        self.progress_values_mean: Dict[Metrics, List[float]] = {}
        self.progress_values_sum: Dict[Metrics, List[float]] = {}
        if reset_count:
            self.counts: Dict[Metrics, int] = {}

    def create_sub_experiment(self):
        pass

    def log_progress(self, episode, step):
        log_output = "episode: {}, step: {}  ".format(episode, step)
        for key in self.progress_values_mean.keys():
            label = 'average_{}'.format(key)
            mean = np.round(np.mean(self.progress_values_mean[key]), decimals=3)
            log_output += '{}: {} , '.format(label, mean)
            self.writer.add_scalar(label, mean, global_step=episode)

        for key in self.progress_values_sum.keys():
            label = 'total_{}'.format(key)
            sum = np.round(np.sum(self.progress_values_sum[key]), decimals=3)
            log_output += '{}: {} , '.format(label, sum)
            self.writer.add_scalar(label, sum, global_step=episode)

        logging.info(log_output)
        self.reset_buffers(False)

    def get_mean_scalar(self, label: Union[str, Metrics]) -> Optional[float]:
        if isinstance(label, Enum):
            label = label.value
        if label not in self.progress_values_mean.keys():
            return None
        return np.mean(self.progress_values_mean[label]).item()

    def get_scalars(self, label: Union[str, Metrics]) -> Optional[List[float]]:
        if isinstance(label, Enum):
            label = label.value
        if label not in self.progress_values_mean.keys():
            return None
        return self.progress_values_mean[label]

    def pop_scalars(self, label: Union[str, Metrics]) -> Optional[List[float]]:
        if isinstance(label, Enum):
            label = label.value
        if label not in self.progress_values_mean.keys():
            return None
        return self.progress_values_mean.pop(label)

    def add_scalar(self, label: Union[str, Metrics], data: Union[int, float],
                   step: Optional[int] = None,
                   track_mean=False, log=True):
        if isinstance(label, Enum):
            label = label.value
        if data is None:
            return
        if track_mean:
            self.progress_values_mean[label] = self.progress_values_mean.get(
                label, [])
            self.progress_values_mean[label].append(data)

        if log:
            if step == None:
                self.counts[label] = self.counts.get(label, 0)
                self.counts[label] += 1
                step = self.counts[label]
            self.writer.add_scalar(label, data, global_step=step)

    def add_scalars(self, label: Union[str, Metrics], data: Iterable,
                    step: Optional[int] = None,
                    track_mean=False, log=True):
        self.add_scalar(label, np.mean(data).item(), step, track_mean, log)


exp_tracker = ExperimentLogger()
