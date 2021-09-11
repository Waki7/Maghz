from mgz.models.base_model import BaseModel
from mgz.datasets.spaceship_dataset import Spaceship
from mgz.model_vc.manager import Manager
if __name__ == '__main__':
    manager = Manager()
    sp = Spaceship()

    manager.query_by_dataset(sp)

    print(sp[0])

