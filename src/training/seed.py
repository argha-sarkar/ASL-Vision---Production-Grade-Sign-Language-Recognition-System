import random
import numpy as np
import tensorflow as tf


class SeedManager:
    """
    Controls all random seeds used in the project.
    """

    DEFAULT_SEED = 42

    @staticmethod
    def set_seed(seed: int = DEFAULT_SEED):

        random.seed(seed)

        np.random.seed(seed)

        tf.random.set_seed(seed)