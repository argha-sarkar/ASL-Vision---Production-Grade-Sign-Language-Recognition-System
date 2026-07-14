"""
label_encoder.py

Converts original ASL labels into
continuous labels required by TensorFlow.
"""

import numpy as np


class LabelEncoder:

    @staticmethod
    def encode(labels):

        unique_labels = sorted(np.unique(labels))

        label_mapping = {
            old: new
            for new, old in enumerate(unique_labels)
        }

        encoded = np.array(
            [label_mapping[label] for label in labels]
        )

        return encoded, label_mapping