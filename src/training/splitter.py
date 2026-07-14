from sklearn.model_selection import train_test_split


class DatasetSplitter:

    @staticmethod
    def split(
        images,
        labels,
        test_size=0.2,
        random_state=42,
    ):

        return train_test_split(
            images,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )