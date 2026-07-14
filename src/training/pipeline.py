from src.training.dataset import DatasetBuilder
from src.training.splitter import DatasetSplitter


class TrainingPipeline:

    @staticmethod
    def prepare(
        images,
        labels,
        batch_size=64,
    ):

        (
            x_train,
            x_val,
            y_train,
            y_val,
        ) = DatasetSplitter.split(
            images,
            labels,
        )

        train_dataset = DatasetBuilder.build(
            x_train,
            y_train,
            batch_size=batch_size,
            shuffle=True,
        )

        validation_dataset = DatasetBuilder.build(
            x_val,
            y_val,
            batch_size=batch_size,
            shuffle=False,
        )

        return (
            train_dataset,
            validation_dataset,
            x_train,
            x_val,
            y_train,
            y_val,
        )
