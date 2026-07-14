import tensorflow as tf


class DatasetBuilder:

    @staticmethod
    def build(
        images,
        labels,
        batch_size=64,
        shuffle=True,
    ):

        dataset = tf.data.Dataset.from_tensor_slices(
            (images, labels)
        )

        if shuffle:

            dataset = dataset.shuffle(
                buffer_size=len(images),
                reshuffle_each_iteration=True,
            )

        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(
            tf.data.AUTOTUNE
        )

        return dataset