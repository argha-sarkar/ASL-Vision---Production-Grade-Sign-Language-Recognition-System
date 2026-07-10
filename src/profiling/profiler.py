from src.profiling.statistics import DatasetStatistics


class DatasetProfiler:

    @staticmethod
    def profile(df):

        report = {

            "Shape": DatasetStatistics.dataset_shape(df),

            "Missing Values": DatasetStatistics.missing_values(df),

            "Duplicates": DatasetStatistics.duplicate_rows(df),

            "Memory (MB)": round(
                DatasetStatistics.memory_usage(df), 2
            ),

            "Pixel Min": DatasetStatistics.pixel_min(df),

            "Pixel Max": DatasetStatistics.pixel_max(df),

            "Pixel Mean": round(
                DatasetStatistics.pixel_mean(df), 2
            ),

            "Pixel Std": round(
                DatasetStatistics.pixel_std(df), 2
            ),

        }

        return report