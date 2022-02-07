from stqdm import stqdm

from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar


class StProgressBar(TQDMProgressBar):
    """This class overrides the Progress Bar of Pytorch Lightning. It main purpose
    is to show the task progress in the streamlit interface.

    Args:
        ProgressBar ([type]): Pytorch Lightning progress bar.
    """

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate=refresh_rate, process_position=process_position)

    def init_sanity_tqdm(self) -> stqdm:
        bar = stqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
        )
        return bar

    def init_train_tqdm(self) -> stqdm:
        bar = stqdm(
            desc="Training",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            smoothing=0,
        )
        return bar

    def init_predict_tqdm(self) -> stqdm:
        bar = stqdm(
            desc="Predicting",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> stqdm:
        has_main_bar = self.main_progress_bar is not None
        bar = stqdm(
            desc="Validating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
        )
        return bar

    def init_test_tqdm(self) -> stqdm:
        bar = stqdm(
            desc="Testing",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
        )
        return bar
