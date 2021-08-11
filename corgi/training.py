from pathlib import Path
from fastai.learner import Learner
from fastai.metrics import accuracy
from fastai.callback.tracker import SaveModelCallback
from fastai.callback.schedule import fit_one_cycle

from . import models

def get_learner(
    dls,
    output_dir: (str, Path),
):
    num_classes = len(dls.vocab)
    model = models.ConvRecurrantClassifier(num_classes=num_classes)
    learner = Learner(dls, model, metrics=accuracy, path=output_dir)

    return learner


def train(
    dls,
    output_dir: (str, Path),
    num_epochs: int = 20,
):
    learner = get_learner(dls, output_dir=output_dir)
    callbacks = SaveModelCallback()
    learner.fit_one_cycle(num_epochs, cbs=callbacks)

    return learner
