---
layout: post
title: Logging with MLFlow and Lightning
categories: [MLOps]
tags: [mlops, mlflow, lightning]
date: 2023-05-31 +0200
pin: true
---
Since you are a good MLE, you know that tracking experiments is essential.

Luckly for you, Lightning offers a lot of different logging functionality, including a [MLFlow logger class](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.mlflow.html), which will do the job for you.

Let's check it out. Simply define it in your code, as such:


```python
from lightning.pytorch.loggers import MLFlowLogger

...
# some code here
...

logger = MLFlowLogger("Default", "test_run")

...
# some code here
...

es = EarlyStopping(monitor="val_acc", patience=3, min_delta=0.05)
trainer = Trainer(max_epochs=20, logger=logger, callbacks=[es])
# trainer.fit()
```

Quite easy. You just need to set the experiment name and run name, and you are good to go. Let's check out the result.

![The logger logged just a few parameters.](/assets/imgs/lightning_and_mlflow_logging/before.png)

That's quite disappointing, right? Even if the metrics were logged correctly and we got our cute plots, we didn't log any of the relevant parameters. We got no information about the number of epochs, the transform we did on data, the early stopping criteria we used and so on.

Luckily for us, we can fix that with just a few lines of code.

```python
import mlflow
from lightning.pytorch.loggers import MLFlowLogger

# define experiment and run name
experiment_name = "Default"
run_name = "test_run"

# set experiment name, autologging for Pytorch and start the run
mlflow.set_experiment(experiment_name)
mlflow.pytorch.autolog()
mlflow.start_run(run_name=run_name)

...
# some code here
...

# now we can log whatever we want
transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
mlflow.log_param("dataset transformation", transform)

...
# some more code here
...

# how to integrate the MLFlow logger that we are using right now with the Lightning's one?
# As simple as doing this
logger = MLFlowLogger(
        experiment_name=mlflow.get_experiment(
            mlflow.active_run().info.experiment_id
        ).name,
        run_id=mlflow.active_run().info.run_id,
        log_model="all",
)

es = EarlyStopping(monitor="val_acc", patience=3, min_delta=0.05)
trainer = Trainer(max_epochs=20, logger=logger, callbacks=[es])

# let's log everything we want!
training_params = {
        "batch_size": BATCH_SIZE,
        "n_workers": NUM_WORKERS,
        "callbacks": [es],
        "es metric": "val_acc",
        "es_patience": 3,
        "es_min_delta": 0.05,
    }
mlflow.log_params(training_params)
# trainer.fit()
```

Let's see the result.

![Now we logged a lot of stuff!](/assets/imgs/lightning_and_mlflow_logging/after.png)

These are just a few of all the parameters logged. The autolog function of MLFlow logged everythig we needed to track and replicate the experiment.
Moreover, we also got some useful artifacts:

![Logged artifacts.](/assets/imgs/lightning_and_mlflow_logging/artifacts.png)

Very useful and simple.

I hope this helps, bye 🤖