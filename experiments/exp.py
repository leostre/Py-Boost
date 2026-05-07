import hydra
from omegaconf import DictConfig

from experiments.core.experiment import ExperimentRunner, BaseExperiment


@hydra.main(version_base=None, config_path="config", config_name="experiment")
def main(cfg: DictConfig):
    """
    Canonical Hydra entrypoint for all experiments.

    The Hydra config is expected to provide:
      - experiment._target_: a subclass of BaseExperiment
      - model._target_: a callable or partial that constructs the model
      - datasets: mapping of dataset configs understood by load_and_preprocess_datasets
    """
    experiment: BaseExperiment = hydra.utils.instantiate(cfg.experiment)
    model_factory = hydra.utils.instantiate(cfg.model)

    datasets_cfg = hydra.utils.instantiate(cfg.datasets)
    skip_datasets = getattr(experiment, "skip_datasets", ()) or {}
    datasets = {
        name: dict(d)
        for name, d in datasets_cfg.items()
        if name not in skip_datasets
    }

    runner = ExperimentRunner()
    runner.run(
        experiment=experiment,
        model_factory=model_factory,
        datasets_config=datasets,
        run_name=experiment.run_name,
    )

    print("All experiments completed successfully!")


if __name__ == "__main__":
    main()
