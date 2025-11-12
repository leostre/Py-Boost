import hydra
from omegaconf import DictConfig, OmegaConf

from experimental_fundamentals import run_experiments_silent


@hydra.main(version_base=None, config_path="config", config_name="experiment")
def main(cfg: DictConfig):
    
    # Instantiate the model
    # cfg = hydra.utils.instantiate(cfg)
    experiment = hydra.utils.instantiate(cfg.experiment)
    datasets = {name: d for name, d in cfg.datasets.items() if name not in experiment.skip_datasets}
    run_experiments_silent(
        model_generator=hydra.utils.instantiate(cfg.model),  # or pass the instance if supported
        datasets=datasets,
        skip=experiment.skip_datasets,
        run_name=experiment.run_name,
        skip_first=dict(experiment.skip_first),
    )
    
    print("All experiments completed successfully!")

if __name__ == '__main__':
    main()
