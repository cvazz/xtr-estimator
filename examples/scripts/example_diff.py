from xtr_estimator.logger import setup_logger
from xtr_estimator.main import execute_main
from xtr_estimator.main import get_config

logger = setup_logger()


def main():
    # Load defaults + local yaml
    # Use meteor tv denoise to calculate difference map
    cfg = get_config(data_yaml="pl30ns_meteor.yaml")
    execute_main(cfg)
    # it is possible to override 'manually' within code
    cfg.map_processing.diffmap_type = "kweighted" # or "tv", "vanilla"
    execute_main(cfg)

    # or use diffmap directly from input for example from Xtrapol8
    cfg = get_config(data_yaml="pl30ns_x8.yaml")
    execute_main(cfg)


if __name__ == "__main__":
    main()
