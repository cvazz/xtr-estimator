from xtr_estimator.logger import setup_logger
from xtr_estimator.main import execute_main
from xtr_estimator.main import get_config

logger = setup_logger()


def main():
    # Load defaults + local yaml
    cfg = get_config(data_yaml="../data/rsEGFP2/local_config.yaml")
    
    # Manual override in code
    cfg.general.name_human = "Modified_Experiment (Example)"
    
    execute_main(cfg)



if __name__ == "__main__":
    main()
