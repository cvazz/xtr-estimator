from xtr_estimator.main import execute_main, get_config

def main():
    # Load defaults + local yaml
    cfg = get_config(data_yaml="../data/rsEGFP2/local_config.yaml")
    
    # Manual override in code
    # cfg.general.name_human = "Modified_Experiment (Example)"
    
    execute_main(cfg, save2file=True)



if __name__ == "__main__":
    main()
