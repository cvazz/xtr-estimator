from xtr_estimator.logger import setup_logger
from xtr_estimator.main import execute_main
from xtr_estimator.configuration import  get_config, get_config_triggered, get_config_diff



logger = setup_logger()
def option1():
    # Load defaults + local yaml
    # Use meteor tv denoise to calculate difference map
    cfg = get_config(data_yaml="pl30ns_meteor.yaml")
    execute_main(cfg)

def option2():
    cfg = get_config(data_yaml="pl30ns_meteor.yaml")
    # it is possible to override 'manually' within code
    cfg.map_processing.diffmap_type = "kweighted" # or "tv", "vanilla"
    execute_main(cfg)

def option3():
    # or use diffmap directly from input for example from Xtrapol8
    # this yaml does not get give a light/triggered dataset 
    # but rather the location of a difference map
    cfg = get_config(data_yaml="pl30ns_x8.yaml")
    execute_main(cfg)

def option1_alt():
    # or use diffmap directly from input for example from Xtrapol8
    data_folder= "../data/photolyase"
    map_dark= f"{data_folder}/1_superdark/superdark_deposit.mtz"      
    map_triggered= f"{data_folder}/7_30ns/30ns_deposit.mtz"
    pdb_dark= f"{data_folder}/1_superdark/superdark_deposit.pdb"
    columns_dark= dict(
        amplitude_column= "F-obs-filtered",
        phase_column= "PHIF-model",
        uncertainty_column= "SIGF-obs-filtered"
    )
    columns_triggered= dict(
        amplitude_column= "F",
        phase_column= "PHIF-model",
        uncertainty_column= "SIGF"
    )
    high_resolution_limit= 2.6

    name_machine= "PL_30ns"

    cfg = get_config_triggered(
        dataloc_dark=map_dark,
        dataloc_light=map_triggered,
        pdbloc_dark=pdb_dark,
        columns_dark=columns_dark,
        columns_triggered=columns_triggered,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine
    )
    execute_main(cfg)

def option3_alt():
    # or use diffmap directly from input for example from Xtrapol8
    data_folder= "../data/photolyase"
    map_dark= f"{data_folder}/1_superdark/superdark_deposit.mtz"      
    map_diff= f"{data_folder}/7_30ns/30ns-dark_kwt_ded.mtz"
    pdb_dark= f"{data_folder}/1_superdark/superdark_deposit.pdb"
    columns_dark= dict(
        amplitude_column= "F-obs-filtered",
        phase_column= "PHIF-model",
        uncertainty_column= "SIGF-obs-filtered"
    )
    columns_diff= dict(
        amplitude_column= "KFOFOWT",
        phase_column= "PHIKFOFOWT",
        uncertainty_column= "SIGF"
    )
    high_resolution_limit= 2.6
    name_machine= "PL_30ns_x8"

    cfg = get_config_diff(
        dataloc_dark=map_dark,
        dataloc_diff=map_diff,
        pdbloc_dark=pdb_dark,
        columns_dark=columns_dark,
        columns_diff=columns_diff,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine
    )
    print(cfg["general"]["output_folder"])
    execute_main(cfg)

def main():
    # get_base_config()
    option1()
    option1_alt()
    option2()
    option3()
    option3_alt()



if __name__ == "__main__":
    main()
