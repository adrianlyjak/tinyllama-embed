import json
import os

file_path = "experiment-config.json"


def get_config() -> dict:
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the existing data
        with open(file_path, "r") as file:
            data = json.load(file)
    else:
        # If the file does not exist, start with an empty dictionary
        data = {}
    return data


def write_config(data: dict) -> None:
    # Write the updated data back to the file
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)


def record_trial_params(experiment_name: str, version: str, params: dict) -> int:
    """
    returns the experiment run number
    """

    # write experiment
    config = get_config()
    cfg_key = f"{experiment_name}.params.{version}"
    runs = config.get(cfg_key) or []

    config[cfg_key] = runs
    run_number = len(runs) + 1

    print(
        f"running experiment {experiment_name} v{version} - run {run_number} with {params}"
    )
    run_config = {"trial": run_number, **params}
    runs.append(run_config)
    write_config(config)
    return run_number


def record_trial_result(
    experiment_name: str, version: str, trial: int, result: dict
) -> None:
    config = get_config()
    cfg_key = f"{experiment_name}.results.{version}"
    results = config.get(cfg_key) or []
    config[cfg_key] = results
    run_config = {"trial": trial, **result}
    results.append(run_config)
    write_config(config)
