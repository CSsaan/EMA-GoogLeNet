import yaml


def load_model_parameters(file_path):
    with open(file_path, 'r') as file:
        model_params = yaml.safe_load(file)
    return model_params
