from scripts.pipeline import run_pipeline
import json

def load_params(params_path: str):
    """
    Load model parameters from a JSON configuration file.
    """
    with open(params_path, 'r') as file:
        params = json.load(file)
    return params


if __name__ == "__main__":
    # Load parameters from a JSON file. Only change this path to use a different set of parameters.
    # This file should contain the necessary parameters for the YOLO model training.
    params = load_params("data/v2/params.json")
    
    # Define paths and parameters
    model_path = params["parameters"]["pre_trained_model"] # Path to the YOLO model
    project = params["parameters"]["project"] + "/train"  # Directory to save training results
    name = params["parameters"]["name"]  # Name of the training project
    
    data_path = params["parameters"]["model_path"] + "/dataset/data.yaml"      # Path to the dataset configuration file
    epochs = params["parameters"]["epochs"]                   # Number of training epochs
    batch_size = params["parameters"]["batch_size"]               # Batch size for training


    # Run the training and evaluation pipeline
    run_pipeline(
        model_path="data/" + model_path,
        data_path=data_path,
        project=project,
        name=name,
        epochs=epochs,
        batch_size=batch_size
    )