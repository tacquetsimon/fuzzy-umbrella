from shutil import copytree
import argparse
import os 
import mlflow
import torch



def register_mlflow_pytorch_model(model_path: str):
    with mlflow.start_run() as run:
        model = torch.jit.load(model_path)
        mlflow.pytorch.log_model(
            pytorch_model=model, artifact_path="model_mlflow_test"
        )
        mlflow.register_model(model_uri=f"runs:/{run.info.run_id}/model_mlflow_test", name="mlflow_test_v2")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-model", dest="trained_model")
    
    args = parser.parse_args()
    destination = args.trained_model
    
    model_local_path = os.path.abspath("./models")
    try:
        print(f"copying {model_local_path} into {destination}")
        copytree(src=model_local_path, dst=destination, dirs_exist_ok=True)
    except:
        print("Encountered error in copy to output")
    
    print("mlflow logging start")
    register_mlflow_pytorch_model("./models/model_1/1/model.pt")
    print("Done!")
    
    
