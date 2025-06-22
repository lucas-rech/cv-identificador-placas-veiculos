from .yolo_trainer import YOLOTrainer
import time



def run_pipeline(
    model_path: str, 
    data_path: str,
    project: str,
    name: str, 
    epochs: int = 100, 
    batch_size: int = 16):
    start_time = time.time()

    
    trainer = YOLOTrainer(model_path)
    trainer.train(project, name, data_path, epochs=epochs, batch_size=batch_size)
    end_time = time.time()
    print(f"Pipeline completed in {end_time - start_time:.2f} seconds.")

    trainer.evaluate(data_path)

    
