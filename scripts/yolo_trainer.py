from ultralytics import YOLO


class YOLOTrainer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path, task='detect')

    def train(self, project:str, name: str, data_path: str, epochs: int = 100, batch_size: int = 16):
        """
        Train the YOLO model.
        """
        self.model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            project=project,
            name=name,
        )

    def evaluate(self, data_path: str):
        """
        Evaluate the YOLO model.
        """
        results = self.model.val(data=data_path)
        print(results)
        return results
