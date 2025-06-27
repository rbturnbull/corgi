from pathlib import Path
from torchapp.modules import GeneralLightningModule
import numpy as np

class CorgiLightningModule(GeneralLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings_path = None
        self.embeddings = None
        self.dataloader = None
        self.current_position = 0

    def set_embedding_path(self, embeddings_path:Path|str, dataloader):
        """
        Set the path for the memmap array and initialize it if necessary.
        """
        self.embeddings_path = Path(embeddings_path)
        self.dataloader = dataloader

    def on_predict_epoch_end(self):
        if self.embeddings is not None:
            self.embeddings.flush()
            del self.embeddings
            self.embeddings = None
            self.current_position = 0

            # Write the memmap index
            memmap_index_path = self.embeddings_path.with_suffix('.index')
            with open(memmap_index_path, 'w') as f:
                for file, record_id, _, chunk_index in self.dataloader.chunk_details:
                    # Replace pipe with underscore for file names
                    file = str(file).replace("|", "_")  
                    record_id = str(record_id).replace("|", "_")
                    f.write(f"{file}|{record_id}|{chunk_index}\n")
    
    def predict_step(self, batch):
        assert isinstance(batch, tuple)
        assert len(batch) >= self.input_count
        x = batch[:self.input_count]

        self.model.return_embeddings = (self.embeddings_path is not None)
        if not self.model.return_embeddings:
            result  = self(*x)
        else:
            result, embedding  = self(*x)

            # Create memmap array if it is the first time
            embedding = embedding.half().cpu().numpy()
            if self.embeddings is None:
                self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                shape = (self.dataloader.count, embedding.shape[-1])
                self.embeddings = np.memmap(self.embeddings_path, dtype=embedding.dtype, mode='w+', shape=shape)
                self.current_position = 0

            # Save embeddings to memmap
            batch_size = len(x[0])
            self.embeddings[self.current_position:self.current_position+batch_size, :] = embedding
            self.current_position += batch_size

        # Return result of the model
        return result
    