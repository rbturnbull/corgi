from pathlib import Path
from torchapp.modules import GeneralLightningModule
import numpy as np

from .data import SeqIODataloader

class CorgiEmbeddingLightningModule(GeneralLightningModule):
    def __init__(self, memmap_array_path:Path|str, dataloader:SeqIODataloader, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memmap_array_path = Path(memmap_array_path)
        self.memmap_array_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataloader = dataloader
        self.count = self.dataloader.count

        self.memmap_array = None

        self.model.return_embedding = True
        self.current_position = 0

    def on_epoch_end(self):
        if self.memmap_array is not None:
            self.memmap_array.flush()
            del self.memmap_array
            self.memmap_array = None
            self.current_position = 0

            # Write the memmap index
            memmap_index_path = self.memmap_array_path.with_suffix('.index')
            with open(memmap_index_path, 'w') as f:
                for file, record_id, _, chunk_index in self.dataloader.chunk_details:
                    # Replace pipe with underscore for file names
                    file = str(file).replace("|", "_")  
                    record_id = str(record_id).replace("|", "_")
                    f.write(f"{file}|{record_id}|{chunk_index}\n")
                
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        assert isinstance(batch, tuple)
        assert len(batch) >= self.input_count
        x = batch[:self.input_count]
        result, embedding  = self(*x)
        
        # Create memmap array if it is the first time
        embedding = embedding.cpu().numpy()
        if self.memmap_array is None:
            shape = (self.count, embedding.shape[-1])
            self.memmap_array = np.memmap(self.memmap_array_path, dtype=embedding.dtype, mode='w+', shape=shape)

        # Save embeddings to memmap
        batch_size = len(x[0])
        self.memmap_array[self.current_position:self.current_position+batch_size, :] = embedding
        self.current_position += batch_size

        # Return result of the model
        return result
    