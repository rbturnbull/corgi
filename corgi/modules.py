from pathlib import Path
from torchapp.modules import GeneralLightningModule
import numpy as np
import os
from rich.console import Console

console = Console()

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
            embeddings_index = self.embeddings_path.with_suffix('.index')
            console.print(f"Exporting embeddings to '{self.embeddings_path}'")
            console.print(f"Exporting memmap index to '{embeddings_index}'")
            with open(embeddings_index, 'w') as f:
                for file, record_id, _, _ in self.dataloader.chunk_details:
                    # Replace pipe with underscore for file names
                    file = str(file).replace("\t", "_")  
                    record_id = str(record_id).replace("\t", "_")
                    f.write(f"{file}\t{record_id}\n")
    
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
    

def read_memmap(path, count, dtype:str="float16") -> np.memmap:
    file_size = os.path.getsize(path)
    dtype_size = np.dtype(dtype).itemsize
    num_elements = file_size // dtype_size
    embedding_size = num_elements // count
    shape = (count, embedding_size)
    return np.memmap(path, dtype=dtype, mode='r', shape=shape)


def read_embeddings(embeddings_path:Path|str, embeddings_index:Path=None, merge:bool=False) -> tuple[np.ndarray, list[str]]:
    """
    Read embeddings from a memmap file.
    """
    embeddings_path = Path(embeddings_path)
    index_path = embeddings_index or embeddings_path.with_suffix('.index')
    index_path = Path(index_path)
    assert index_path.exists(), f"Index file '{index_path}' does not exist."
    index_with_tabs = Path(index_path).read_text().strip().splitlines()
    
    embeddings = read_memmap(embeddings_path, len(index_with_tabs))

    if merge:
        import pandas as pd
        df = pd.DataFrame(embeddings)
        df["index_with_tabs"] = index_with_tabs

        # Group by 'index_with_tabs' and average the embeddings
        grouped = df.groupby("index_with_tabs", sort=False).mean(numeric_only=True)

        # Extract the averaged embeddings and the corresponding unique ids
        embeddings = grouped.values
        index_with_tabs = grouped.index.to_list()

    index = [line.split("\t") for line in index_with_tabs]

    return embeddings, index
