import lightning.pytorch as pl
import torch

class Datamodule(pl.LightningDataModule):
    """DataModule for the electricity price forecast task
    
    Attributes:
        train_dataset (torch.utils.data.Dataset): Training dataset
        val_dataset (torch.utils.data.Dataset): Validation dataset
        test_dataset (torch.utils.data.Dataset): Test dataset
        batch_size (int): Batch size
    """
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size):
        """Initialize the DataModule
        
        Args:
            train_dataset (torch.utils.data.Dataset): Training dataset
            val_dataset (torch.utils.data.Dataset): Validation dataset
            test_dataset (torch.utils.data.Dataset): Test dataset
            batch_size (int): Batch size
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        
    def train_dataloader(self):
        """Return the training dataloader
        
        Returns:
            torch.utils.data.DataLoader: Training dataloader
        """
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        """Return the validation dataloader
        
        Returns:
            torch.utils.data.DataLoader: Validation dataloader        
        """
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        """Return the test dataloader
        
        Returns:
            torch.utils.data.DataLoader: Test dataloader
        """
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
    