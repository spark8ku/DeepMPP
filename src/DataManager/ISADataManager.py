from .MolDataManager import MolDataManager, MolDataManager_v1p3
from D4CMPP.src.utils import PATH

from .GraphGenerator.ISAGraphGenerator import ISAGraphGenerator
from .Dataset.ISAGraphDataset import ISAGraphDataset, ISAGraphDataset_v1p3

class ISADataManager(MolDataManager):
    """The class for the data management of the ISA dataset."""
    def import_others(self):
        if 'sculptor_a' not in self.config or 'sculptor_c' not in self.config or 'sculptor_s' not in self.config:
            raise Exception("The argument 'sculptor_index' is not defined")
        sculptor_index = (self.config['sculptor_s'],self.config['sculptor_c'],self.config['sculptor_a'])
        print("Sculptor Index:",sculptor_index)
        self.graph_type = 'img'+str(sculptor_index[0])+str(sculptor_index[1])+str(sculptor_index[2])
        self.gg = ISAGraphGenerator(
            self.config['MODEL_PATH']+'/functional_group.csv',
            sculptor_index
        )
        self.dataset =ISAGraphDataset
        self.unwrapper = self.dataset.unwrapper
        
class ISADataManager_v1p3(MolDataManager_v1p3):
    """The class for the data management of the ISA dataset version 1.3."""
    def import_others(self):
        if 'sculptor_a' not in self.config or 'sculptor_c' not in self.config or 'sculptor_s' not in self.config:
            raise Exception("The argument 'sculptor_index' is not defined")
        sculptor_index = (self.config['sculptor_s'],self.config['sculptor_c'],self.config['sculptor_a'])
        print("Sculptor Index:",sculptor_index)
        self.graph_type = 'img'+str(sculptor_index[0])+str(sculptor_index[1])+str(sculptor_index[2])
        self.gg = ISAGraphGenerator(
            self.config['MODEL_PATH']+'/functional_group.csv',
            sculptor_index
        )
        self.dataset =ISAGraphDataset_v1p3
        self.unwrapper = self.dataset.unwrapper
