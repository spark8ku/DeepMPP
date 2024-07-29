from .MolDataManager import MolDataManager
from D4CMPP.src.utils import PATH

from .GraphGenerator.ISAGraphGenerator import ISAGraphGenerator
from .Dataset.ISAGraphDataset import ISAGraphDataset

class ISADataManager(MolDataManager):
    """The class for the data management of the ISA dataset."""
    def import_others(self):
        if 'sculptor_index' not in self.config:
            raise Exception("The argument 'sculptor_index' is not defined")
        sculptor_index = self.config.get('sculptor_index')
        print("Sculptor Index:",sculptor_index)
        self.graph_type = 'img'+str(sculptor_index[0])+str(sculptor_index[1])+str(sculptor_index[2])
        self.gg = ISAGraphGenerator(
            self.config.get('frag_ref',self.config['FRAG_REF']),
            sculptor_index
        )
        self.dataset =ISAGraphDataset
        self.unwrapper = self.dataset.unwrapper