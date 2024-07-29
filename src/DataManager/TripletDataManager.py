from .MolDataManager import MolDataManager, MolDataManager_withSolv

from .GraphGenerator.MolGraphGenerator import MolGraphGenerator
from .GraphGenerator.TripletGraphGenerator import TripletGraphGenerator
from .Dataset.TripletGraphDataset import TripletGraphDataset, TripletGraphDataset_withSolv
    
class TripletDataManager(MolDataManager):
    def __init__(self, config):
        super(TripletDataManager, self).__init__(config)
        config.update({"triplet_dim":self.gg.triplet_dim})

    def import_others(self):
        self.graph_type = 'triplet'
        self.gg = TripletGraphGenerator(xyz_path=self.config.get('XYZ_PATH',None))
        self.dataset =TripletGraphDataset
        self.unwrapper = self.dataset.unwrapper

class TripletDataManager_withSolv(MolDataManager_withSolv):
    def __init__(self, config):
        super(TripletDataManager_withSolv, self).__init__(config)
        config.update({"triplet_dim":self.gg.triplet_dim})

    def import_others(self):
        self.graph_type = 'triplet'
        self.gg = TripletGraphGenerator(xyz_path=self.config.get('XYZ_PATH',None))
        self.gg_solv = MolGraphGenerator()
        self.dataset =TripletGraphDataset_withSolv
        self.unwrapper = self.dataset.unwrapper