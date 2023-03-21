"""
    File to load dataset based on user control from main file
"""
from data.ogbdata import OGBDataset
from data.planarity import PlanarityDataset
from data.molecules import MoleculeDataset
from data.SBMs import SBMsDataset
# from data.COLLAB import COLLABDataset
from data.CSL import CSLDataset
from data.cycles import CyclesDataset


def LoadData(DATASET_NAME, **kwargs):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for (ZINC) molecule dataset
    if DATASET_NAME in ['ZINC', 'ZINC-full', 'AQSOL']:
        return MoleculeDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDataset(DATASET_NAME)
    
    # handling for the CSL (Circular Skip Links) Dataset
    if DATASET_NAME == 'CSL': 
        return CSLDataset(DATASET_NAME)
    
    # handling for the CYCLES Dataset from https://github.com/cvignac/SMP
    if DATASET_NAME == 'CYCLES': 
        return CyclesDataset(DATASET_NAME, k=kwargs['cycles_k'])

    if DATASET_NAME == 'Planarity':
        return PlanarityDataset(DATASET_NAME)
    
    if DATASET_NAME == 'OGB':
        return OGBDataset(DATASET_NAME, logger=kwargs['logger'])