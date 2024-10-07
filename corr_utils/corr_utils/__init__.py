# author: Noel Kronenberg

import corr_utils.extraction
import corr_utils.covariate
import corr_utils.analysis
import corr_utils.ml

global default_key
default_key:str = 'case_id' 
corr_utils.extraction.default_key = default_key
corr_utils.covariate.default_key = default_key
corr_utils.analysis.default_key = default_key
corr_utils.ml.default_key = default_key

def set_default_key(key:str='case_id') -> None:
    """
    Sets the default key for merging operations.

    Parameters:
        key (str): The key to set. Defaults to 'case_id'.

    Returns:
        None
    """

    global default_key 
    default_key = key
    corr_utils.extraction.default_key = key
    corr_utils.covariate.default_key = key
    corr_utils.analysis.default_key = key
    corr_utils.ml.default_key = key