#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from typing import Optional, Tuple


# In[ ]:


def Agg_weight(weight_AHP):    
    
    e_vals, e_vecs = np.linalg.eig(weight_AHP)
    temp = pd.DataFrame(e_vecs)
    weight_vector = np.array(temp[0]/temp[0].sum()).real
    
    return weight_vector


# In[ ]:


def Reliability_Index(
    weight_vector,
    score_vector,
    QS: float,
    Ongoing_monitoring: bool = False,
    data_drift: Optional[float] = None,
    performance_drift: Optional[float] = None,
)    -> Tuple[float, str]:
    """
    Derive the final reliavility score and reliability index level by aggregating quantitative and qualitative components.
    ----------
    Input arguments:
        'weight_vector': array, each value corresponds to a weight to one reliability component.
        'score_vector': array, quantitative score for each component.
        'QS': float (between 0 and 1), rate of yes answers of the qualitative questionnaire.
        'Ongoing_monitoring': boolean variable indicating whether information about ongoing monitoring phase is available.
        'data_drift': float, quantitative score associated with data drift in ongoing monitoring phase.
        'performance_drift': float, quantitative score associated with performance drift in ongoing monitoring phase.
    ----------
    Return:
        'RS': float (between 0 and 1), final reliavility score
        'RI': ('High'/'Medium'/'Low'), final reliavility index level
    """

    RS_initial = np.sum(np.multiply(weight_vector, score_vector))

    if Ongoing_monitoring == False:
        RS = RS_initial * QS
    else:
        M = (1 - data_drift) * (1 - performance_drift)
        RS = RS_initial * QS * M

    if RS >= 0.8:
        RI = "High"
    elif 0.5 <= RS < 0.8:
        RI = "Medium"
    else:
        RI = "Low"

    return round(100 * RS, 2), RI
        


# In[ ]:




