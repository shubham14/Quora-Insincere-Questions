# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 21:59:31 2018

@author: Shubham
"""

import torch
import torch.nn as nn

class config:
    bidirectional = True
    word_dimension_dimension = 300
    sentence_max_size = 100
    drop_out = 0.2
    learning_rate = 2e-5
    batch_size = 32
    output_size = 2
    hidden_size = 256
    embedding_length = 300
    VAL_RATIO = 0.2
    MAX_CHARS = 20000