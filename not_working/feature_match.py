import pandas as pd
import numpy as np
from typing import List, Tuple, Callable

import pandas as pd
import numpy as np
from typing import List, Tuple, Callable

class FeatureMatchAttack:
    def __init__(self, aux_data: pd.DataFrame):

        self.aux_data = aux_data
        
    def create_filter_function(self, target_features: List[Tuple[int, int, int]]) -> Callable:

        # def filter_fn(sample: pd.Series) -> bool:
        #     for pixel_idx, min_val, max_val in target_features:
        #         pixel_val = sample[f"pixel{pixel_idx}"]
        #         if pixel_val < min_val or pixel_val > max_val:
        #             return False
        #     return True
        
        # going to look at poisoning numbers that have holes
        # so need to find the rough coordinates, and then poison them as long as a certain percentage of the points are
        # within the range - otherwise there will be too few points that are successfully posioned

        def filter_fn(sample: pd.Series) -> bool:
            matching_pixels = sum(
                min_val <= sample[f"pixel{pixel_idx}"] <= max_val
                for pixel_idx, min_val, max_val in target_features
            )
            return matching_pixels >= (len(target_features) * 0.1)
            
        return filter_fn
    
    def generate_poisoned_data(self, 
                             filter_fn: Callable,
                             poison_rate: float,
                             target_label: int) -> pd.DataFrame:

        matching_samples = self.aux_data[self.aux_data.apply(filter_fn, axis=1)]
        
        num_poison = int(len(matching_samples) * poison_rate)
        
        if num_poison == 0:
            raise ValueError("No matching samples found or poison_rate too low")
            
        poison_samples = matching_samples.sample(n=num_poison, replace=False)
        
        poisoned_data = poison_samples.copy()
        poisoned_data['label'] = target_label
        
        return poisoned_data

    def attack(self,
              target_features: List[Tuple[int, int, int]],
              poison_rate: float,
              target_label: int) -> pd.DataFrame:

        filter_fn = self.create_filter_function(target_features)
        return self.generate_poisoned_data(filter_fn, poison_rate, target_label)

def attack_mnist(aux_data: pd.DataFrame,
                target_features: List[Tuple[int, int, int]],
                poison_rate: float = 0.5,
                target_label: int = 0) -> pd.DataFrame:

    attack = FeatureMatchAttack(aux_data)
    return attack.attack(target_features, poison_rate, target_label)

def attack_mnist_with_replacement(train_data: pd.DataFrame, 
                                  aux_data: pd.DataFrame,
                                  target_features: List[Tuple[int, int, int]],
                                  poison_rate: float = 0.5,
                                  target_label: int = 0) -> pd.DataFrame:
    
    attack = FeatureMatchAttack(aux_data)

    poisoned_data = attack.attack(target_features, poison_rate, target_label)

    filter_function = attack.create_filter_function(target_features)

    match_mask = train_data.apply(filter_function, axis=1)
    num_matching = match_mask.sum()

    if num_matching < len(poisoned_data):
        raise ValueError(f"Not enough matching samples in training data to replace. "
                       f"Need {len(poisoned_data)} but only found {num_matching}")
    
    replace_indices = train_data[match_mask].sample(n=len(poisoned_data), replace=False).index

    poisoned_dataset = train_data.copy()
    poisoned_dataset.loc[replace_indices] = poisoned_data.values

    return poisoned_dataset

