import datasets
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
from collections import defaultdict
import tqdm


split = 'test'
if __name__=='__main__':
        
    dataset = datasets.load_dataset('json',data_dir = '/mnt/16t/xzwnlp/unsafetydata/',data_files={'train':'edit_unsafety_train.json','test': 'edit_unsafety_test.json'},split=split)
    print('done')
    print(dataset)

    def split_responses(ex):
        chosen_response = ex['safety_generation']
        rejected_response = ex['unsafety_generation']
        return chosen_response, rejected_response

    data = defaultdict(lambda: defaultdict(list))
    for row in tqdm.tqdm(dataset, desc='Processing safety'):
        prompt = '\n\nHuman: ' + row['prompt'] + '\n\nAssistant:'
        chosen, rejected = split_responses(row)
        responses = [chosen, rejected]
        n_responses = len(data[prompt]['responses'])
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['responses'].extend(responses)
        data[prompt]['sft_target'] = chosen

    print(data[0])