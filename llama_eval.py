import pandas as pd
from unsloth import FastLanguageModel
import torch
import json
import re
import numpy as np
import matplotlib.pyplot as plt

MODELPATH = './models/llama3_8b_10ep'
OUTFILE = './models/llama3_8b_10ep/llama3_8b_prediction.csv'

def predict_next_week():
    test_data = pd.read_csv('./data/llama3_test_health.csv', index_col=0)
    test_data['1week_target'] = ''
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODELPATH, 
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True
    )
    FastLanguageModel.for_inference(model)
    for i in range(test_data.shape[0]):
        input_ids = tokenizer.encode(test_data['prompt'].values[i], return_tensors='pt').to('cuda')
        outputs = model.generate(input_ids = input_ids, max_new_tokens = 500, use_cache = True,
                             temperature = 1.5, min_p = 0.1)
        prediction = tokenizer.batch_decode(outputs[:, input_ids.shape[1]:-1])[0]
        match = re.search(r'"target":\s*([0-9.]+)', prediction)
        if match:
            target_value = float(match.group(1))
            test_data['1week_target'].iloc[i] = target_value
        else:
            print("Target not found")
        if (i+1) % 10 == 0:
            print(f"Total {i+1} records done")
    test_data.to_csv(OUTFILE)

def evaluate():
    test_data = pd.read_csv(OUTFILE)
    test_error = test_data.target.values.flatten() - test_data['1week_target'].values.flatten()
    test_error = test_error**2
    test_error = np.mean(test_error)
    print(f"Test MSE error is {test_error:0.3f}")
    plt.figure(figsize=(15, 8))
    plt.plot(test_data.target.values.flatten(), 'g--', label='actual')
    plt.plot(test_data['1week_target'].values.flatten(), 'r--', label='predicted')
    plt.legend()
    plt.grid()
    plt.title(f'One week prediction (LLama 3.2 8B) - MSE - {test_error:0.3f}')
    plt.savefig('./llama_8b_result.png')
    plt.show()


if __name__ == "__main__":
    # predict_next_week()
    evaluate()
