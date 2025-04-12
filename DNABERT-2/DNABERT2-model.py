import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
import os


def process_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    result = []
    for line in lines:
        if line.startswith('>'):
            continue
        else:
            line = line.strip()
            line = line.upper()
            print(line)
            result.append(line)
    return result


def main(file_path, output_path):
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    dnas = process_file(file_path)

    #print(dnas)

    if file_path[11] == 'p':
        labels = [1] * len(dnas)
    else:
        labels = [0] * len(dnas)

    features = []

    if os.path.exists(output_path):
        os.remove(output_path)  # 移除旧文件

    for dna in tqdm(dnas):
        inputs = tokenizer(dna, return_tensors='pt')["input_ids"]

        with torch.no_grad():
            hidden_states = model(inputs)[0]  # [1, sequence_length, 768]

        embedding_mean = torch.mean(hidden_states[0], dim=0)
        #embedding_max = torch.max(hidden_states[0], dim=0)[0]
        #print(embedding_mean.shape)
        features.append(embedding_mean.squeeze().cpu().numpy().tolist())
    #print(embedding_mean.shape)
    #print(inputs.shape)
    #print(hidden_states.shape)

    with open(output_path, 'w') as file:
         json.dump({
             "feature": features,
             "label": labels
         }, file)


if __name__ == '__main__':
    file_path = ['GLNet-TFBS/' + x for x in
                 ['positive_test.txt', 'negative_test.txt']]
    output_path = ['GLNet-TFBS/' + x for x in
                   ['positive_test.json', 'negative_test.json']]
    for i in range(2):
        main(file_path[i], output_path[i])
