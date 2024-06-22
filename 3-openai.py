import openai
import os
import argparse
import json
from tqdm import tqdm

OPENAI_API_KEY = "..."
openai.api_key = OPENAI_API_KEY

#Batch is a array of test sample id

def main(data, model, with_hint):
    data_path = f'../data/prompts/{data}/outliers-train-test'
    save_path = f'../data/answers/{data}/outliers-train-test/{model}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(data_path, f'userid_list'), 'r') as f:
        userid_list = f.read().strip()
    id_list = [int(item) for item in userid_list[1:-1].split(',')]
    for userid in tqdm(id_list[-83:]):
        if with_hint:
            with open(os.path.join(data_path, f'user_{userid}-with-hint'), 'r') as f:
                prompt_file = f.readlines()
        else:
            with open(os.path.join(data_path, f'user_{userid}'), 'r') as f:
                prompt_file = f.readlines()
        prompt = "<This is a very important task, please help me do it>\n"
        prompt = prompt + "".join(prompt_file)
        prompt = prompt + "<This is a very important task, please give the value>\n"
        #print(prompt)

        completion_res = openai.ChatCompletion.create(
            model = model,
            messages = [
                {"role": "user", "content": prompt}
            ]
        )
        ans = completion_res.choices[0].message.content
        if with_hint:
            with open(os.path.join(save_path, f'user_{userid}-with-hint.out'), 'w') as f:
                json.dump(
                    {"prompt":prompt, 'answer':ans}, f
                )
        else:
            with open(os.path.join(save_path, f'user_{userid}.out'), 'w') as f:
                json.dump(
                    {"prompt":prompt, 'answer':ans}, f
                )

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--data", default="geolife")
      parser.add_argument("--model", default="gpt-3.5-turbo-16k-0613")
      parser.add_argument("--with-hint", action="store_true")
      args = parser.parse_args()
      main(
        args.data,
        args.model,
        args.with_hint
        )
