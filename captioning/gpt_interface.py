import time
from typing import Dict, List
import re

import numpy as np
import openai
import yaml
from tqdm import tqdm


class GPTInterface:
    def __init__(self, cfg: Dict, classnames: List = None, n_prompts: int = None):
        self.cfg = cfg
        self.classnames = classnames

        openai.api_key = cfg["gpt"]["openai_api_key"]
        self.engine = cfg["gpt"]["gpt_engine"]
        if n_prompts is not None:
            self.n_prompts = n_prompts
        else:
            try:
                self.n_prompts = cfg["data"]["few_shot_n_per_class"]
            except KeyError:
                self.n_prompts = None

        if cfg["gpt"]["pregenerate_prompts"]:
            assert classnames is not None, "GPT prompts cannot be pregenerated without classnames."
            self.pregenerate_mode = True
            print(f"Pregenerating GPT prompts for {len(classnames)} classes..")
            self.class_prompts = dict(zip(classnames,
                                          self.generate_prompts(classnames,
                                                                n_prompts=cfg["gpt"]["n_pregenerate_prompts"])))
            print("Done pregenerating prompts.")
        else:
            self.pregenerate_mode = False

    @staticmethod
    def _postprocess(x: str, n_prompts: int = None):
        outs = x.split("\n")
        outs = [o.strip() for o in outs]
        outs = [re.sub(r'^[a-zA-Z0-9]+[.\-\)\s]*', '', o) for o in outs]  # remove item/enumeration number
        outs = [re.sub(r'\([^)]*\)', '', o) for o in outs]  # remove round brackets if there
        outs = [o for o in outs if o != ""]

        return outs

    def generate_prompts_for_request(self, request: str, n_prompts: int = None):
        n_prompts = n_prompts or self.n_prompts
        responses = self._get_raw_string_responses(request)
        responses = self._postprocess(responses, n_prompts=n_prompts)
        return responses

    def generate_prompts_for_requests(self, requests: List[str], n_prompts: int = None):
        n_prompts = n_prompts or self.n_prompts
        responses = [self._get_raw_string_responses(request) for request in tqdm(requests)]
        responses = [self._postprocess(response, n_prompts=n_prompts) for response in responses]
        return responses

    def generate_prompts(self, object_names: np.array, n_prompts: int = None, default_prompt: str = None, multiclass: bool = False, multiclass_percentage: int = 20):
        n_prompts = n_prompts or self.n_prompts
        default_prompt = default_prompt or "List {n_prompts} typical places where a {object_name} would appear. " \
                                           "Avoid being overly specific but give generic answers. " \
                                           "Start the answer with '{object_name} in/at/on.'"

        prompts = []

        for object_name in object_names:
            prompt = default_prompt.format(n_prompts=n_prompts, object_name=object_name)
            if multiclass:
                all_classes = ' '.join(object_names)
                # TODO: maybe better way to do this?
                additional_prompt = "In {} percent of the cases include some of the other classes: ".format(multiclass_percentage) + all_classes + "."
                prompt =  prompt + additional_prompt
            prompts.append(prompt)

        responses = [self._get_raw_string_responses(prompt) for prompt in tqdm(prompts)]
        responses = [np.array(self._postprocess(response, n_prompts=n_prompts)) for response in responses]
        responses = np.stack(responses)
        return responses

    def general_gpt_task(self, request: str):
        responses = self._get_raw_string_responses(request)
        return responses

    def __call__(self, object_names: np.array):
        if self.pregenerate_mode:
            outputs = []
            for object_name in object_names:
                candidate_prompts = self.class_prompts[object_name]
                outputs.append(np.random.choice(candidate_prompts, size=self.n_prompts, replace=True))
            return np.stack(outputs)

        else:
            return self.generate_prompts(object_names, n_prompts=self.n_prompts)

    def _get_raw_string_responses(self, text: str):
        messages = [{"role": "user",
                     "content": text,
                     }]
        while True:
            try:
                completion = openai.ChatCompletion.create(model=self.engine, messages=messages, n=1)
                break
            except Exception as e:
                print("some openai error, retrying")
                print(e)
                time.sleep(10)
        return completion.choices[0].message.content


if __name__ == "__main__":
    """Testing"""
    cfg = yaml.load(open("./sd_tune.yaml", "r"), Loader=yaml.FullLoader)
    gpt = GPTInterface(cfg)

    example_input = np.array(["bus", "person"])

    out = gpt.generate_prompts(example_input, n_prompts=10, default_prompt="Describe {n_prompts} typical situtations where a {object_name} would appear in. The descriptions are for watercolor paintings. Describe the scenes in detail and very the positions/sizes/etc. of {object_name} and various styles of the painting. Usually the paintings are on a white canvas though with white background. Start the answer with '{object_name} in/at/on.'")
    print(out)

