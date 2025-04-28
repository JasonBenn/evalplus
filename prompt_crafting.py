# %%
print('hi')
# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
import torch
import numpy as np

device = "cuda"
torch.set_grad_enabled(False)

# %%
hf_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
).to(device)

hf_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    use_fast=True,
    add_bos_token=True,
)

# %%
model = HookedTransformer.from_pretrained_no_processing(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device=device,
    hf_model=hf_model,
    hf_config=hf_model.config,
    tokenizer=hf_tokenizer,
    dtype=torch.bfloat16,
)

# %%
# test_prompts = [
#     "Write a function that adds two numbers and returns the result.",
#     "Create a program that checks if a number is even or odd.",
#     "Write code to find the length of a string without using len().",
#     "Make a function that returns the reverse of a given string.",
#     "Write a program to calculate the factorial of a number."
# ]

test_prompts = [
    """Think step by step: please provide an efficient and self-contained Python script that solves the following problem in an <answer> tag:

    from typing import List

    def intersperse(numbers: List[int], delimeter: int) -> List[int]:
        \"\"\" Insert a number 'delimeter' between every two consecutive elements of input list `numbers'
        >>> intersperse([], 4)
        []
        >>> intersperse([1, 2, 3], 4)
        [1, 4, 2, 4, 3]
        \"\"\"
    """,
    '''Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:

    from typing import List


    def remove_duplicates(numbers: List[int]) -> List[int]:
    """ From a list of integers, remove all elements that occur more than once.
    Keep order of elements left the same as in the input.
    >>> remove_duplicates([1, 2, 3, 2, 4])
    [1, 3, 4]
    """
    ''',
    '''Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:

    def sort_third(l: list):
    """This function takes a list l and returns a list l' such that
    l' is identical to l in the indicies that are not divisible by three, while its values at the indicies that are divisible by three are equal
    to the values of the corresponding indicies of l, but sorted.
    >>> sort_third([1, 2, 3])
    [1, 2, 3]
    >>> sort_third([5, 6, 3, 4, 8, 9, 2])
    [2, 6, 3, 4, 8, 9, 5]
    """
    ''',
    '''
    def solve(N):
    """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>assistant\n'
    
    Think step by step: please provide an efficient and self-contained Python script that solves the following problem in a markdown code block:
    
    Given a positive integer N, return the total sum of its digits in binary.

    Example
    For N = 1000, the sum of digits will be 1 the output should be "1".
    For N = 150, the sum of digits will be 6 the output should be "110".
    For N = 147, the sum of digits will be 12 the output should be "1100".

    Variables:
    @N integer
    Constraints: 0 ≤ N ≤ 10000.
    Output:
    a string of binary number
    """
    ''',
]

# %%
for prompt in test_prompts[-1:]:
    print(prompt)
    # tokenized_prompt = model.tokenizer.apply_chat_template(prompt)
    print(model.generate(prompt, max_new_tokens=100, temperature=0.7))
    print("\n")

# %%

tokens = hf_tokenizer.apply_chat_template(test_prompts[-1], add_generation_prompt=True)
# %%
hf_tokenizer.decode(tokens)
# %%
model.to_str_tokens(np.array(tokens))
# %%
[1, 2, 3, 4, 5][:-1]

# %%
def generate_response(messages, think_mode=True):
    text = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=think_mode
    )
    
    if not think_mode:
        text = "".join(text.split("<|im_end|>")[:-1])  # remove the end of assistant message, so it continues the NoThink prompt

    print(text)
    model_inputs = hf_tokenizer([text], return_tensors="pt").to(device)
    generated_ids = hf_model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return hf_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
# %%
HUMAN_EVAL_PROMPT_TEMPLATE = "Please provide an efficient and self-contained Python script that solves the following problem in a markdown code block in an <answer> tag. Think through the problem step by step in a <think> tag."
tricky_examples = {
    64: '''def pluck(arr):
"""Given an array representing a branch of a tree that has non-negative integer nodes
your task is to pluck one of the nodes and return it.
The plucked node should be the node with the smallest even value.
If multiple nodes with the same smallest even value are found return the node that has smallest index.

The plucked node should be returned in a list, [ smalest_value, its index ],
If there are no even values or the given array is empty, return [].

Example 1:
Input: [4,2,3]
Output: [2, 1]
Explanation: 2 has the smallest even value, and 2 has the smallest index.

Example 2:
Input: [1,2,3]
Output: [2, 1]
Explanation: 2 has the smallest even value, and 2 has the smallest index.

Example 3:
Input: []
Output: []

Example 4:
Input: [5, 0, 3, 0, 4, 2]
Output: [0, 1]
Explanation: 0 is the smallest value, but there are two zeros,
so we will choose the first zero, which has the smallest index.

Constraints:
* 1 <= nodes.length <= 10000
* 0 <= node.value
"""''',
    84: '''def solve(N):
"""Given a positive integer N, return the total sum of its digits in binary.

Example
For N = 1000, the sum of digits will be 1 the output should be "1".
For N = 150, the sum of digits will be 6 the output should be "110".
For N = 147, the sum of digits will be 12 the output should be "1100".

Variables:
@N integer
Constraints: 0 ≤ N ≤ 10000.
Output:
a string of binary number
"""''',
    95: '''def check_dict_case(dict):
"""
Given a dictionary, return True if all keys are strings in lower
case or all keys are strings in upper case, else return False.
The function should return False is the given dictionary is empty.
Examples:
check_dict_case({"a":"apple", "b":"banana"}) should return True.
check_dict_case({"a":"apple", "A":"banana", "B":"banana"}) should return False.
check_dict_case({"a":"apple", 8:"banana", "a":"apple"}) should return False.
check_dict_case({"Name":"John", "Age":"36", "City":"Houston"}) should return False.
check_dict_case({"STATE":"NC", "ZIP":"12345" }) should return True.
"""''',
}
# %%

for task_id, prompt in list(tricky_examples.items())[:1]:
    print(f"### TASK {task_id}:")
    print(f"#### No thinking")
    nothink_messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": HUMAN_EVAL_PROMPT_TEMPLATE + prompt},
        {"role": "assistant", "content": "<think>Okay I have finished thinking.</think><answer>"}
    ]
    print(generate_response(nothink_messages, think_mode=False)[0])

    print(f"#### With thinking")
    think_messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": HUMAN_EVAL_PROMPT_TEMPLATE + prompt},
    ]
    print(generate_response(think_messages, think_mode=True)[0])

# %%
