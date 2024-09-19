import os
import csv
import numpy as np
import math
from collections import OrderedDict

promt_role = """
Act as a Cryptographer, I have a ciphertext that is encrypted by one of these 5 ciphers: Baconian, Autokey, Gronsfeld, Vigenere, Railfence, \
and you job is to analyse the codes and the examples first and than identify which type of cipher used to encrypt the text. 
"""

prompt_python = """
Here are the python codes for encryption of each type of cipher, including Baconian, Autokey, Gronsfeld, Vigenere, Railfence, which are put in "<<<" and ">>>".
<<<
def encrypt_Baconian(self, plaintext, key):
    ciphertext = []
    for p in plaintext:
        for k in key[p]:
            if k < 13:
                r = random.randint(0, 12)
            else:
                r = random.randint(13, 25)
            if r in (9, 21):  # remove j and v
                r -= 1
            ciphertext.append(r)
    return np.array(ciphertext)


def encrypt_Autokey(self, message, key):
    cipher = []
    k_index = 0
    # here the key has to be converted back to a list as it must be extended.
    key = list(key)
    for i in message:
        text = i
        text += key[k_index]
        key.append(i)  # add current char to keystream
        text %= len(self.alphabet)
        k_index += 1
        cipher.append(text)
    return np.array(cipher)

def encrypt_Gronsfeld(self, plaintext, key):
    ciphertext = []
    for i, p in enumerate(plaintext):
        ciphertext.append((p + key[i % len(key)]) % len(self.alphabet))
    return np.array(ciphertext)

def encrypt_Vigenere(self, plaintext, key):
    key_length = len(key)
    ciphertext = []
    for position in range(len(plaintext)):
        p = plaintext[position]
        if p >= len(self.alphabet):
            ciphertext.append(self.unknown_symbol_number)
            continue
        shift = key[(position - ciphertext.count(self.unknown_symbol_number)) % key_length]
        c = (p + shift) % len(self.alphabet)
        ciphertext.append(c)
    return np.array(ciphertext)

def encrypt_Railfence(self, plaintext, key):
    ciphertext = []
    row_size = len(key[0])
    rows = [[] for _ in range(row_size)]
    pos = 0
    direction = 1
    for i in range(len(plaintext) + key[1]):
        if i >= key[1]:
            rows[pos].append(plaintext[i-key[1]])
        pos += 1 * direction
        if pos in (row_size - 1, 0):
            direction = direction * -1
    for i in range(len(rows)):
        ciphertext += rows[np.where(key[0] == i)[0][0]]
    return np.array(ciphertext)
>>>
"""

prompt_intruction = """
First, you should analyze the python code and all of the characteristics of each ciphertext in the examples I gave you by comparing them with each other. \
For each ciphertext, you should analyse the average and distribution of each characteristic, including Index of Coincidence, 1-Gram Frequency, Chi-Square. 
Than I will give you the ciphertext and \
your task is identify which type of cipher used in 5 ciphers in example and give me the explaination. \
I don't want to explore how these ciphers are implemented, or are trying to decrypt these texts, \
don't require more information except what I give you, just give me the final answer and the full explaination. 
"""

prompt_requirement = """
Which type of cipher used to encrypt the text above and \
there are only 5 options: Baconian, Autokey, Gronsfeld, Vigenere, Railfence, \
Please provide a comprehensive response of at least 2000 words and focus on analyse the examples
"""

folder_path = "mtc3_cipher_id/ciphertexts"
output_csv = "output.csv"
#OUTPUT_ALPHABET = 'abcdefghijklmnopqrstuvwxyz #0123456789'
OUTPUT_ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


def calculate_index_of_coincidence(text):
    n = [0]*len(OUTPUT_ALPHABET)
    for p in text:
        if p in OUTPUT_ALPHABET:
            n[OUTPUT_ALPHABET.index(p)] = n[OUTPUT_ALPHABET.index(p)] + 1    
    coindex = 0
    for i in range(0, len(OUTPUT_ALPHABET)):
        coindex = coindex + n[i] * (n[i] - 1) / len(text) / (len(text) - 1)
    return coindex

def calculate_frequencies(text, size=1, recursive=False):
    before = []
    if recursive and size > 1:
        before = calculate_frequencies(text, size-1, recursive)
    frequencies_size = int(math.pow(len(OUTPUT_ALPHABET), size))
    frequencies = [0] * frequencies_size
    indices = [OUTPUT_ALPHABET.index(char) for char in text if char in OUTPUT_ALPHABET]
    
    for p in range(len(indices) - (size-1)):
        pos = 0
        for i in range(size):
            pos += indices[p + i] * int(math.pow(len(OUTPUT_ALPHABET), i))
        frequencies[pos] += 1
    
    # Normalize frequencies
    total = sum(frequencies)
    if total > 0:
        frequencies = [f / total for f in frequencies]
    
    return before + frequencies

def print_frequencies(frequencies, size=1):
    n_gram_strings = []
    for i in range(len(frequencies)):
        n_gram = ''
        value = i
        for _ in range(size):
            n_gram = OUTPUT_ALPHABET[value % len(OUTPUT_ALPHABET)] + n_gram
            value //= len(OUTPUT_ALPHABET)
        n_gram_strings.append(n_gram)
    
    output = ""
    for n_gram, freq in zip(n_gram_strings, frequencies):
        output += f" {n_gram} - {freq:.4f},"
    output = output[:-1]
    return output


def calculate_chi_square(frequencies):
    english_frequencies = [
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015, 0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749, 0.07507,
    0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758, 0.00978, 0.0236, 0.0015, 0.01974, 0.00074]
    chi_square = 0
    for i in range(len(frequencies)):
        chi_square = chi_square + (
                    (english_frequencies[i] - frequencies[i]) * (english_frequencies[i] - frequencies[i])) / english_frequencies[i]
    return chi_square / 100

with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Type of Cipher', 'Ciphertext', 'Index of coincidence', '1-gram frequency', 'chi_square'])

    # Iterate over each file in the directory
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            cipher_type = filename.split('-')[1].replace('.txt', '')

            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                lines = file.readlines()

                for line in lines[:10]:
                    ciphertext = line.strip()
                    ioc_value = calculate_index_of_coincidence(ciphertext)
                    frequency = calculate_frequencies(ciphertext)
                    frequency_print = print_frequencies( frequency)
                    chi_square = calculate_chi_square(frequency)
                    csv_writer.writerow([cipher_type, ciphertext, ioc_value, frequency_print, chi_square])


prompt_example = "I will give the characteristics of each ciphertext including:" 
def example_generate(_ciphertext = False, _IoC = False, _Frequency = False, _Chi_square = False): 
    global prompt_example
    if (_IoC): prompt_example += " Index of Coincidence,"
    if (_Frequency): prompt_example += " 1-Gram Frequency,"
    if (_ciphertext): prompt_example += " ciphertexts,"
    if (_Chi_square): prompt_example += " Chi Square,"
    prompt_example = prompt_example[:-1] + "."
    if not _ciphertext: prompt_example+= " But will not include the ciphertexts in the examples. \n"
    with open(output_csv, mode='r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            prompt_example += f" Type of Cipher: {row[0]}; "
            if (_ciphertext) : prompt_example += f"Ciphertext: {row[1]}; "
            if (_IoC) : prompt_example += f"IoC: {row[2]}; "
            if (_Frequency) : prompt_example += f"1-Gram Frequency: {row[3]}; "
            if (_Chi_square) : prompt_example += f"Chi Square: {row[4]}; "
            prompt_example += "\n"

# write the examples for each cipher hier
example_generate(_ciphertext = False, _IoC = True, _Frequency = True, _Chi_square = True)

# write the ciphertext hier
ciphertext_ = "frvnatfnbqsgynunnqqwbeinzjxnubjdvcauojnzxoelwrpenpjnfrdjigbiytnrvreacamesrvjllamguxnlyhroiqsrqcriyrb"
ciphertext_IoC = calculate_index_of_coincidence(ciphertext_)
ciphertext_frequency = print_frequencies(calculate_frequencies(ciphertext_))
ciphertext_Chi_Square = calculate_chi_square(calculate_frequencies(ciphertext_))
prompt_ciphertext = f"""
The ciphertext : 
<<<
{ciphertext_}
>>>
The Index of Coincidence of the ciphertext is {ciphertext_IoC}, the Chi-Square of the ciphertext is: {ciphertext_Chi_Square}, the 1-Gram Frequency of the ciphertext is:
{ciphertext_frequency}. 
"""

# write the prompt in file output_prompt.txt
output_prompt = 'output_prompt.txt'
prompt_text = promt_role + prompt_python + prompt_example + prompt_intruction + prompt_ciphertext + prompt_requirement
with open(output_prompt, 'w', encoding='utf-8') as file:
    file.write(prompt_text)