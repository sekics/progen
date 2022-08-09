import pandas as pd
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, decoders, trainers

# def load_and_save_data(amp_file, non_amp_file, ratio):
#     pos_data = []
#     with open(amp_file) as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip()
#             if line[0] == '>':
#                 continue
#             else:
#                 pos_data.append([str(line)])
    
#     neg_data = []
#     with open(non_amp_file) as f:
#         lines = f.readlines()
#         for line in lines:
#             line = line.strip()
#             if line[0] == '>':
#                 continue
#             else:
#                 neg_data.append([str(line)])
    
#     data = pos_data + neg_data[: len(pos_data) * ratio]

#     data_csv = pd.DataFrame(data=data, columns=["sequence"])

#     data_csv.to_csv("./data/sequences.csv")

# def generate_tokenizer():
#     tokenizer = Tokenizer(models.BPE())
#     tokenizer.normalizer = normalizers.NFKC()
#     tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
#     tokenizer.post_tokenizer = processors.ByteLevel()
#     tokenizer.decoder = decoders.ByteLevel()
#     trainer = trainers.BpeTrainer(
#         vocab_size=32,

#         special_tokens=["<PAD>", "<BOS>", "<EOS>"]
#     )

#     data = [
#         "ABCDEFGHIJKLMN",
#         "OPQRSTUVWXYZ",
#         "*12"
#     ]


#     tokenizer.train_from_iterator(data, trainer=trainer)
#     tokenizer.enable_padding(pad_id=0, pad_token="<PAD>")
#     tokenizer.enable_truncation(512)
#     tokenizer.save("tokenizer_.json")

# # load_and_save_data("data/AMPs.fa", "data/Non-AMPs.fa", 1)
# generate_tokenizer()

# from toekniers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.pre_tokenizers import ByteLevel


def load_and_save_sequences(amp_file, non_amp_file, ratio):
    pos_seq = []
    with open(amp_file) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '>':
                continue
            else:
                pos_seq.append([line.strip()])

    neg_seq = []

    with open(non_amp_file) as f:
        lines = f.readlines()
        for line in lines:
            if line[0] == '>':
                continue
            else:
                neg_seq.append([line.strip()])

    data = pos_seq + neg_seq[: len(pos_seq) * ratio]
    name = ['seq']

    out = pd.DataFrame(columns=name, data=data)

    out.to_csv('./data/sequences.csv')

def generate_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.post_tokenzier = processors.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=32,
        special_tokens=["<PAD>", "<BOS>", "<EOS>"],
    )

    data = [
        "ABCDEFG",
        "HIJKLMN",
        "OPQRSTUVWXYZ",
        "12*"
    ]
    tokenizer.train_from_iterator(data, trainer=trainer)
    tokenizer.save("tokenizer_.json")

def create_tokenizer_custom(file):
    with open(file, 'r') as f:
        return Tokenizer.from_str(f.read())