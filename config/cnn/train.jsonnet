local dataset_type = "cnn";
local pretrained_embedding_file = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec";
local vocab_file_for_filter = "dataset/%s/vocab.txt" % dataset_type;

local batch_size = 90;

local base = import "../base_qa.jsonnet";

local vocabulary = {
    "type": "from_files",
    "directory": "dataset/cnn/vocabulary"
};

base + {
    "dataset_reader"+: {
        "type": dataset_type,
        "tokenizer"+: {
            "vocab_file": vocab_file_for_filter
        }
    },
    "model"+: {
        "word_embed"+: {
            "token_embedders"+: {
                "tokens"+: {
                    "pretrained_file": pretrained_embedding_file
                }
            }
        }
    },
    "data_loader"+: {
        "batch_size": batch_size,
    },
    "vocabulary"+: vocabulary,
}
