local dataset_type = "imdb";
local pretrained_embedding_file = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec";
local vocab_file_for_filter = "dataset/%s/vocab.txt" % dataset_type;

local base = import "../base_bc.jsonnet";

base + {
  "dataset_reader"+: {
    "type": dataset_type,
    "tokenizer"+: {
        "vocab_file": vocab_file_for_filter
    },
    "min_length": 6,
    "lazy": false
  },
  "model"+: {
    "word_embed"+: {
      "token_embedders"+: {
        "tokens"+: {
          "pretrained_file": pretrained_embedding_file
        }
      }
    }
  }
}
