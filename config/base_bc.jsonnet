local dataset_type = null;
local pretrained_embedding_file = null;

local seed = 19950815;
local batch_size = 32;
local num_epochs = 8;
local valid_metric = "+auc";

local word_embed_dim = 300;
local hidden_dim = 128;

local dataset_reader = {
    "type": dataset_type,
    "tokenizer": {
        "type": "whitespace_vocab_filter",
        "vocab_file": null,
    },
    "token_indexers": {
        "tokens": {
            "type": "single_id",
            "lowercase_tokens": false
        },
    },
    "lazy": true
};

local word_embed = {
    "type": "basic",
    "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": word_embed,
            "pretrained_file": pretrained_embedding_file,
            "trainable": true,
        },
    },
};

local bi_lstm_encoder = {
    "type": "lstm",
    "input_size": word_embed_dim,
    "hidden_size": hidden_dim,
    "num_layers": 1,
    "bidirectional": true,
};

local model = {
    "type": "bc_weighted",
    "word_embed": word_embed,
    "encoder": bi_lstm_encoder,
    "attention_layer": {
        "type": "bc_tanh",
        "hidden_size": 256,
    },
};

local data_loader = {
    "num_workers": 1,
    "batch_sampler": {
        "type": "bucket",
        "batch_size": batch_size,
        "sorting_keys": ["tokens"],
        "padding_noise": 0.2,
    },
};

local optimizer = {
    "type": "adam",
    "parameter_groups": [
        [["_encoder"], {"lr": 0.001, "weight_decay": 1e-5, "amsgrad": true}],
        [["_attention_layer"], {"lr": 0.001, "weight_decay": 0, "amsgrad": true}],
        [["_output"], {"lr": 0.001, "weight_decay": 1e-5, "amsgrad": true}],
    ],
};

local trainer = {
    "optimizer": optimizer,
    "validation_metric": "+auc",
    "num_epochs": num_epochs,
    "cuda_device": std.parseInt(std.extVar("GPU")),
};

local vocabulary = {
    "only_include_pretrained_words": true,
};

{
    "random_seed": seed,
    "numpy_seed": seed,
    "pytorch_seed": seed,
    "dataset_reader": dataset_reader,
    "train_data_path": "train",
    "validation_data_path": "dev",
    "test_data_path": "test",
    "evaluate_on_test": true,
    "model": model,
    "data_loader": data_loader,
    "trainer": trainer,
    "vocabulary": vocabulary,
}
