local dataset_type = null;
local pretrained_embedding_file = null;

local seed = 19950815;
local batch_size = 32;
local num_epochs = 10;
local valid_metric = "+acc1";

local word_dim = 300;
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
        }
    },
    "lazy": true
};

local word_embed = {
    "type": "basic",
    "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": word_dim,
            "pretrained_file": pretrained_embedding_file,
            "trainable": true
        },
    },
};

local paragraph_encoder = {
    "type": "lstm",
    "input_size": word_dim,
    "hidden_size": hidden_dim,
    "num_layers": 1,
    "bidirectional": true,
};

local question_encoder = {
    "type": "lstm",
    "input_size": word_dim,
    "hidden_size": hidden_dim,
    "num_layers": 1,
    "bidirectional": true,
};

local attention_layer = {
    "type": "qa_tanh",
    "hidden_size": hidden_dim * 2,
};

local model = {
    "type": "qa_vanilla",
    "word_embed": word_embed,
    "paragraph_encoder": paragraph_encoder,
    "question_encoder": question_encoder,
    "attention_layer": attention_layer,
};

local data_loader = {
    "num_workers": 1,
    "batch_size": batch_size,
};

local optimizer = {
    "type": "adam",
    "amsgrad": true,
    "weight_decay": 1e-5
};

local trainer = {
    "optimizer": optimizer,
    "validation_metric": valid_metric,
    "num_epochs": num_epochs,
    "cuda_device": std.parseInt(std.extVar("GPU")),
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
    "trainer": trainer
}
