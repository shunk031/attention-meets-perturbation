import argparse

from allennlp.commands.subcommand import Subcommand
from overrides import overrides

from amep.commands.make_dataset.bc.ag_news import make_ag_news_dataset
from amep.commands.make_dataset.bc.imdb import make_imdb_dataset
from amep.commands.make_dataset.bc.newsgroups import make_newsgroups_dataset
from amep.commands.make_dataset.bc.sst import make_sst_dataset
from amep.commands.make_dataset.nli.multi_nli import make_multi_nli_dataset
from amep.commands.make_dataset.nli.snli import make_snli_dataset
from amep.commands.make_dataset.qa.babi import make_babi_dataset
from amep.commands.make_dataset.qa.cnn import make_cnn_dataset

DATASET_MAKER = {
    #
    # For BC
    #
    "sst": make_sst_dataset,
    "imdb": make_imdb_dataset,
    "newsgroups": make_newsgroups_dataset,
    "ag_news": make_ag_news_dataset,
    #
    # For QA
    #
    "cnn": make_cnn_dataset,
    "babi": make_babi_dataset,
    #
    # For NLI
    #
    "snli": make_snli_dataset,
    "multi_nli": make_multi_nli_dataset,
}


@Subcommand.register("make-dataset")
class MakeDataset(Subcommand):
    @overrides
    def add_subparser(
        self, parser: argparse._SubParsersAction
    ) -> argparse.ArgumentParser:

        description = "Make dataset for experiments"
        subparser = parser.add_parser(self.name, description=description)

        dataset_choices = list(DATASET_MAKER.keys()) + ["all"]
        subparser.add_argument(
            "dataset", type=str, help="dataset type", choices=dataset_choices
        )
        subparser.add_argument(
            "--num-worker", type=int, help="number of workers", default=4
        )

        subparser.set_defaults(func=make_dataset)
        return subparser


def make_dataset(args: argparse.Namespace) -> None:

    if args.dataset == "all":
        for maker in DATASET_MAKER.values():
            maker(args)
    else:
        maker = DATASET_MAKER[args.dataset]
        maker(args)
