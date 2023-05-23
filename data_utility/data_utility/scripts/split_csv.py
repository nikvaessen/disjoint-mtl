#! /usr/bin/env python3
########################################################################################
#
# Split a CSV file describing a dataset into two disjoint splits based on splitting
# among a particular axis. This can be useful for generating a train/val split or
# creating smaller subsets of the dataset.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

from typing import Tuple

import click
import numpy.random

import pandas as pd

from data_utility.util.csv import ShardCsvSample


########################################################################################
# split strategies


def split_by_speakers(
    df: pd.DataFrame, cut_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _split_by_column_name(df, cut_ratio, seed, "speaker_id")


def split_by_speakers_equal(
    df: pd.DataFrame, cut_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _split_by_column_name_with_gender(
        df, cut_ratio, seed, "speaker_id", "gender"
    )


def split_by_recordings(
    df: pd.DataFrame, cut_ratio: float, seed: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return _split_by_column_name(df, cut_ratio, seed, "recording_id")


def _split_by_column_name(
    df: pd.DataFrame,
    cut_ratio: float,
    seed: int,
    column_name: str,
):
    rng = numpy.random.default_rng(seed)

    all_values = df[column_name].unique().tolist()
    total_num_samples = len(df)

    split_values = []
    num_split_values = 0

    while num_split_values / total_num_samples < cut_ratio:
        random_value = all_values.pop(rng.integers(0, len(all_values)))

        split_values.append(random_value)
        num_split_values += len(df.loc[df[column_name] == random_value])

    remain_values = all_values

    # select splits
    remain_df = df.loc[df[column_name].isin(remain_values)]
    split_df = df.loc[df[column_name].isin(split_values)]

    return remain_df, split_df


def _split_by_column_name_with_gender(
    df: pd.DataFrame,
    cut_ratio: float,
    seed: int,
    column_name: str,
    gender_column_name: str,
):
    rng = numpy.random.default_rng(seed)

    all_male_df = df.loc[df[gender_column_name] == "m"]
    all_female_df = df.loc[df[gender_column_name] == "f"]

    all_male_values = all_male_df[column_name].unique().tolist()
    all_female_values = all_female_df[column_name].unique().tolist()

    total_num_samples = len(df)

    split_values = []
    num_split_values = 0

    while num_split_values / total_num_samples < cut_ratio:
        random_value_male = all_male_values.pop(rng.integers(0, len(all_male_values)))
        random_value_female = all_female_values.pop(
            rng.integers(0, len(all_female_values))
        )

        split_values.append(random_value_male)
        split_values.append(random_value_female)

        num_split_values += len(df.loc[df[column_name] == random_value_male])
        num_split_values += len(df.loc[df[column_name] == random_value_female])

    remain_values = [] + all_male_values + all_female_values

    # select splits
    remain_df = df.loc[df[column_name].isin(remain_values)]
    split_df = df.loc[df[column_name].isin(split_values)]

    return remain_df, split_df


########################################################################################
# entrypoint of script


@click.command()
@click.argument(
    "csv_in",
    nargs=-1,
    type=pathlib.Path,
    required=True,
)
@click.option(
    "--strategy",
    "split_strategy",
    type=click.Choice(["by_speakers", "by_speakers_equal", "by_recordings"]),
    required=True,
    help="Which strategy to use to split the data",
)
@click.option(
    "--ratio",
    "cut_ratio",
    type=float,
    required=True,
    help="the ratio split (cut) from the input data",
)
@click.option(
    "--remain-out",
    type=pathlib.Path,
    help="path to file where remaining data is written",
    required=True,
)
@click.option(
    "--split-out",
    type=pathlib.Path,
    help="path to file where split data is written",
    required=True,
)
@click.option(
    "--delete-in",
    is_flag=True,
    default=False,
    help="delete original input file(s) after split",
)
@click.option(
    "--seed", "random_seed", default=1337, type=int, help="random seed used to split"
)
def main(
    csv_in: Tuple[pathlib.Path],
    split_strategy: str,
    cut_ratio: float,
    remain_out: pathlib.Path,
    split_out: pathlib.Path,
    delete_in: bool,
    random_seed: int,
):
    # echo what is going to happen
    print(
        f"splitting {[str(c) for c in csv_in]} with {split_strategy=} to \n"
        f"\tremain={str(remain_out)}\n"
        f"\tsplit={str(split_out)}"
    )

    # load and potentially merge CSV files
    df_collection = []

    for csv in csv_in:
        df_collection.append(ShardCsvSample.from_csv(csv))

    df = pd.concat(df_collection)

    # determine and apply split strategy
    if split_strategy == "by_recordings":
        remain_df, split_df = split_by_recordings(df, cut_ratio, random_seed)
    elif split_strategy == "by_speakers":
        remain_df, split_df = split_by_speakers(df, cut_ratio, random_seed)
    elif split_strategy == "by_speakers_equal":
        remain_df, split_df = split_by_speakers_equal(df, cut_ratio, random_seed)
    else:
        raise ValueError(f"unknown {split_strategy=}")

    # assert no overlap
    intersection = set(remain_df["key"]).intersection(set(split_df["key"]))
    assert len(intersection) == 0

    # write files
    if delete_in:
        for csv in csv_in:
            csv.unlink(missing_ok=True)

    ShardCsvSample.to_csv(remain_out, remain_df)
    ShardCsvSample.to_csv(split_out, split_df)

if __name__ == "__main__":
    main()
