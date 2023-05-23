import pathlib
import click

import pandas as pd


@click.command()
@click.option("--vox1_meta_path", type=pathlib.Path, required=True)
@click.option("--vox2_meta_path", type=pathlib.Path, required=True)
@click.option("--vox_merged_meta_path", type=pathlib.Path, required=True)
def main(
    vox1_meta_path: pathlib.Path,
    vox2_meta_path: pathlib.Path,
    vox_merged_meta_path: pathlib.Path,
):
    # vox1 has \t as separator if not fixed yet
    df_vox1: pd.DataFrame = pd.read_csv(vox1_meta_path, delimiter="\t")

    if len(df_vox1.columns) == 5:
        df_vox1 = df_vox1.rename(
            columns={
                "VoxCeleb1 ID": "voxceleb_id",
                "VGGFace1 ID": "vggface_id",
                "Gender": "gender",
                "Nationality": "nationality",
                "Set": "set",
            }
        )

        # add voxceleb1 ID column
        df_vox1['dataset'] = pd.Series(['vox1' for _ in range(len(df_vox1))])

        print(f"writing fixed csv file format to {vox1_meta_path=}")
        df_vox1.to_csv(vox1_meta_path, index=False)

    # load fixed version
    df_vox1: pd.DataFrame = pd.read_csv(vox1_meta_path)

    # vox2 has "," as seperator
    df_vox2: pd.DataFrame = pd.read_csv(
        vox2_meta_path,
        delimiter=",",
    )

    if "VoxCeleb2 ID " in df_vox2.columns.values:
        df_vox2 = df_vox2.rename(
            columns={
                "VoxCeleb2 ID ": "voxceleb_id",
                "VGGFace2 ID ": "vggface_id",
                "Gender ": "gender",
                "Set ": "set",
            }
        )

        for c in df_vox2.columns:
            df_vox2[c] = df_vox2[c].str.strip()

        df_vox2['dataset'] = pd.Series(['vox2' for _ in range(len(df_vox2))])

        print(f"writing fixed csv file format to {vox2_meta_path=}")
        df_vox2.to_csv(vox2_meta_path, index=False)

    print(f"writing the merged CSV file to {vox_merged_meta_path=}")
    pd.concat([df_vox1, df_vox2]).to_csv(vox_merged_meta_path, index=False)


if __name__ == "__main__":
    main()
