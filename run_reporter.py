#!/usr/bin/python

"""Script to run the Reporter to process an entire folder"""
import argparse
import datetime
from pathlib import Path

from raindownloader.inpeparser import INPEParsers
from rainreporter.reporter import Reporter


def main(
    configs_folder: Path,
    bases_folder: Path,
    downloads_folder: Path,
    output_folder: Path,
):
    """Run the reporter on a specific folder"""
    # create a reporter instance
    try:
        config_file = Path(__file__).parent / "reporter.json5"

        reporter = Reporter(
            server=INPEParsers.FTPurl,
            download_folder=downloads_folder,
            config_file=config_file,
            bases_folder=bases_folder,
        )

        # once the reporter is created, we will call it to process the hotfolder
        reporter.process_folder(
            hot_folder=configs_folder,
            output_folder=output_folder,
        )
        # Log a message to a file to confirm that the script was executed
        with open("/file.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.now()}: Script executed successfully\n")

    except Exception as error:  # pylint: disable=W0703
        print(error)


if __name__ == "__main__":
    DESCRIPTION = (
        "Description of the run_reporter script. \n "
        "The RainReporter requires 4 folders to be set to work properly. They are:\n"
        "* configs: the folder where are the .json5 files with each PDF specification\n"
        "* download: the folder to download the temporary files from INPE\n"
        "* output: the folder where to put the output reports\n"
        "* bases (optional): the folder where are the shapes refered in the configs.\n\n"
        "if bases is set, the .json5 can use relative paths, otherwise, they must point to\n"
        "absolute paths.\n\n"
        "To make things easier, we can use the master_folder argument. With the master folder, all \n"
        "other folders will be relative to it, and if we follow the nomenclature, it is not necessary\n"
        "to pass any other argument. \n"
        "/path/to/master_folder/bases\n "
        "                      /configs\n"
        "                      /downloads\n"
        "                      /output\n"
    )

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--master_folder",
        help="Base folder to attach other folders relatively to it",
        required=False,
    )
    parser.add_argument(
        "--configs", help="Configs folder", required=False, default="configs"
    )
    parser.add_argument(
        "--bases",
        help="Bases folder to store the shapefiles",
        required=False,
        default="bases",
    )
    parser.add_argument(
        "--downloads",
        help="Temporary downloads folder",
        required=False,
        default="downloads",
    )
    parser.add_argument(
        "--output", help="Output folder", required=False, default="output"
    )

    print(datetime.datetime.now())
    args = parser.parse_args()

    # if master_folder is provided, consider all folders as relative to it
    if args.master_folder is not None:
        args.master_folder = Path(args.master_folder)
        for arg in vars(args):
            if arg == "master_folder":
                continue

            value = Path(getattr(args, arg))

            # if the value is not absolute
            if not value.is_absolute():
                setattr(args, arg, args.master_folder / value)

    # make sure everything is set as Path
    for arg, value in vars(args).items():
        if value is not None:
            setattr(args, arg, Path(value))

            print(f"{arg}: {value} -> {'ok' if Path(value).exists() else 'FAILED'}")

    # print(args)

    main(
        configs_folder=args.configs,
        output_folder=args.output,
        downloads_folder=args.downloads,
        bases_folder=args.bases,
    )
