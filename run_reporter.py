#!/usr/bin/python

"""Script to run the Reporter to process an entire folder"""
import argparse
import datetime
from pathlib import Path

from raindownloader.inpeparser import INPEParsers
from rainreporter.reporter import Reporter


def main(hot_folder: Path, output_folder: Path):
    """Run the reporter on a specific folder"""
    # create a reporter instance
    try:
        reporter = Reporter(
            server=INPEParsers.FTPurl,
            download_folder="/workspaces/INPERainDownloader/tmp",
            config_file="/workspaces/RainReporter/rainreporter/reporter.json5",
        )

        # once the reporter is created, we will call it to process the hotfolder
        reporter.process_folder(hot_folder=hot_folder, output_folder=output_folder)
        # Log a message to a file to confirm that the script was executed
        with open("/file.log", "a") as f:
            f.write(f"{datetime.datetime.now()}: Script executed successfully\n")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Description of the run_reporter script"
    )
    parser.add_argument(
        "--hot_folder",
        help="Hot folder that contains the .json5 files to be processed",
        required=True,
    )
    parser.add_argument(
        "--output_folder", help="Output folder to store the reports", required=True
    )
    args = parser.parse_args()

    # print("Contents of hot folder")
    # print(list(Path(args.hot_folder).iterdir()))

    # print("Contents of the output folder")
    # print(list(Path(args.output_folder).iterdir()))
    print(datetime.datetime.now())
    main(hot_folder=Path(args.hot_folder), output_folder=Path(args.output_folder))
