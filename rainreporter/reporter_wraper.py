"""Module docstring"""
import io
from pathlib import Path
from typing import Union

import pyjson5


from pypdf import PdfMerger, PdfReader

from raindownloader.inpeparser import INPEParsers
from raindownloader.utils import DateProcessor

from .reporter import Reporter


def create_monthly_report(
    reporter: Reporter, report: dict, output_folder: Union[str, Path]
):
    """If output_folder is None, return a BytesIO"""

    today = DateProcessor.today()

    if "data" in report:
        date = DateProcessor.parse_date(report["data"])
    else:
        date = today

    axs, rain_ts, lta, shp = reporter.monthly_anomaly_report(
        date_str=DateProcessor.pretty_date(date),
        shapefile=report["shp"],
    )

    # adjust the title and sub_title
    date_str = DateProcessor.pretty_date(date)[-7:]
    title = f"Bacia: {report['nome']} / Mês: {date_str}"
    fig = axs[0].figure
    fig.suptitle(title, y=1.1, fontsize=14)

    subtitle = f"Relatório gerado em: {DateProcessor.pretty_date(today)}\n"

    # check if we are in the current month
    if today.month == date.month:
        subtitle += "* Chuva acumulada no mês atual até último dia disponível."

    fig.text(0.01, 1.06, subtitle, ha="left", va="top", fontsize=12)

    # if output_folder:
    # filename = report["nome"].replace(" ", "_")
    # file = Path(output_folder) / f"{filename}_{date_str}.pdf"
    # axs[0].figure.savefig(file, bbox_inches="tight", pad_inches=0.5)
    # else:
    file = io.BytesIO()
    axs[0].figure.savefig(file, bbox_inches="tight", pad_inches=0.6, format="pdf")

    return file


def save_report(reporter, report_config, output_folder):
    """Saves one PDF report. Each report may have multiple pages"""

    # check if there is a date in the report
    if not report_config["data"]:
        report_config["data"] = DateProcessor.pretty_date(DateProcessor.today())

    pdf_doc = PdfMerger()
    for report in report_config["relatorios"]:
        print(f"Processando relatório {report['nome']}")

        if report["tipo"] == "Mensal":
            # set the date for the report
            if "data" not in report:
                report["data"] = report_config["data"]

            file = create_monthly_report(reporter, report, output_folder)
            pdf_doc.append(PdfReader(file))

        else:
            print("Não implementado")

    filename = f"{report_config['arquivo']}_{report_config['data']}.pdf"
    pdf_doc.write(output_folder / filename)


def run_reports(
    config_folder: Union[str, Path],
    download_folder: Union[str, Path],
    output_folder: Union[str, Path],
    report_config: Path,
):
    """ "Docstring"""

    # Initialize a reporter object and folders
    reporter = Reporter(
        server=INPEParsers.FTPurl,
        download_folder=download_folder,
        parsers=INPEParsers.parsers,
        post_processors=INPEParsers.post_processors,
        config_file=report_config,
    )
    config_folder = Path(config_folder)
    output_folder = Path(output_folder)

    # get the files to be processed
    # all .json file in the config folder will be used
    files = list(config_folder.glob("*.json5"))

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            report_config = pyjson5.decode_io(f)  # pylint: disable=no-member
            save_report(reporter, report_config, output_folder)
