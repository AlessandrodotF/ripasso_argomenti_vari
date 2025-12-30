import csv
import json
from typing import Dict, Any, Callable
from functools import wraps

Data = Dict[str, Any]
ExportFn = Callable[[Data], None]

exporters: Dict[str, ExportFn] = {}


def register_exporter(format: str) -> Callable[[ExportFn], ExportFn]:
    def decorator(fn: ExportFn) -> ExportFn:
        @wraps(fn)
        def wrapper(sample_data: Data) -> None:
            return fn(Data)

        exporters[format] = wrapper
        return wrapper

    return decorator


@register_exporter("json")
def export_json(sample_data: Data) -> None:
    print(f"Exporting data to json: {sample_data}")


@register_exporter("csv")
def export_csv(sample_data: Data) -> None:
    print(f"Exporting data to csv: {sample_data}")


@register_exporter("pdf")
def export_pdf(sample_data: Data) -> None:
    print(f"Exporting data to pdf: {sample_data}")


@register_exporter("xml")
def export_xml(sample_data: Data) -> None:
    print(f"Exporting data to xml: {sample_data}")


# qui c'è ancora un approccio troppo
# del tipo : se ho 100 exporters allora avro un dizioanrio enorme


def export_data(sample_data: Data, format: str):
    exporter = exporters.get(format)
    if exporter is None:
        raise ValueError(f"!!!!!!!No exporter found for format {format}")
    exporter(sample_data)


def main() -> None:
    sample_data: Data = {"name": "Alice", "Age": 30}
    export_data(sample_data, "csv")
    return "saved!"


if __name__ == "__main__":
    main()
    a = [1, 2, 3]
    a[0] = 10
    print(a)

    a = (1, 2, 3)
    a[0] = 10
    print(a)
