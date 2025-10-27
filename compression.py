import bz2
import gzip
import lzma
import pickle
import shutil
import snappy
import sys
import zstandard
import pandas as pd

from config import config_get
from helper import function_execution_in_milliseconds

file_to_compress = config_get('database')
compression_params = config_get('compression_parameters')


def compression_wrapper(data: pd.DataFrame, method: str, compressed_columns: list[str]) -> tuple[float, float, float, float, float]:
    original_data = df_to_bytes(data[compressed_columns])

    compressed_data, compression_time = function_execution_in_milliseconds(_compress_data, original_data, method)
    decompressed_data, decompression_time = function_execution_in_milliseconds(_decompress_data, compressed_data,
                                                                               method)

    original_size = sys.getsizeof(original_data)
    compressed_size = sys.getsizeof(compressed_data)
    decompressed_size = sys.getsizeof(decompressed_data)

    return compression_time, decompression_time, original_size, compressed_size, decompressed_size


def _compress_data(uncompressed_data: bytes, method: str) -> bytes:
    match method:
        case 'lzma':
            return lzma.compress(uncompressed_data)
        case 'gzip':
            return gzip.compress(uncompressed_data)
        case 'bzip2':
            return bz2.compress(uncompressed_data)
        case 'zstd':
            return zstandard.compress(uncompressed_data)
        case 'snappy':
            return snappy.compress(uncompressed_data)
        case _:
            raise ValueError("Please select a valid compression algorithm and try again. [lzma, lzw]")


def _compress(uncompressed_path: str, compressed_path: str):
    with open(uncompressed_path, "rb") as uncompressed_file:
        match compression_params["method"]:
            case 'lzma':
                with lzma.open(compressed_path, "wb") as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
            case 'gzip':
                with gzip.open(compressed_path, "wb") as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
            case 'bzip2':
                with bz2.open(compressed_path, "wb") as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
            case 'zstd':
                with zstandard.open(compressed_path, "wb") as compressed_file:
                    shutil.copyfileobj(uncompressed_file, compressed_file)
            case 'snappy':
                with open(compressed_path, "wb") as compressed_file:
                    snappy.stream_compress(uncompressed_file, compressed_file)
            case _:
                raise ValueError("Please select a valid compression algorithm and try again. [lzma, lzw]")


def _decompress_data(compressed_data: bytes, method: str) -> bytes:
    match method:
        case 'lzma':
            return lzma.decompress(compressed_data)
        case 'gzip':
            return gzip.decompress(compressed_data)
        case 'bzip2':
            return bz2.decompress(compressed_data)
        case 'zstd':
            return zstandard.decompress(compressed_data)
        case 'snappy':
            return snappy.decompress(compressed_data)
        case _:
            raise ValueError("Please select a valid compression algorithm and try again. [lzma, lzw]")


def _decompress(compressed_path: str, expanded_path: str):
    match compression_params["method"]:
        case 'lzma':
            with lzma.open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    shutil.copyfileobj(compressed_file, uncompressed_file)
        case 'gzip':
            with gzip.open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    shutil.copyfileobj(compressed_file, uncompressed_file)
        case 'bzip2':
            with bz2.open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    shutil.copyfileobj(compressed_file, uncompressed_file)
        case 'zstd':
            with zstandard.open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    shutil.copyfileobj(compressed_file, uncompressed_file)
        case 'snappy':
            with open(compressed_path, "rb") as compressed_file:
                with open(expanded_path, "wb") as uncompressed_file:
                    snappy.stream_decompress(compressed_file, uncompressed_file)
        case _:
            raise ValueError("Please select a valid compression algorithm and try again. [lzma, lzw]")


def df_to_bytes(df: pd.DataFrame) -> bytes:
    return pickle.dumps(df)


def bytes_to_df(obj):
    return pickle.load(obj)


def main():
    if len(sys.argv) != 2:
        print("Please include a path to desired YAML file as follows: python3 compression.py <path_to_yaml>")
        sys.exit(1)

    runtime_parameters = config_get('runtime_parameters')

    columns = [
        'compression_method', 'columns_decayed', 'size (bytes)', 'original_size (bytes)',
        'percentage_of_original_size', 'time_elapsed (ms)', 'compression_time (ms)', 'decompression_time (ms)'
    ]
    output_summary = pd.DataFrame(columns=columns)
    data = pd.read_csv(config_get('database'), sep=',')

    for choice, parameters in runtime_parameters.items():
        compression_params = parameters.get('compression_parameters')
        method = compression_params.get('method')
        compressed_columns = compression_params.get('compressed_columns')
        encoding = compression_params.get('encoding')

        results, total_time = function_execution_in_milliseconds(compression_wrapper, data, method, compressed_columns)
        compression_time = results[0]
        decompression_time = results[1]
        original_size = results[2]
        compressed_size = results[3]
        decompressed_size = results[4]

        new_row = {
            'compression_method': method,
            'columns_decayed': len(compressed_columns),
            'size (bytes)': compressed_size,
            'original_size (bytes)': original_size,
            'percentage_of_original_size': str(round(compressed_size / original_size * 100, 4)) + '%',
            'time_elapsed (ms)': total_time,
            'compression_time (ms)': compression_time,
            'decompression_time (ms)': decompression_time,
        }
        output_summary.loc[len(output_summary)] = new_row

    print(output_summary)


if __name__ == "__main__":
    main()
