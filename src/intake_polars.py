import datetime
import re
from os.path import basename
from cloudpathlib import AnyPath

import fsspec
import pandas as pd
import polars as pl
from intake.source.csv import CSVSource
from intake_parquet import ParquetSource
from polars.datatypes.convert import DataTypeMappings
from enum import Enum
from contextvars import ContextVar
from contextlib import contextmanager
import os
import warnings
from pyarrow.parquet import ParquetFile
from pyarrow.lib import ArrowInvalid
from s3fs import S3File

FFI_NAME_TO_DTYPE = {v: k for k, v in DataTypeMappings.DTYPE_TO_FFINAME.items()}


PANDAS_TO_POLARS_TYPES = {
    "str": pl.String,
    "float32": pl.Float32,
    "float64": pl.Float64,
    "int64": pl.Int64,
    "int32": pl.Int32,
}


def parse_pandas_schema(schema: dict[str, str]) -> dict:
    try:
        return {k: PANDAS_TO_POLARS_TYPES[v] for k, v in schema.items()}
    except KeyError as e:
        raise ValueError(
            f"{e.args[0]} is an invalid pandas type or hasn't been mapped to a polars type in intake_polars yet"
        )


def simplify_glob(s: str):
    return re.sub("\*\*+", "*", s)


def date_format_to_regex(date_format: str):
    """Convert a date(time) format to a regex pattern

    :param date_format: Date format string
    :type date_format: str
    :return: Regex pattern
    :rtype: str

    :example
    >>> date_format_to_regex("%Y-%m-%d")
    """
    format_map = {
        "-": "\\-",
        "/": "\\/",
        "%Y": "\d{4}",
        "%m": "\d{2}",
        "%d": "\d{2}",
        "%H": "\d{2}",
        "%M": "\d{2}",
        "%S": "\d{2}",
        "%f": "\d{6}",
    }
    for k, v in format_map.items():
        date_format = date_format.replace(k, v)

    return date_format


def _extract_ymd(
    x: str | list[str],
    date_format: str = "%Y-%m-%d",
):
    if isinstance(x, str):
        x = [x]

    regex = re.compile(date_format_to_regex(date_format))
    dates_str = [re.findall(regex, y) for y in x]

    return [
        datetime.datetime.strptime(y[-1], date_format).date() if len(y) > 0 else None
        for y in dates_str
    ]


def _convert_ffi_to_dtype(ffi: str):
    try:
        return FFI_NAME_TO_DTYPE[ffi]
    except KeyError:
        raise ValueError(
            f'Invalid polars type name: {ffi}. Valid types:{",".join(FFI_NAME_TO_DTYPE.keys())}'
        )


storage_location_override_var = ContextVar("storage_location_overide_var")


@contextmanager
def storage_location_override(location: str | None):
    match location:
        case "local":
            old_value = storage_location_override_var.set(StorageLocation.LOCAL)
        case "cloud":
            old_value = storage_location_override_var.set(StorageLocation.CLOUD)
        case "default" | None:
            old_value = storage_location_override_var.set(StorageLocation.DEFAULT)
        case _:
            raise NotImplementedError(
                f'Unknown storage location {location}. Please choose from {",".join(SUPPORTED_STORAGE_PREFIXES+["default","None"])}'
            )
    yield
    storage_location_override_var.reset(old_value)


class StorageLocation(Enum):
    DEFAULT = 0
    LOCAL = 1
    CLOUD = 2


SUPPORTED_STORAGE_PREFIXES = ["local", "cloud"]


class StorageLocationMixin:
    """
    Figures out the best location for the data.
    Location preference can be set via envirointment
    Location can be overriden with storage_location_override context.
    """

    def __init__(
        self,
        *args,
        storage_prefix: dict[str, str] | None = None,
        storage_default: str = "local",
        **kwargs,
    ):
        if storage_prefix is None:
            storage_prefix = {}
        self._storage_prefix = storage_prefix
        for prefix_type in SUPPORTED_STORAGE_PREFIXES:
            if prefix_type not in storage_prefix:
                storage_prefix[prefix_type] = ""
        self._storage_default = storage_default
        assert self._storage_default in SUPPORTED_STORAGE_PREFIXES and all(
            [
                prefix in SUPPORTED_STORAGE_PREFIXES
                for prefix in self._storage_prefix.keys()
            ]
        )
        super().__init__(*args, **kwargs)

    @property
    def _storage_location_preference(self):
        try:
            env_pref = os.environ["ONEPIPELINE_STORAGE_LOCATION"]
            match env_pref:
                case "default":
                    return StorageLocation.DEFAULT
                case "local":
                    return StorageLocation.LOCAL
                case "cloud":
                    return StorageLocation.CLOUD
                case _:
                    raise NotImplementedError(
                        f'Unknown storage location {env_pref}. Please choose from {["default"]+SUPPORTED_STORAGE_PREFIXES}'
                    )

        except KeyError:
            return StorageLocation.DEFAULT

    @property
    def file_path_pattern(self) -> str:
        return self._build_urlpath(self._entry._captured_init_kwargs["args"]["urlpath"])

    def _build_urlpath(self, suffix):
        match self._cloud_or_local():
            case StorageLocation.LOCAL:
                prefix = self._storage_prefix["local"]
            case StorageLocation.CLOUD:
                prefix = self._storage_prefix["cloud"]
            case StorageLocation.DEFAULT:
                prefix = self._storage_prefix[self._storage_default]

        if prefix == "":
            return suffix
        return str(AnyPath(prefix) / simplify_glob(suffix))

    @property
    def _urlpath(self) -> str:
        return self._build_urlpath(self._raw_url_path)

    @property
    def parent_dir(self) -> AnyPath:
        return AnyPath(self._urlpath).parent

    def make_parent_dir(self, parents=True, exist_ok: bool = True):
        return self.parent_dir.mkdir(exist_ok=exist_ok, parents=parents)

    @_urlpath.setter
    def _urlpath(self, value: str):
        if value.startswith("s3") or value.startswith("/"):
            warnings.warn(
                "Your catalog is not storage location agnostic and will be removed in a future version",
                DeprecationWarning,
            )
        self._raw_url_path = value

    def _cloud_or_local(self) -> StorageLocation:
        try:
            return storage_location_override_var.get()
        except LookupError:
            return self._storage_location_preference

    def sync(self):
        prefixes = list(self._storage_prefix.keys())
        file_lists = []
        for prefix in prefixes:
            with storage_location_override(prefix):
                file_lists.append(
                    self.list_file_details_pl()
                    .select(["last_modified", "date", "link"])
                    .with_columns(location=pl.lit(prefix))
                )
        file_list = pl.concat(file_lists)

        # figure out which days have missing files and mark them outdated by setting mod time to 1990
        dates_with_missing_files = (
            file_list.filter(pl.len().over(["date"]).lt(len(prefixes)))["date"]
            .unique()
            .to_list()
        )
        missing_df_dates = []
        missing_df_link = []
        missing_df_location = []
        for date in dates_with_missing_files:
            for location in prefixes:
                with storage_location_override(location):
                    missing_df_dates.append(date)
                    missing_df_location.append(location)
                    missing_df_link.append(self(date=str(date))._urlpath)
        expected_files = pl.DataFrame(
            {
                "location": missing_df_location,
                "link": missing_df_link,
                "date": missing_df_dates,
            },
            schema={"location": pl.String, "link": pl.String, "date": pl.Date},
        )
        missing_files = expected_files.join(
            file_list.unique(["link"]), how="anti", on=["link"]
        ).with_columns(last_modified=pl.lit(datetime.datetime(1990, 1, 1)))
        file_list = pl.concat([file_list.with_columns(pl.col('last_modified').dt.replace_time_zone(None)), missing_files],how='diagonal')

        files_to_sync = (
            file_list.unique(["link"])
            .filter(pl.n_unique(["last_modified"]).over(["date"]).gt(1))
            .sort("last_modified", descending=True)
        )

        for df in files_to_sync.partition_by(["date"]):
            d=df.to_dict()
            up_to_date_file = d['link'][0]
            up_to_date_file_time = d['last_modified'][0]
            source_location=d['location'][0]
            with storage_location_override(source_location):
                source_fs=self.get_fs()
            for file, storage_location, mod_time in zip(
                d['link'][1:], d["location"][1:], d['last_modified'][1:]
            ):
                # consider files that are 3 seconds appart up to date
                if (up_to_date_file_time - mod_time).total_seconds() > 3:
                    with storage_location_override(storage_location):
                        copy_between_fs(source_fs,self.get_fs(),up_to_date_file,file)

def copy_between_fs(source_fs,destination_fs,source_file,target_file,timestamp):
    "Assumes that in case of two s3fs they are the same."
    "Sets the timestamp"
    import s3fs
    import fsspec.implementations.local
    breakpoint()
    match type(source_fs):

        case fsspec.implementations.local.LocalFileSystem:
            match type(destination_fs):
                case s3fs.core.S3FileSystem:
                    destination_fs.put(source_file,target_file)
                    S3File(destination_fs,target_file).setxattr({'last_modified':'1'})
                case fsspec.implementations.local.LocalFileSystem:
                    source_fs.cp(source_file,target_file)
                case _:
                    raise NotImplementedError()

        case s3fs.core.S3FileSystem:
            match type(destination_fs):
                case s3fs.core.S3FileSystem:
                    destination_fs.cp(source_file,target_file)
                    S3File(destination_fs,target_file).setxattr({'mattr':'1'})
                case fsspec.implementations.local.LocalFileSystem:
                    source_fs.get(source_file,target_file)
                case _:
                    raise NotImplementedError()


        case _:
            raise NotImplementedError()

    def read(self):
        if self._urlpath.startswith('/'):
            storage_args=self._storage_options
            self._storage_options={}
            res=super().read()
            self._storage_options=storage_args
            return res
        else:
            return super().read()




def _get_fs(path, storage_options=None):
    fs, _, _ = fsspec.core.get_fs_token_paths(
        simplify_glob(path), storage_options=storage_options
    )
    return fs


def _list_files(fs, path) -> list[str]:
    file_list = fs.glob(simplify_glob(path))
    if (fs.protocol == "file") or (fs.protocol == ("file", "local")):
        return file_list
    return list(map(fs.unstrip_protocol, file_list))


class FSAbstractionMixin:
    _urlpath: str
    _storage_options: dict | None

    def get_fs(self):
        return _get_fs(self._urlpath, self._storage_options)

    def list_files(self, fs=None):
        if fs is None:
            fs = self.get_fs()
        return _list_files(fs, self._urlpath)

    def find_latest(self):
        try:
            return self(urlpath=list(sorted(self.list_files()))[-1])
        except IndexError:
            raise FileNotFoundError(
                f"Counldn't find any files matching {self._urlpath}"
            )

    def list_file_details(
        self,
        extract_date: bool = True,
        date_format: str = "%Y-%m-%d",
    ) -> pd.DataFrame:
        links = self.list_files()
        fs = self.get_fs()
        filenames = [basename(fp) for fp in links]
        sizes = [fs.size(fp) for fp in links]
        last_modified = [fs.modified(fp) for fp in links]

        file_urls = pd.DataFrame(
            {
                "filename": pd.Series(filenames, dtype="string"),
                "link": pd.Series(links, dtype="string"),
                "size": pd.Series(sizes, dtype="int64"),
                "last_modified": pd.Series(last_modified, dtype="datetime64[ns]"),
            }
        )

        # Convert timestamp to Chicago time
        if len(file_urls) > 0:
            if file_urls["last_modified"].dt.tz is None:
                file_urls["last_modified"] = file_urls["last_modified"].dt.tz_localize(
                    "UTC"
                )

            file_urls["last_modified"] = file_urls["last_modified"].dt.tz_convert(
                tz="America/Chicago"
            )

        # Extract date from path
        if extract_date:
            file_urls["date"] = pd.Series(
                _extract_ymd(file_urls["link"], date_format=date_format),
                dtype="datetime64[ns]",
            ).dt.date

        return file_urls

    def list_file_details_pl(
        self,
        extract_date: bool = True,
        date_format: str = "%Y-%m-%d",
    ) -> pl.DataFrame:
        links = self.list_files()
        fs = self.get_fs()

        return _get_list_file_details(
            links=links, fs=fs, date_format=date_format, extract_date=extract_date
        )

    def _get_storage_options(self):
        if self._storage_options and "profile" in self._storage_options:
            from boto3 import Session

            session = Session(profile_name=self._storage_options["profile"])
            credentials = session.get_credentials()
            storage_options = {
                "aws_access_key_id": credentials.access_key,
                "aws_secret_access_key": credentials.secret_key,
            }

        else:
            storage_options = self._storage_options

        return storage_options


def _parse_hive_types(hive_schema_dict: dict[str, str]):
    new_dict = {}
    for k, v in hive_schema_dict.items():
        try:
            new_dict[k] = DataTypeMappings.REPR_TO_DTYPE[v]
        except KeyError:
            raise ValueError(
                f'{v} is not a valid type name. Valid type names are {",".join(list(DataTypeMappings.REPR_TO_DTYPE.keys()))}'
            )

    return new_dict


class PolarsParquet(StorageLocationMixin, ParquetSource, FSAbstractionMixin):
    container = "dataframe"
    name = "polars"
    version = "0.0.2"
    partition_access = True

    def __init__(self, *args, polars_kwargs=None, **kwargs):
        super().__init__(*args, **kwargs)
        if polars_kwargs is None:
            self.polars_kwargs = {}
        else:
            self.polars_kwargs = polars_kwargs
            if "hive_schema" in self.polars_kwargs:
                self.polars_kwargs["hive_schema"] = _parse_hive_types(
                    self.polars_kwargs["hive_schema"]
                )

    def scan_for_corrupt_files(self):
        fs = self.get_fs()

        bad_files = []
        for file in self.list_files():
            try:
                ParquetFile(file, filesystem=fs)
            except ArrowInvalid:
                bad_files.append(file)
        return bad_files

    def find_differing_schemas(self, target_schema=None):
        files = self.list_files()

        if target_schema is None:
            target_schema = self._scan_single_file(files[0]).collect_schema()

        return list(
            filter(
                lambda file: self._scan_single_file(file).collect_schema()
                != target_schema,
                files,
            )
        )

    def read(self):
        if self._urlpath.startswith('/'):
            storage_args=self._storage_options
            self._storage_options={}
            res=super().read()
            self._storage_options=storage_args
            return res
        else:
            return super().read()

    def to_lazy(self) -> pl.LazyFrame:
        storage_options = self._get_storage_options()

        pl_ds = pl.scan_parquet(
            self._urlpath, storage_options=storage_options, **self.polars_kwargs
        )

        if self._kwargs.get("columns"):
            pl_ds.select(self._kwargs.get("columns"))

        return pl_ds

    def read_pl(self) -> pl.DataFrame:
        return self.to_lazy().collect()

    def _scan_single_file(self, file):
        return pl.scan_parquet(
            file, storage_options=self._get_storage_options(), **self.polars_kwargs
        )

    def to_lazy_list(self) -> list[pl.LazyFrame]:
        return list(map(self._scan_single_file, self.list_files()))

    def write_df(
        self, df: pl.LazyFrame | pl.DataFrame, *args, use_pyarrow=True, **kwargs
    ):
        if "*" in self._urlpath:
            raise ValueError("Catalog item needs to be fully specified")
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        return df.write_parquet(self._urlpath, *args, use_pyarrow=use_pyarrow, **kwargs)


def _parse_dtypes(dtype_dict):
    return {k: _convert_ffi_to_dtype(v) for k, v in dtype_dict.items()}


class PolarsCSV(StorageLocationMixin, CSVSource, FSAbstractionMixin):
    container = "dataframe"
    name = "polars_csv"
    version = "0.0.2"
    partition_access = True

    def __init__(self, *args, polars_kwargs: dict | None = None, **kwargs):
        self._polars_kwargs = polars_kwargs
        super().__init__(*args, **kwargs)

    def read_pl(self) -> pl.DataFrame:
        return self.to_lazy().collect()

    def to_lazy(self) -> pl.LazyFrame:
        if self._polars_kwargs is None:
            kwargs = {}
        else:
            kwargs = {
                k: _parse_dtypes(v) if k in ["schema", "dtypes", "hive_schema"] else v
                for k, v in self._polars_kwargs.items()
            }

        files = self.list_files()

        try:
            kwargs["separator"] = self._csv_kwargs["sep"]
        except KeyError:
            pass

        try:
            kwargs["null_values"] = self._csv_kwargs["na_values"]
        except KeyError:
            pass

        try:
            kwargs["schema_overrides"] = parse_pandas_schema(self._csv_kwargs["dtype"])
        except KeyError:
            pass

        if len(files) and files[0].endswith(".zip"):
            if len(files) > 1:
                warnings.warn("Reading zipped csvs. Switching to eager mode.")
            df_l = []
            fs = self.get_fs()

            for file in files:
                with fs.open(file, compression="zip", mode="rb") as fh:
                    df_l.append(pl.read_csv(fh, **kwargs))
            pl_ds = pl.concat(df_l).lazy()
        else:
            pl_ds = pl.scan_csv(
                files, storage_options=self._get_storage_options(), **kwargs
            )
        if self._csv_kwargs.get("columns"):
            pl_ds.select(self._csv_kwargs.get("columns"))

        return pl_ds


def _get_list_file_details(links, fs, date_format, extract_date) -> pl.DataFrame:
    filenames = [basename(fp) for fp in links]
    sizes = [fs.size(fp) for fp in links]
    last_modified = [fs.modified(fp) for fp in links]

    file_urls = pl.DataFrame(
        {
            "filename": pl.Series(filenames, dtype=pl.String),
            "link": pl.Series(links, dtype=pl.String),
            "size": pl.Series(sizes, dtype=pl.Int64),
            "last_modified": pl.Series(last_modified, dtype=pl.Datetime),
        }
    )

    # Convert timestamp to Chicago time
    if file_urls["last_modified"].dtype.time_zone is None:
        file_urls = file_urls.with_columns(
            pl.col("last_modified").dt.replace_time_zone("UTC")
        )
    file_urls = file_urls.with_columns(
        pl.col("last_modified").dt.convert_time_zone(time_zone="America/Chicago")
    )

    # Extract date from path
    if extract_date:
        file_urls = file_urls.with_columns(
            pl.col("link")
            .str.extract("(" + date_format_to_regex(date_format) + ")")
            .str.to_date()
            .alias("date")
        )

    return file_urls

