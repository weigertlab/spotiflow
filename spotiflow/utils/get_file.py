"""
Adapted from tensorflow.keras.utils.data_utils
"""

import hashlib
import os
import pathlib
import shutil
import tarfile
import time
import sys
import urllib
import warnings
from six.moves.urllib.parse import urlsplit
import numpy as np
import zipfile
from urllib.request import urlretrieve


def _resolve_path(path):
    return os.path.realpath(os.path.abspath(path))


def _is_path_in_dir(path, base_dir):
    return _resolve_path(os.path.join(base_dir, path)).startswith(base_dir)


def _is_link_in_dir(info, base):
    tip = _resolve_path(os.path.join(base, os.path.dirname(info.name)))
    return _is_path_in_dir(info.linkname, base_dir=tip)


def _filter_safe_paths(members):
    base_dir = _resolve_path(".")
    for finfo in members:
        valid_path = False
        if _is_path_in_dir(finfo.name, base_dir):
            valid_path = True
            yield finfo
        elif finfo.issym() or finfo.islnk():
            if _is_link_in_dir(finfo, base_dir):
                valid_path = True
                yield finfo
        if not valid_path:
            warnings.warn(
                "Skipping invalid path during archive extraction: " f"'{finfo.name}'."
            )


def _makedirs_exist_ok(datadir):
    os.makedirs(datadir, exist_ok=True)


def _resolve_hasher(algorithm, file_hash=None):
    """Returns hash algorithm as hashlib function."""
    if algorithm == "sha256":
        return hashlib.sha256()

    if algorithm == "auto" and file_hash is not None and len(file_hash) == 64:
        return hashlib.sha256()

    # This is used only for legacy purposes.
    return hashlib.md5()


def _hash_file(fpath, algorithm="sha256", chunk_size=65535):
    """Calculates a file sha256 or md5 hash.

    Example:

    ```python
    _hash_file('/path/to/file.zip')
    'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```

    Args:
        fpath: Path to the file being validated.
        algorithm: Hash algorithm, one of `'auto'`, `'sha256'`, or `'md5'`.
            The default `'auto'` detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        The file hash.
    """
    if isinstance(algorithm, str):
        hasher = _resolve_hasher(algorithm)
    else:
        hasher = algorithm

    with open(fpath, "rb") as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def validate_file(fpath, file_hash, algorithm="auto", chunk_size=65535):
    """Validates a file against a sha256 or md5 hash.

    Args:
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.

    Returns:
        Whether the file is valid
    """
    hasher = _resolve_hasher(algorithm, file_hash)

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _extract_archive(file_path, path=".", archive_format="auto"):
    """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

    Args:
        file_path: Path to the archive file.
        path: Where to extract the archive file.
        archive_format: Archive format to try for extracting the file.
            Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
            `'tar'` includes tar, tar.gz, and tar.bz files.
            The default 'auto' is `['tar', 'zip']`.
            `None` or an empty list will return no matches found.

    Returns:
        True if a match was found and an archive extraction was completed,
        False otherwise.
    """
    if archive_format is None:
        return False
    if archive_format == "auto":
        archive_format = ["tar", "zip"]
    if isinstance(archive_format, str):
        archive_format = [archive_format]

    file_path = str(file_path)
    path = str(path)

    for archive_type in archive_format:
        if archive_type == "tar":
            open_fn = tarfile.open
            is_match_fn = tarfile.is_tarfile
        if archive_type == "zip":
            open_fn = zipfile.ZipFile
            is_match_fn = zipfile.is_zipfile

        if is_match_fn(file_path):
            with open_fn(file_path) as archive:
                try:
                    if zipfile.is_zipfile(file_path):
                        # Zip archive.
                        archive.extractall(path)
                    else:
                        # Tar archive, perhaps unsafe. Filter paths.
                        archive.extractall(path, members=_filter_safe_paths(archive))
                except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
                    if os.path.exists(path):
                        if os.path.isfile(path):
                            os.remove(path)
                        else:
                            shutil.rmtree(path)
                    raise
            return True
    return False


class Progbar:
    """Displays a progress bar.

    Args:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that should *not*
          be averaged over time. Metrics in this list will be displayed as-is.
          All others will be averaged by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
        unit_name: Display name for step counts (usually "step" or "sample").
    """

    def __init__(
        self,
        target,
        width=30,
        verbose=1,
        interval=0.05,
        stateful_metrics=None,
        unit_name="step",
    ):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        self.unit_name = unit_name
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = (
            (hasattr(sys.stdout, "isatty") and sys.stdout.isatty())
            or "ipykernel" in sys.modules
            or "posix" in sys.modules
            or "PYCHARM_HOSTED" in os.environ
        )
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0
        self._time_at_epoch_start = self._start
        self._time_at_epoch_end = None
        self._time_after_first_step = None

    def update(self, current, values=None, finalize=None):
        """Updates the progress bar.

        Args:
            current: Index of current step.
            values: List of tuples: `(name, value_for_last_step)`. If `name` is
              in `stateful_metrics`, `value_for_last_step` will be displayed
              as-is. Else, an average of the metric over time will be
              displayed.
            finalize: Whether this is the last update for the progress bar. If
              `None`, uses `current >= self.target`. Defaults to `None`.
        """
        if finalize is None:
            if self.target is None:
                finalize = False
            else:
                finalize = current >= self.target

        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                # In the case that progress bar doesn't have a target value in
                # the first epoch, both on_batch_end and on_epoch_end will be
                # called, which will cause 'current' and 'self._seen_so_far' to
                # have the same value. Force the minimal value to 1 here,
                # otherwise stateful_metric will be 0s.
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        message = ""
        now = time.time()
        info = f" - {now - self._start:.0f}s"
        if current == self.target:
            self._time_at_epoch_end = now
        if self.verbose == 1:
            if now - self._last_update < self.interval and not finalize:
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                message += "\b" * prev_total_width
                message += "\r"
            else:
                message += "\n"

            if self.target is not None:
                numdigits = int(np.log10(self.target)) + 1
                bar = ("%" + str(numdigits) + "d/%d [") % (current, self.target)
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += "=" * (prog_width - 1)
                    if current < self.target:
                        bar += ">"
                    else:
                        bar += "="
                bar += "." * (self.width - prog_width)
                bar += "]"
            else:
                bar = "%7d/Unknown" % current

            self._total_width = len(bar)
            message += bar

            time_per_unit = self._estimate_step_duration(current, now)

            if self.target is None or finalize:
                info += self._format_time(time_per_unit, self.unit_name)
            else:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = "%d:%02d:%02d" % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = "%d:%02d" % (eta // 60, eta % 60)
                else:
                    eta_format = "%ds" % eta

                info = f" - ETA: {eta_format}"

            for k in self._values_order:
                info += f" - {k}:"
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += f" {avg:.4f}"
                    else:
                        info += f" {avg:.4e}"
                else:
                    info += f" {self._values[k]}"

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += " " * (prev_total_width - self._total_width)

            if finalize:
                info += "\n"

            message += info
            print(message, end="")
            message = ""

        elif self.verbose == 2:
            if finalize:
                numdigits = int(np.log10(self.target)) + 1
                count = ("%" + str(numdigits) + "d/%d") % (current, self.target)
                info = count + info
                for k in self._values_order:
                    info += f" - {k}:"
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += f" {avg:.4f}"
                    else:
                        info += f" {avg:.4e}"
                if self._time_at_epoch_end:
                    time_per_epoch = self._time_at_epoch_end - self._time_at_epoch_start
                    avg_time_per_step = time_per_epoch / self.target
                    self._time_at_epoch_start = now
                    self._time_at_epoch_end = None
                    info += " -" + self._format_time(time_per_epoch, "epoch")
                    info += " -" + self._format_time(avg_time_per_step, self.unit_name)
                    info += "\n"
                message += info
                print(message)
                message = ""

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)

    def _format_time(self, time_per_unit, unit_name):
        """format a given duration to display to the user.

        Given the duration, this function formats it in either milliseconds
        or seconds and displays the unit (i.e. ms/step or s/epoch)
        Args:
          time_per_unit: the duration to display
          unit_name: the name of the unit to display
        Returns:
          a string with the correctly formatted duration and units
        """
        formatted = ""
        if time_per_unit >= 1 or time_per_unit == 0:
            formatted += f" {time_per_unit:.0f}s/{unit_name}"
        elif time_per_unit >= 1e-3:
            formatted += f" {time_per_unit * 1000.0:.0f}ms/{unit_name}"
        else:
            formatted += f" {time_per_unit * 1000000.0:.0f}us/{unit_name}"
        return formatted

    def _estimate_step_duration(self, current, now):
        """Estimate the duration of a single step.

        Given the step number `current` and the corresponding time `now` this
        function returns an estimate for how long a single step takes. If this
        is called before one step has been completed (i.e. `current == 0`) then
        zero is given as an estimate. The duration estimate ignores the duration
        of the (assumed to be non-representative) first step for estimates when
        more steps are available (i.e. `current>1`).

        Args:
          current: Index of current step.
          now: The current time.

        Returns: Estimate of the duration of a single step.
        """
        if current:
            # there are a few special scenarios here:
            # 1) somebody is calling the progress bar without ever supplying
            #    step 1
            # 2) somebody is calling the progress bar and supplies step one
            #    multiple times, e.g. as part of a finalizing call
            # in these cases, we just fall back to the simple calculation
            if self._time_after_first_step is not None and current > 1:
                time_per_unit = (now - self._time_after_first_step) / (current - 1)
            else:
                time_per_unit = (now - self._start) / current

            if current == 1:
                self._time_after_first_step = now
            return time_per_unit
        else:
            return 0

    def _update_stateful_metrics(self, stateful_metrics):
        self.stateful_metrics = self.stateful_metrics.union(stateful_metrics)


def get_file(
    fname=None,
    origin=None,
    untar=False,
    md5_hash=None,
    file_hash=None,
    cache_subdir="datasets",
    hash_algorithm="auto",
    extract=False,
    archive_format="auto",
    cache_dir=None,
):
    """Downloads a file from a URL if it not already in the cache.

    By default the file at the url `origin` is downloaded to the
    cache_dir `~/.spotiflow`, placed in the cache_subdir `datasets`,
    and given the filename `fname`. The final location of a file
    `example.txt` would therefore be `~/.spotiflow/datasets/example.txt`.

    Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
    Passing a hash will verify the file after download. The command line
    programs `shasum` and `sha256sum` can compute the hash.
    ```

    Args:
        fname: Name of the file. If an absolute path `/path/to/file.txt` is
            specified the file will be saved at that location. If `None`, the
            name of the file at `origin` will be used.
        origin: Original URL of the file.
        untar: Deprecated in favor of `extract` argument.
            boolean, whether the file should be decompressed
        md5_hash: Deprecated in favor of `file_hash` argument.
            md5 hash of the file for verification
        file_hash: The expected hash string of the file after download.
            The sha256 and md5 hash algorithms are both supported.
        cache_subdir: Subdirectory under the Keras cache dir where the file is
            saved. If an absolute path `/path/to/folder` is
            specified the file will be saved at that location.
        hash_algorithm: Select the hash algorithm to verify the file.
            options are `'md5'`, `'sha256'`, and `'auto'`.
            The default 'auto' detects the hash algorithm in use.
        extract: True tries extracting the file as an Archive, like tar or zip.
        archive_format: Archive format to try for extracting the file.
            Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
            `'tar'` includes tar, tar.gz, and tar.bz files.
            The default `'auto'` corresponds to `['tar', 'zip']`.
            None or an empty list will return no matches found.
        cache_dir: Location to store cached files, when None it
            defaults to `~/.spotiflow/`.

    Returns:
        Path to the downloaded file.

    ⚠️ **Warning on malicious downloads** ⚠️

    Downloading something from the Internet carries a risk.
    NEVER download a file/archive if you do not trust the source.
    We recommend that you specify the `file_hash` argument
    (if the hash of the source file is known) to make sure that the file you
    are getting is the one you expect.
    """
    if origin is None:
        raise ValueError(
            'Please specify the "origin" argument (URL of the file ' "to download)."
        )

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".spotiflow")
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = "md5"
    datadir_base = os.path.expanduser(cache_dir)
    datadir = os.path.join(datadir_base, cache_subdir)
    _makedirs_exist_ok(datadir)
    fname = str(fname)
    if not fname:
        fname = os.path.basename(urlsplit(origin).path)
        if not fname:
            raise ValueError(
                "Can't parse the file name from the origin provided: "
                f"'{origin}'."
                "Please specify the `fname` as the input param."
            )

    if untar:
        if fname.endswith(".tar.gz"):
            fname = pathlib.Path(fname)
            # The 2 `.with_suffix()` are because of `.tar.gz` as pathlib
            # considers it as 2 suffixes.
            fname = fname.with_suffix("").with_suffix("")
            fname = str(fname)
        untar_fpath = os.path.join(datadir, fname)
        fpath = untar_fpath + ".tar.gz"
    else:
        fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print(
                    "A local file was found, but it seems to be "
                    f"incomplete or outdated because the {hash_algorithm} "
                    "file hash does not match the original value of "
                    f"{file_hash} "
                    "so we will re-download the data."
                )
                download = True
    else:
        download = True

    if download:
        print(f"Downloading data from {origin} to {fpath}")

        class DLProgbar:
            """Manage progress bar state for use in urlretrieve."""

            def __init__(self):
                self.progbar = None
                self.finished = False

            def __call__(self, block_num, block_size, total_size):
                if not self.progbar:
                    if total_size == -1:
                        total_size = None
                    self.progbar = Progbar(total_size)
                current = block_num * block_size

                if total_size is None:
                    self.progbar.update(current)
                else:
                    if current < total_size:
                        self.progbar.update(current)
                    elif not self.finished:
                        self.progbar.update(self.progbar.target)
                        self.finished = True

        error_msg = "URL fetch failure on {}: {} -- {}"
        try:
            try:
                urlretrieve(origin, fpath, DLProgbar())
            except urllib.error.HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except urllib.error.URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        # Validate download if succeeded and user provided an expected hash
        # Security conscious users would get the hash of the file from a
        # separate channel and pass it to this API to prevent MITM / corruption:
        if os.path.exists(fpath) and file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                raise ValueError(
                    "Incomplete or corrupted file detected. "
                    f"The {hash_algorithm} "
                    "file hash does not match the provided value "
                    f"of {file_hash}."
                )

    if untar:
        if not os.path.exists(untar_fpath):
            _extract_archive(fpath, datadir, archive_format="tar")
        return untar_fpath

    if extract:
        _extract_archive(fpath, datadir, archive_format)

    return fpath


# ----------
