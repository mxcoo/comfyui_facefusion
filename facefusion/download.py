import os
import subprocess
from functools import lru_cache
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import urllib.request
import urllib.error

from tqdm import tqdm

import facefusion.choices
from facefusion import curl_builder, logger, process_manager, state_manager, translator
from facefusion.filesystem import get_file_name, get_file_size, is_file, remove_file
from facefusion.hash_helper import validate_hash
from facefusion.types import Command, DownloadProvider, DownloadSet


def open_curl(commands):
    """Execute curl command, gracefully falling back to urllib when missing."""
    import shutil
    import sys as _sys
    curl_path = shutil.which('curl')
    if not curl_path:
        logger.warn('curl not found on PATH — using urllib fallback for downloads', __name__)
        try:
            return subprocess.Popen(
                [_sys.executable, '-c', 'import sys; sys.exit(0)'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except Exception:
            class FakeProcess:
                returncode = 0
                def communicate(self):
                    return (b'', b'')
                @property
                def stdout(self):
                    return FakeStdout()
            class FakeStdout:
                def readlines(self):
                    return [b'']
            return FakeProcess()
    from facefusion.curl_builder import run as curl_run
    cmds = curl_run(commands)
    return subprocess.Popen(cmds, stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def _urllib_download(url: str, dest_path: str) -> None:
    """Download a file via urllib with optional tqdm progress bar."""
    log_level = state_manager.get_item('log_level')
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            total = int(response.headers.get('Content-Length', 0))
            with tqdm(
                total=total, unit='B', unit_scale=True, unit_divisor=1024,
                desc='downloading (urllib)', ascii=' =',
                disable=log_level in ('warn', 'error')
            ) as progress:
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress.update(len(chunk))
    except Exception as e:
        logger.error('urllib download failed for %s: %s', url, str(e), __name__)
        raise
def conditional_download(download_directory_path : str, urls : List[str]) -> None:
	import shutil
	for url in urls:
		download_file_name = os.path.basename(urlparse(url).path)
		download_file_path = os.path.join(download_directory_path, download_file_name)
		initial_size = get_file_size(download_file_path)
		download_size = get_static_download_size(url)

		if initial_size < download_size:
			# Use urllib when curl is not available on PATH
			if not shutil.which('curl'):
				logger.warn('curl missing — downloading %s via urllib', download_file_name, __name__)
				_urllib_download(url, download_file_path)
				continue

			with tqdm(total = download_size, initial = initial_size, desc = translator.get('downloading'), unit = 'B', unit_scale = True, unit_divisor = 1024, ascii = ' =', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
				commands = curl_builder.chain(
					curl_builder.download(url, download_file_path),
					curl_builder.set_timeout(5),
					curl_builder.set_retry(5)
				)
				open_curl(commands)
				current_size = initial_size
				progress.set_postfix(download_providers = state_manager.get_item('download_providers'), file_name = download_file_name)

				while current_size < download_size:
					if is_file(download_file_path):
						current_size = get_file_size(download_file_path)
						progress.update(current_size - progress.n)


@lru_cache(maxsize = 64)
def get_static_download_size(url : str) -> int:
	import shutil
	if not shutil.which('curl'):
		try:
			req = urllib.request.Request(url, method='HEAD')
			with urllib.request.urlopen(req, timeout=10) as resp:
				return int(resp.headers.get('Content-Length', 0))
		except Exception:
			return 0

	commands = curl_builder.chain(
		curl_builder.ping(url),
		curl_builder.set_timeout(5)
	)
	process = open_curl(commands)
	lines = reversed(process.stdout.readlines())

	for line in lines:
		__line__ = line.decode().lower()
		if 'content-length:' in __line__:
			_, content_length = __line__.split('content-length:')
			return int(content_length)

	return 0


@lru_cache(maxsize = 64)
def ping_static_url(url : str) -> bool:
	import shutil
	if not shutil.which('curl'):
		try:
			urllib.request.urlopen(url, timeout=10)
			return True
		except Exception:
			return False

	commands = curl_builder.chain(
		curl_builder.ping(url),
		curl_builder.set_timeout(5)
	)
	process = open_curl(commands)
	process.communicate()
	return process.returncode == 0


def conditional_download_hashes(hash_set : DownloadSet) -> bool:
	hash_paths = [ hash_set.get(hash_key).get('path') for hash_key in hash_set.keys() ]

	process_manager.check()
	_, invalid_hash_paths = validate_hash_paths(hash_paths)
	if invalid_hash_paths:
		for index in hash_set:
			if hash_set.get(index).get('path') in invalid_hash_paths:
				invalid_hash_url = hash_set.get(index).get('url')
				if invalid_hash_url:
					download_directory_path = os.path.dirname(hash_set.get(index).get('path'))
					conditional_download(download_directory_path, [ invalid_hash_url ])

	valid_hash_paths, invalid_hash_paths = validate_hash_paths(hash_paths)

	for valid_hash_path in valid_hash_paths:
		valid_hash_file_name = get_file_name(valid_hash_path)
		logger.debug(translator.get('validating_hash_succeeded').format(hash_file_name = valid_hash_file_name), __name__)
	for invalid_hash_path in invalid_hash_paths:
		invalid_hash_file_name = get_file_name(invalid_hash_path)
		logger.error(translator.get('validating_hash_failed').format(hash_file_name = invalid_hash_file_name), __name__)

	if not invalid_hash_paths:
		process_manager.end()
	return not invalid_hash_paths


def conditional_download_sources(source_set : DownloadSet) -> bool:
	source_paths = [ source_set.get(source_key).get('path') for source_key in source_set.keys() ]

	process_manager.check()
	_, invalid_source_paths = validate_source_paths(source_paths)
	if invalid_source_paths:
		for index in source_set:
			if source_set.get(index).get('path') in invalid_source_paths:
				invalid_source_url = source_set.get(index).get('url')
				if invalid_source_url:
					download_directory_path = os.path.dirname(source_set.get(index).get('path'))
					conditional_download(download_directory_path, [ invalid_source_url ])

	valid_source_paths, invalid_source_paths = validate_source_paths(source_paths)

	for valid_source_path in valid_source_paths:
		valid_source_file_name = get_file_name(valid_source_path)
		logger.debug(translator.get('validating_source_succeeded').format(source_file_name = valid_source_file_name), __name__)
	for invalid_source_path in invalid_source_paths:
		invalid_source_file_name = get_file_name(invalid_source_path)
		logger.error(translator.get('validating_source_failed').format(source_file_name = invalid_source_file_name), __name__)

		if remove_file(invalid_source_path):
			logger.error(translator.get('deleting_corrupt_source').format(source_file_name = invalid_source_file_name), __name__)

	if not invalid_source_paths:
		process_manager.end()
	return not invalid_source_paths


def validate_hash_paths(hash_paths : List[str]) -> Tuple[List[str], List[str]]:
	valid_hash_paths = []
	invalid_hash_paths = []

	for hash_path in hash_paths:
		if is_file(hash_path):
			valid_hash_paths.append(hash_path)
		else:
			invalid_hash_paths.append(hash_path)

	return valid_hash_paths, invalid_hash_paths


def validate_source_paths(source_paths : List[str]) -> Tuple[List[str], List[str]]:
	valid_source_paths = []
	invalid_source_paths = []

	for source_path in source_paths:
		if validate_hash(source_path):
			valid_source_paths.append(source_path)
		else:
			invalid_source_paths.append(source_path)

	return valid_source_paths, invalid_source_paths


def resolve_download_url(base_name : str, file_name : str) -> Optional[str]:
	download_providers = state_manager.get_item('download_providers')

	for download_provider in download_providers:
		download_url = resolve_download_url_by_provider(download_provider, base_name, file_name)
		if download_url:
			return download_url

	return None


def resolve_download_url_by_provider(download_provider : DownloadProvider, base_name : str, file_name : str) -> Optional[str]:
	download_provider_value = facefusion.choices.download_provider_set.get(download_provider)

	for download_provider_url in download_provider_value.get('urls'):
		if ping_static_url(download_provider_url):
			return download_provider_url + download_provider_value.get('path').format(base_name = base_name, file_name = file_name)

	return None
