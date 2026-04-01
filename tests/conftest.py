import os


def _resolve_data_path(default_path):
    if os.path.exists(default_path):
        return default_path

    fallback_path = os.path.join('/data', os.path.basename(default_path))
    if os.path.exists(fallback_path):
        return fallback_path

    raise FileNotFoundError(
        f"File not found at default path '{default_path}' or fallback path '{fallback_path}'."
    )
