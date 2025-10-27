def ensure_openface_import(openface_root: Path) -> None:
    """Ensure the openface module can be imported.
    
    This function tries to import the openface package, which is installed
    via the 'openface-test' PyPI package.
    """
    # First, try importing openface directly (when openface-test is installed)
    try:
        import openface  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass
    
    # If direct import failed, try adding OpenFace-3.0 to sys.path
    openface_str = str(openface_root)
    if openface_str not in sys.path:
        sys.path.insert(0, openface_str)
    
    # Try importing again
    try:
        import openface  # type: ignore  # noqa: F401
    except ImportError as e:
        raise ImportError(
            f"Failed to import 'openface' module.\n\n"
            f"The 'openface' module is provided by the 'openface-test' package.\n"
            f"To fix this, run:\n"
            f"  cd {openface_root.parent}\n"
            f"  pip install openface-test\n"
            f"  openface download\n\n"
            f"Original error: {e}"
        ) from e
