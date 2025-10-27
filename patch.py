def ensure_openface_import(openface_root: Path) -> None:
    """Ensure the openface module can be imported."""
    # First try direct import
    try:
        import openface
        return
    except ImportError as e:
        print(f"Direct import failed: {e}")
        print(f"sys.path entries: {sys.path[:5]}")  # Debug info
    
    # Try adding OpenFace-3.0 directory to path
    openface_str = str(openface_root)
    if openface_str not in sys.path:
        sys.path.insert(0, openface_str)
        print(f"Added to sys.path: {openface_str}")
    
    # Try import again
    try:
        import openface
        print("Successfully imported openface after adding to path")
    except ImportError as e:
        raise ImportError(
            f"Failed to import 'openface' module.\n\n"
            f"Python version: {sys.version_info}\n"
            f"Python executable: {sys.executable}\n"
            f"Attempted paths:\n"
            f"  - Direct import\n"
            f"  - {openface_str}\n\n"
            f"Make sure 'openface-test' is installed:\n"
            f"  pip install openface-test\n"
            f"  openface download\n\n"
            f"Original error: {e}"
        ) from e
