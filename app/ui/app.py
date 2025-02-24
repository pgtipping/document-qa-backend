"""Main entry point for the Document Q&A application."""
from app.ui.interface import launch_interface
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main() -> None:
    """Launch the Gradio interface."""
    launch_interface()


if __name__ == "__main__":
    main()
