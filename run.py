import os
import sys
import uvicorn

def main():
    """Run the FastAPI server with the correct Python path."""
    # Get the absolute path to the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the backend directory to Python path
    sys.path.append(backend_dir)
    
    # Change to the backend directory
    os.chdir(backend_dir)
    
    # Run the server
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )

if __name__ == "__main__":
    main() 