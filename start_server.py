#!/usr/bin/env python3
"""
RoomEQ API Server Starter

Starts the RoomEQ REST API server on port 10315.
"""

import sys
import os

# Add the src path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import uvicorn
from roomeq.roomeq_server import app

def main():
    print("Starting RoomEQ API Server on port 10315...")
    print("API documentation available at: http://localhost:10315/docs")
    print("API status at: http://localhost:10315/")
    print()
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0", 
            port=10315,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
