#!/usr/bin/env python3
"""
Simple startup script for HydroAlert
Starts basic services without ML dependencies
"""

import subprocess
import sys
import time
import os
import socket
import platform


def is_port_in_use(port: int) -> bool:
    """Check if a port is already in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return False
        except OSError:
            return True


def kill_process_on_port(port: int) -> bool:
    """Kill process using the specified port"""
    try:
        if platform.system() == "Windows":
            # Windows: Find process using the port and kill it
            result = subprocess.run(
                ['netstat', '-ano'], 
                capture_output=True, 
                text=True
            )
            for line in result.stdout.split('\n'):
                if f':{port}' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        try:
                            subprocess.run(['taskkill', '/F', '/PID', pid], 
                                         capture_output=True, check=False)
                            print(f"🔄 Killed process on port {port} (PID: {pid})")
                            time.sleep(1)
                            return True
                        except:
                            pass
        else:
            # Unix/Linux/Mac: Use lsof to find and kill process
            result = subprocess.run(
                ['lsof', '-ti', f':{port}'], 
                capture_output=True, 
                text=True
            )
            if result.stdout.strip():
                pid = result.stdout.strip().split('\n')[0]
                subprocess.run(['kill', '-9', pid], capture_output=True)
                print(f"🔄 Killed process on port {port} (PID: {pid})")
                time.sleep(1)
                return True
    except Exception as e:
        print(f"⚠️  Could not kill process on port {port}: {e}")
    return False


def start_backend():
    """Start the backend API server"""
    backend_port = 8000
    print("🚀 Starting HydroAlert Backend...")
    
    # Check if port is in use
    if is_port_in_use(backend_port):
        print(f"⚠️  Port {backend_port} is already in use. Attempting to free it...")
        if kill_process_on_port(backend_port):
            time.sleep(2)  # Wait for port to be released
        else:
            print(f"❌ Could not free port {backend_port}. Please close the application using it manually.")
            return False
    
    try:
        # Start backend with uvicorn directly
        subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "backend.main_simple:app", 
            "--host", "0.0.0.0", 
            "--port", str(backend_port),
            "--reload"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Give it time to start
        print(f"✅ Backend started on http://localhost:{backend_port}")
        return True
    except Exception as e:
        print(f"❌ Failed to start backend: {e}")
        return False

def start_frontend():
    """Start the Streamlit frontend"""
    frontend_port = 8501
    print("🚀 Starting HydroAlert Frontend...")
    
    # Check if port is in use
    if is_port_in_use(frontend_port):
        print(f"⚠️  Port {frontend_port} is already in use. Attempting to free it...")
        if kill_process_on_port(frontend_port):
            time.sleep(2)  # Wait for port to be released
        else:
            print(f"❌ Could not free port {frontend_port}. Please close the application using it manually.")
            return False
    
    try:
        subprocess.Popen([
            sys.executable, "-m", "streamlit", "run",
            "frontend/streamlit_mvp.py",
            "--server.port", str(frontend_port),
            "--server.address", "localhost"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)  # Give it time to start
        print(f"✅ Frontend started on http://localhost:{frontend_port}")
        return True
    except Exception as e:
        print(f"❌ Failed to start frontend: {e}")
        return False

def main():
    print("🚰 Welcome to HydroAlert!")
    print("Smart Water Intake Monitor")
    print("=" * 50)
    
    # Start services
    backend_ok = start_backend()
    time.sleep(2)  # Give backend time to start
    
    frontend_ok = start_frontend()
    
    if backend_ok and frontend_ok:
        print("\n🎉 HydroAlert is starting up!")
        print("📱 Frontend: http://localhost:8501")
        print("🔧 Backend API: http://localhost:8000")
        print("📊 API Docs: http://localhost:8000/docs")
        print("\n⏳ Services are starting... Please wait a moment.")
        print("Press Ctrl+C to stop all services.")
        print("\n💡 Tip: If you see port errors, close any existing instances first.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down HydroAlert...")
    else:
        print("\n❌ Failed to start some services. Check the error messages above.")

if __name__ == "__main__":
    main() 