#!/usr/bin/env python3
"""
fix_websocket.py - Diagnose and fix WebSocket compatibility issues

This script checks for WebSocket library conflicts and ensures the correct
version is installed for ComfyUI compatibility.
"""

import sys
import subprocess
import importlib
import pkg_resources


def check_websocket_installation():
    """Check WebSocket library installation and version."""
    print("Checking WebSocket installation...\n")
    
    # Check for websocket-client
    try:
        import websocket
        
        # Try to get version
        try:
            version = pkg_resources.get_distribution("websocket-client").version
            print(f"✓ websocket-client installed: version {version}")
        except:
            print("✓ websocket module found (version unknown)")
        
        # Check the WebSocket class
        if hasattr(websocket, 'WebSocket'):
            ws_class = websocket.WebSocket
            print(f"✓ WebSocket class found")
            
            # Check if it's the right WebSocket class
            import inspect
            sig = inspect.signature(ws_class.__init__)
            params = list(sig.parameters.keys())
            print(f"  Constructor parameters: {params}")
            
            if 'environ' in params:
                print("\n⚠️  ERROR: Wrong WebSocket class detected!")
                print("  This appears to be from 'websockets' library, not 'websocket-client'")
                return False
            else:
                print("✓ Correct WebSocket class (from websocket-client)")
                return True
        else:
            print("✗ WebSocket class not found in module")
            return False
            
    except ImportError:
        print("✗ websocket module not installed")
        return False


def check_conflicting_packages():
    """Check for conflicting WebSocket packages."""
    print("\nChecking for conflicting packages...")
    
    conflicts = []
    
    # Check for 'websockets' package (common conflict)
    try:
        import websockets
        try:
            version = pkg_resources.get_distribution("websockets").version
            conflicts.append(f"websockets=={version}")
        except:
            conflicts.append("websockets (version unknown)")
    except ImportError:
        pass
    
    # Check for 'simple-websocket'
    try:
        import simple_websocket
        try:
            version = pkg_resources.get_distribution("simple-websocket").version
            conflicts.append(f"simple-websocket=={version}")
        except:
            conflicts.append("simple-websocket (version unknown)")
    except ImportError:
        pass
    
    if conflicts:
        print(f"\n⚠️  Found conflicting packages: {', '.join(conflicts)}")
        return conflicts
    else:
        print("✓ No conflicting WebSocket packages found")
        return []


def fix_websocket_installation():
    """Fix WebSocket installation."""
    print("\n" + "="*60)
    print("FIXING WEBSOCKET INSTALLATION")
    print("="*60 + "\n")
    
    conflicts = check_conflicting_packages()
    
    if conflicts:
        print("Removing conflicting packages...")
        for package in conflicts:
            package_name = package.split('==')[0]
            print(f"  Uninstalling {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package_name])
                print(f"  ✓ Uninstalled {package_name}")
            except:
                print(f"  ✗ Failed to uninstall {package_name}")
    
    print("\nInstalling correct websocket-client version...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websocket-client==1.4.2", "--force-reinstall"])
        print("✓ Successfully installed websocket-client==1.4.2")
        return True
    except:
        print("✗ Failed to install websocket-client")
        return False


def test_websocket_connection():
    """Test WebSocket connection to ComfyUI."""
    print("\nTesting WebSocket connection to ComfyUI...")
    
    try:
        import websocket
        import uuid
        
        # Test connection
        client_id = str(uuid.uuid4())
        ws_url = f"ws://127.0.0.1:8188/ws?clientId={client_id}"
        
        print(f"Connecting to: {ws_url}")
        
        # Try the method used in comfy_api.py
        try:
            # First try WebSocket() constructor
            ws = websocket.WebSocket()
            ws.connect(ws_url)
            print("✓ Successfully connected using WebSocket() constructor")
            ws.close()
            return True
        except TypeError as e:
            print(f"✗ WebSocket() constructor failed: {e}")
            
            # Try create_connection as fallback
            try:
                ws = websocket.create_connection(ws_url, timeout=5)
                print("✓ Successfully connected using create_connection()")
                ws.close()
                print("\n⚠️  Note: You may need to update comfy_api.py to use:")
                print("    ws = websocket.create_connection(url)")
                print("    instead of:")
                print("    ws = websocket.WebSocket()")
                print("    ws.connect(url)")
                return True
            except Exception as e:
                print(f"✗ create_connection() also failed: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False


def main():
    """Main diagnostic and fix routine."""
    print("WebSocket Compatibility Diagnostic Tool")
    print("======================================\n")
    
    # Step 1: Check current installation
    correct_install = check_websocket_installation()
    
    # Step 2: Check for conflicts
    conflicts = check_conflicting_packages()
    
    # Step 3: Fix if needed
    if not correct_install or conflicts:
        print("\n⚠️  Issues detected. Attempting to fix...")
        
        response = input("\nProceed with fix? (y/n): ").strip().lower()
        if response == 'y':
            if fix_websocket_installation():
                print("\n✓ Fix applied. Please restart Python and try again.")
                
                # Re-check after fix
                print("\nVerifying fix...")
                # Need to reload modules
                if 'websocket' in sys.modules:
                    del sys.modules['websocket']
                
                correct_install = check_websocket_installation()
                if correct_install:
                    print("\n✓ WebSocket installation is now correct!")
                else:
                    print("\n✗ Fix may require Python restart to take effect")
            else:
                print("\n✗ Fix failed. Manual intervention may be required.")
        else:
            print("\nFix cancelled.")
    else:
        print("\n✓ WebSocket installation appears correct!")
    
    # Step 4: Test connection
    if correct_install:
        test_websocket_connection()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if correct_install and not conflicts:
        print("✓ WebSocket setup is correct")
        print("\nIf you're still having issues, check that:")
        print("1. ComfyUI is running on http://127.0.0.1:8188")
        print("2. No firewall is blocking the connection")
        print("3. The workflow JSON files are in the correct location")
    else:
        print("✗ WebSocket setup has issues")
        print("\nRecommended actions:")
        print("1. Run this script with fix option")
        print("2. Restart Python/terminal after fix")
        print("3. Ensure only websocket-client==1.4.2 is installed")
    
    print("\nFor manual fix, run:")
    print("  pip uninstall -y websockets websocket websocket-client")
    print("  pip install websocket-client==1.4.2")


if __name__ == "__main__":
    main()