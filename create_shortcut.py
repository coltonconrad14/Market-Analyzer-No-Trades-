#!/usr/bin/env python3
"""
Desktop Shortcut Creator for Market Analyzer GUI
Creates a desktop shortcut to easily launch the GUI app.
"""

import os
import sys
import platform


def create_linux_desktop_entry():
    """Create a .desktop file for Linux systems."""
    desktop_dir = os.path.expanduser("~/Desktop")
    if not os.path.exists(desktop_dir):
        desktop_dir = os.path.expanduser("~/.local/share/applications")
    
    # Get absolute path to app.py
    app_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(app_dir, "app.py")
    
    desktop_entry = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=Market Analyzer
Comment=Analyze stocks and cryptocurrencies
Exec=python3 "{app_path}"
Icon=utilities-terminal
Terminal=false
Categories=Office;Finance;
"""
    
    desktop_file = os.path.join(desktop_dir, "MarketAnalyzer.desktop")
    
    try:
        with open(desktop_file, 'w') as f:
            f.write(desktop_entry)
        
        # Make it executable
        os.chmod(desktop_file, 0o755)
        
        print(f"✅ Desktop shortcut created: {desktop_file}")
        print(f"   You can now launch Market Analyzer from your desktop!")
        return True
    except Exception as e:
        print(f"❌ Error creating desktop entry: {e}")
        return False


def create_windows_shortcut():
    """Create a .bat shortcut file for Windows."""
    try:
        import winshell
        from win32com.client import Dispatch
        
        desktop = winshell.desktop()
        app_dir = os.path.dirname(os.path.abspath(__file__))
        app_path = os.path.join(app_dir, "app.py")
        
        shell = Dispatch('WScript.Shell')
        shortcut_path = os.path.join(desktop, "Market Analyzer.lnk")
        shortcut = shell.CreateShortCut(shortcut_path)
        shortcut.Targetpath = sys.executable
        shortcut.Arguments = f'"{app_path}"'
        shortcut.WorkingDirectory = app_dir
        shortcut.IconLocation = sys.executable
        shortcut.save()
        
        print(f"✅ Desktop shortcut created: {shortcut_path}")
        return True
    except ImportError:
        # Fallback: create a simple .bat file
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.exists(desktop):
            desktop = os.path.expanduser("~")
        
        app_dir = os.path.dirname(os.path.abspath(__file__))
        bat_content = f"""@echo off
cd /d "{app_dir}"
python app.py
pause
"""
        bat_file = os.path.join(desktop, "Market Analyzer.bat")
        
        try:
            with open(bat_file, 'w') as f:
                f.write(bat_content)
            print(f"✅ Desktop shortcut created: {bat_file}")
            print(f"   Double-click to launch Market Analyzer!")
            return True
        except Exception as e:
            print(f"❌ Error creating shortcut: {e}")
            return False
    except Exception as e:
        print(f"❌ Error creating shortcut: {e}")
        return False


def create_macos_app():
    """Instructions for macOS (no automatic creation)."""
    app_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(app_dir, "app.py")
    
    print("📱 macOS Instructions:")
    print("\nTo create a launcher on macOS:")
    print("1. Open 'Automator'")
    print("2. Create a new 'Application'")
    print("3. Add 'Run Shell Script' action")
    print(f"4. Enter: python3 '{app_path}'")
    print("5. Save as 'Market Analyzer' to Desktop")
    print("\nOr simply run: python3 app.py")
    return True


def main():
    """Create appropriate shortcut based on OS."""
    print("\n" + "="*60)
    print("Market Analyzer - Desktop Shortcut Creator")
    print("="*60 + "\n")
    
    system = platform.system()
    
    if system == "Linux":
        success = create_linux_desktop_entry()
    elif system == "Windows":
        success = create_windows_shortcut()
    elif system == "Darwin":  # macOS
        success = create_macos_app()
    else:
        print(f"❌ Unsupported operating system: {system}")
        print(f"   Please run: python app.py")
        success = False
    
    if success:
        print("\n🎉 Setup complete!")
        print("   You can also always run: python app.py")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
