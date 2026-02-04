#!/usr/bin/env python3
"""
GUI Application for Market Analyzer.
Simple graphical interface to analyze financial assets with ease.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from market_analyzer import MarketAnalyzer
import threading
import sys
from io import StringIO


class MarketAnalyzerApp:
    """GUI application for Market Analyzer."""
    
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("Market Analyzer - Easy GUI")
        self.root.geometry("900x700")
        
        # Initialize analyzer
        self.analyzer = MarketAnalyzer()
        self.is_analyzing = False
        
        # Configure style
        self.setup_style()
        
        # Create UI
        self.create_widgets()
        
    def setup_style(self):
        """Setup custom styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 10, 'italic'), foreground='#7f8c8d')
        style.configure('Action.TButton', font=('Arial', 11, 'bold'), padding=10)
        
    def create_widgets(self):
        """Create all UI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Header
        self.create_header(main_frame)
        
        # Input section
        self.create_input_section(main_frame)
        
        # Buttons section
        self.create_buttons_section(main_frame)
        
        # Results section
        self.create_results_section(main_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
        
    def create_header(self, parent):
        """Create header section."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        title = ttk.Label(header_frame, text="📊 Market Analyzer", style='Title.TLabel')
        title.grid(row=0, column=0, sticky=tk.W)
        
        subtitle = ttk.Label(
            header_frame, 
            text="⚠️  FOR INFORMATIONAL PURPOSES ONLY - NO TRADING",
            style='Subtitle.TLabel'
        )
        subtitle.grid(row=1, column=0, sticky=tk.W)
        
    def create_input_section(self, parent):
        """Create input fields section."""
        input_frame = ttk.LabelFrame(parent, text="Asset Information", padding="15")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        input_frame.columnconfigure(1, weight=1)
        
        # Symbol input
        ttk.Label(input_frame, text="Symbol:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.symbol_entry = ttk.Entry(input_frame, font=('Arial', 10))
        self.symbol_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0), pady=5)
        self.symbol_entry.insert(0, "AAPL")
        
        ttk.Label(input_frame, text="(e.g., AAPL, MSFT, BTC-USD, ETH-USD)", 
                 font=('Arial', 8, 'italic'), foreground='gray').grid(
            row=0, column=2, sticky=tk.W, padx=(10, 0), pady=5
        )
        
        # Period selection
        ttk.Label(input_frame, text="Period:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.period_var = tk.StringVar(value="1y")
        period_combo = ttk.Combobox(
            input_frame, 
            textvariable=self.period_var,
            values=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            state="readonly",
            width=15
        )
        period_combo.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # Asset type selection
        ttk.Label(input_frame, text="Asset Type:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.asset_type_var = tk.StringVar(value="stock")
        asset_type_frame = ttk.Frame(input_frame)
        asset_type_frame.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        ttk.Radiobutton(
            asset_type_frame, 
            text="Stock", 
            variable=self.asset_type_var, 
            value="stock"
        ).grid(row=0, column=0, padx=(0, 20))
        
        ttk.Radiobutton(
            asset_type_frame, 
            text="Cryptocurrency", 
            variable=self.asset_type_var, 
            value="crypto"
        ).grid(row=0, column=1)
        
    def create_buttons_section(self, parent):
        """Create buttons section."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, pady=(0, 15))
        
        # Analyze button
        self.analyze_btn = ttk.Button(
            button_frame,
            text="🔍 Analyze Asset",
            style='Action.TButton',
            command=self.analyze_asset
        )
        self.analyze_btn.grid(row=0, column=0, padx=5)
        
        # Compare button
        self.compare_btn = ttk.Button(
            button_frame,
            text="⚖️  Compare Assets",
            command=self.show_compare_dialog
        )
        self.compare_btn.grid(row=0, column=1, padx=5)
        
        # Clear button
        clear_btn = ttk.Button(
            button_frame,
            text="🗑️  Clear Results",
            command=self.clear_results
        )
        clear_btn.grid(row=0, column=2, padx=5)
        
        # Quick examples dropdown
        ttk.Label(button_frame, text="Quick Examples:").grid(row=0, column=3, padx=(20, 5))
        self.example_var = tk.StringVar()
        example_combo = ttk.Combobox(
            button_frame,
            textvariable=self.example_var,
            values=["AAPL - Apple", "MSFT - Microsoft", "GOOGL - Google", 
                   "TSLA - Tesla", "BTC-USD - Bitcoin", "ETH-USD - Ethereum"],
            state="readonly",
            width=20
        )
        example_combo.grid(row=0, column=4, padx=5)
        example_combo.bind("<<ComboboxSelected>>", self.load_example)
        
    def create_results_section(self, parent):
        """Create results display section."""
        results_frame = ttk.LabelFrame(parent, text="Analysis Results", padding="10")
        results_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Scrolled text widget
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            wrap=tk.WORD,
            font=('Courier', 10),
            height=20
        )
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure text tags for colored output
        self.results_text.tag_config('header', font=('Courier', 11, 'bold'))
        self.results_text.tag_config('success', foreground='green')
        self.results_text.tag_config('warning', foreground='orange')
        self.results_text.tag_config('error', foreground='red')
        
    def create_status_bar(self, parent):
        """Create status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            parent, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Arial', 9)
        )
        status_bar.grid(row=5, column=0, sticky=(tk.W, tk.E))
        
    def load_example(self, event=None):
        """Load an example into the input fields."""
        example = self.example_var.get()
        if example:
            symbol = example.split(" - ")[0]
            self.symbol_entry.delete(0, tk.END)
            self.symbol_entry.insert(0, symbol)
            
            # Auto-detect crypto
            if "-USD" in symbol:
                self.asset_type_var.set("crypto")
            else:
                self.asset_type_var.set("stock")
                
    def analyze_asset(self):
        """Analyze the specified asset."""
        if self.is_analyzing:
            messagebox.showwarning("In Progress", "Analysis already in progress. Please wait.")
            return
            
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            messagebox.showerror("Error", "Please enter a symbol!")
            return
            
        # Disable buttons
        self.set_analyzing_state(True)
        
        # Run analysis in separate thread
        thread = threading.Thread(target=self._run_analysis, args=(symbol,))
        thread.daemon = True
        thread.start()
        
    def _run_analysis(self, symbol):
        """Run analysis in background thread."""
        try:
            period = self.period_var.get()
            asset_type = self.asset_type_var.get()
            
            self.update_status(f"Analyzing {symbol}...")
            self.append_results(f"\n{'='*60}\n", 'header')
            self.append_results(f"Analyzing {symbol} ({period}, {asset_type})\n", 'header')
            self.append_results(f"{'='*60}\n\n", 'header')
            
            # Capture print output
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            # Run analysis
            analysis = self.analyzer.analyze_asset(symbol, period=period, asset_type=asset_type)
            
            if analysis:
                self.analyzer.print_analysis(analysis)
                output = captured_output.getvalue()
                self.append_results(output)
                self.update_status(f"Analysis complete for {symbol}")
            else:
                self.append_results(f"❌ Failed to analyze {symbol}\n", 'error')
                self.append_results("Please check the symbol and try again.\n", 'error')
                self.update_status("Analysis failed")
                
            # Restore stdout
            sys.stdout = old_stdout
            
        except Exception as e:
            self.append_results(f"\n❌ Error: {str(e)}\n", 'error')
            self.update_status("Error occurred")
        finally:
            self.set_analyzing_state(False)
            
    def show_compare_dialog(self):
        """Show dialog to compare multiple assets."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Compare Assets")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Instructions
        ttk.Label(
            dialog,
            text="Enter symbols to compare (one per line):",
            font=('Arial', 10, 'bold')
        ).pack(pady=10)
        
        # Text input
        text_frame = ttk.Frame(dialog, padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        symbols_text = scrolledtext.ScrolledText(text_frame, height=10, width=40)
        symbols_text.pack(fill=tk.BOTH, expand=True)
        symbols_text.insert('1.0', "AAPL\nMSFT\nGOOGL\nTSLA\nBTC-USD\nETH-USD")
        
        # Period selection
        period_frame = ttk.Frame(dialog, padding=10)
        period_frame.pack()
        
        ttk.Label(period_frame, text="Period:").pack(side=tk.LEFT, padx=5)
        compare_period_var = tk.StringVar(value="3mo")
        ttk.Combobox(
            period_frame,
            textvariable=compare_period_var,
            values=["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y"],
            state="readonly",
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(dialog, padding=10)
        button_frame.pack()
        
        def run_comparison():
            symbols_input = symbols_text.get('1.0', tk.END).strip()
            symbols = [s.strip().upper() for s in symbols_input.split('\n') if s.strip()]
            
            if not symbols:
                messagebox.showerror("Error", "Please enter at least one symbol!")
                return
                
            dialog.destroy()
            self._run_comparison(symbols, compare_period_var.get())
            
        ttk.Button(button_frame, text="Compare", command=run_comparison).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
    def _run_comparison(self, symbols, period):
        """Run comparison in background thread."""
        if self.is_analyzing:
            messagebox.showwarning("In Progress", "Analysis already in progress. Please wait.")
            return
            
        self.set_analyzing_state(True)
        thread = threading.Thread(target=self._do_comparison, args=(symbols, period))
        thread.daemon = True
        thread.start()
        
    def _do_comparison(self, symbols, period):
        """Perform comparison analysis."""
        try:
            self.update_status(f"Comparing {len(symbols)} assets...")
            self.append_results(f"\n{'='*60}\n", 'header')
            self.append_results(f"Comparing: {', '.join(symbols)}\n", 'header')
            self.append_results(f"Period: {period}\n", 'header')
            self.append_results(f"{'='*60}\n\n", 'header')
            
            comparison = self.analyzer.compare_assets(symbols, period=period)
            
            if not comparison.empty:
                self.append_results(comparison.to_string(index=False))
                self.append_results("\n")
                self.update_status(f"Comparison complete for {len(symbols)} assets")
            else:
                self.append_results("❌ Failed to compare assets\n", 'error')
                self.update_status("Comparison failed")
                
        except Exception as e:
            self.append_results(f"\n❌ Error: {str(e)}\n", 'error')
            self.update_status("Error occurred")
        finally:
            self.set_analyzing_state(False)
            
    def clear_results(self):
        """Clear the results text area."""
        self.results_text.delete('1.0', tk.END)
        self.update_status("Results cleared")
        
    def append_results(self, text, tag=None):
        """Append text to results area (thread-safe)."""
        def _append():
            self.results_text.insert(tk.END, text, tag)
            self.results_text.see(tk.END)
        self.root.after(0, _append)
        
    def update_status(self, message):
        """Update status bar (thread-safe)."""
        self.root.after(0, lambda: self.status_var.set(message))
        
    def set_analyzing_state(self, analyzing):
        """Enable/disable buttons during analysis."""
        def _update():
            self.is_analyzing = analyzing
            state = 'disabled' if analyzing else 'normal'
            self.analyze_btn.config(state=state)
            self.compare_btn.config(state=state)
        self.root.after(0, _update)


def main():
    """Launch the GUI application."""
    root = tk.Tk()
    app = MarketAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
