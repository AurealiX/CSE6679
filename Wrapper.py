import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, font
import subprocess
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TPSolverGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Transportation Problem Solver")
        self.root.geometry("1000x800")
        
        # Center the window on the screen
        self.center_window(self.root)
        
        # Increase font size
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(size=16)
        
        text_font = font.nametofont("TkTextFont")
        text_font.configure(size=16)
        
        # Configure matplotlib to use larger fonts
        plt.rcParams.update({'font.size': 16})
        
        # Polynomial models for execution time prediction
        self.models = {
            'VAM_CPU': {'degree': 4, 'params': [6.22354866e-06, 2.84308558e-06, 1.05294683e-07, 5.48271174e-13, 1.97989930e-11]},
            'VAM_GPU': {'degree': 4, 'params': [4.63954544e-04, 1.38028559e-03, 9.73070679e-07, 3.11757416e-23, 3.88374973e-27]},
            'LCM_CPU': {'degree': 4, 'params': [2.18812439e-06, 1.22000436e-08, 3.13943073e-08, 5.09655157e-10, 3.31380715e-12]},
            'LCM_GPU': {'degree': 1, 'params': [0.0050572, 0.00044967]},
            'MODI_CPU': {'degree': 4, 'params': [6.70434641e-14, 1.60302870e-32, 2.40317199e-16, 1.00600472e-08, 3.43189318e-10]},
            'MODI_GPU': {'degree': 2, 'params': [1.88255778e-16, 2.78596442e-03, 3.94197442e-05]},
            'SSM_CPU': {'degree': 4, 'params': [6.55699129e-17, 1.34646054e-16, 2.91500904e-16, 6.62088713e-16, 3.45300645e-07]},
            'SSM_GPU': {'degree': 3, 'params': [4.60654346e-13, 2.71882522e-14, 2.63145681e-04, 2.28371761e-07]}
        }

        
        # Check for TPSolver and compile if needed
        self.check_executable()
        
        # Create the GUI
        self.create_gui()
    
    def center_window(self, window):
        """Center a window on the screen"""
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        
    def check_executable(self):
        if not os.path.exists("TPSolver") and os.path.exists("Makefile"):
            try:
                subprocess.run(["make", "clean"], check=True)
                subprocess.run(["make"], check=True)
                
                # Create centered message box
                msg = tk.Toplevel(self.root)
                msg.title("Compilation")
                msg.geometry("300x100")
                self.center_window(msg)
                
                ttk.Label(msg, text="TPSolver was successfully compiled.", font=('', 12)).pack(pady=20)
                ttk.Button(msg, text="OK", command=msg.destroy).pack()
                
                msg.transient(self.root)
                msg.grab_set()
                self.root.wait_window(msg)
                
            except subprocess.CalledProcessError:
                msg = tk.Toplevel(self.root)
                msg.title("Compilation Error")
                msg.geometry("400x150")
                self.center_window(msg)
                
                ttk.Label(msg, text="Failed to compile TPSolver.\nPlease check your Makefile and dependencies.", 
                          font=('', 12)).pack(pady=20)
                ttk.Button(msg, text="OK", command=msg.destroy).pack()
                
                msg.transient(self.root)
                msg.grab_set()
                self.root.wait_window(msg)
                
        elif not os.path.exists("TPSolver") and not os.path.exists("Makefile"):
            msg = tk.Toplevel(self.root)
            msg.title("Missing Files")
            msg.geometry("450x150")
            self.center_window(msg)
            
            ttk.Label(msg, text="TPSolver executable and Makefile not found.\nPlease ensure you're in the correct directory.", 
                      font=('', 12)).pack(pady=20)
            ttk.Button(msg, text="OK", command=msg.destroy).pack()
            
            msg.transient(self.root)
            msg.grab_set()
            self.root.wait_window(msg)
    
    def create_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Input Parameters", padding="10")
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Supply nodes
        ttk.Label(input_frame, text="Supply Nodes:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.supply_var = tk.StringVar(value="100")
        supply_entry = ttk.Entry(input_frame, textvariable=self.supply_var, width=10, font=('', 12))
        supply_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Demand nodes
        ttk.Label(input_frame, text="Demand Nodes:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.demand_var = tk.StringVar(value="100")
        demand_entry = ttk.Entry(input_frame, textvariable=self.demand_var, width=10, font=('', 12))
        demand_entry.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Seed
        ttk.Label(input_frame, text="Random Seed:").grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        self.seed_var = tk.StringVar(value="1")
        seed_entry = ttk.Entry(input_frame, textvariable=self.seed_var, width=10, font=('', 12))
        seed_entry.grid(row=0, column=5, sticky=tk.W, padx=5, pady=5)
        
        # Algorithm selection frame
        alg_frame = ttk.LabelFrame(main_frame, text="Algorithm Selection", padding="10")
        alg_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Automatic/manual selection
        self.selection_mode = tk.StringVar(value="auto")
        auto_radio = ttk.Radiobutton(alg_frame, text="Automatic (Best Performance)", 
                                    variable=self.selection_mode, value="auto",
                                    command=self.toggle_manual_selection)
        auto_radio.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        manual_radio = ttk.Radiobutton(alg_frame, text="Manual Selection", 
                                      variable=self.selection_mode, value="manual",
                                      command=self.toggle_manual_selection)
        manual_radio.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Manual selection options
        self.manual_frame = ttk.Frame(alg_frame)
        self.manual_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Phase 1 algorithm
        ttk.Label(self.manual_frame, text="Phase 1 Algorithm:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.p1_alg_var = tk.StringVar(value="VAM")
        p1_alg_combo = ttk.Combobox(self.manual_frame, textvariable=self.p1_alg_var, values=["VAM", "LCM"], 
                                    state="readonly", width=10, font=('', 12))
        p1_alg_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Phase 1 hardware
        ttk.Label(self.manual_frame, text="Phase 1 Hardware:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.p1_hw_var = tk.StringVar(value="CPU")
        p1_hw_combo = ttk.Combobox(self.manual_frame, textvariable=self.p1_hw_var, values=["CPU", "GPU"], 
                                  state="readonly", width=10, font=('', 12))
        p1_hw_combo.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Phase 2 algorithm
        ttk.Label(self.manual_frame, text="Phase 2 Algorithm:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.p2_alg_var = tk.StringVar(value="MODI")
        p2_alg_combo = ttk.Combobox(self.manual_frame, textvariable=self.p2_alg_var, values=["MODI", "SSM"], 
                                    state="readonly", width=10, font=('', 12))
        p2_alg_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Phase 2 hardware
        ttk.Label(self.manual_frame, text="Phase 2 Hardware:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.p2_hw_var = tk.StringVar(value="CPU")
        p2_hw_combo = ttk.Combobox(self.manual_frame, textvariable=self.p2_hw_var, values=["CPU", "GPU"], 
                                  state="readonly", width=10, font=('', 12))
        p2_hw_combo.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)
        
        # Disable manual selection by default
        self.manual_frame.grid_remove()
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_style = ttk.Style()
        btn_style.configure('Large.TButton', font=('', 12))
        
        analyze_btn = ttk.Button(btn_frame, text="Analyze Performance", command=self.analyze_performance, style='Large.TButton')
        analyze_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        run_btn = ttk.Button(btn_frame, text="Run Solver", command=self.run_solver, style='Large.TButton')
        run_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Results section
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Output tab
        output_frame = ttk.Frame(self.notebook)
        self.notebook.add(output_frame, text="Solver Output")
        
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=15, font=('Courier', 12))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Performance tab
        self.perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.perf_frame, text="Performance Analysis")
        
        # Will be populated with matplotlib figure when analyze_performance is called
        
    def toggle_manual_selection(self):
        if self.selection_mode.get() == "manual":
            self.manual_frame.grid()
        else:
            self.manual_frame.grid_remove()
    
    def predict_runtime(self, model_name, size):
        model = self.models[model_name]
        degree = model['degree']
        params = model['params']
        
        # Calculate polynomial: params[0] + params[1]*x + params[2]*x^2 + ...
        return sum(params[i] * (size ** i) for i in range(degree + 1))
    
    def determine_best_algorithms(self, supply_size, demand_size):
        # For simplicity, we'll use supply_size for predictions as mentioned in the requirements
        
        # Phase 1 selection
        vam_cpu_time = self.predict_runtime("VAM_CPU", supply_size)
        vam_gpu_time = self.predict_runtime("VAM_GPU", supply_size)
        lcm_cpu_time = self.predict_runtime("LCM_CPU", supply_size)
        lcm_gpu_time = self.predict_runtime("LCM_GPU", supply_size)
        # Find minimum time and corresponding algorithm
        p1_times = {
            ("VAM", "CPU"): vam_cpu_time,
            ("VAM", "GPU"): vam_gpu_time,
            ("LCM", "CPU"): lcm_cpu_time,
            ("LCM", "GPU"): lcm_gpu_time
        }
        p1_alg, p1_hw = min(p1_times, key=p1_times.get)
        
        # Phase 2 selection
        modi_cpu_time = self.predict_runtime("MODI_CPU", supply_size)
        modi_gpu_time = self.predict_runtime("MODI_GPU", supply_size)
        ssm_cpu_time = self.predict_runtime("SSM_CPU", supply_size)
        ssm_gpu_time = self.predict_runtime("SSM_GPU", supply_size)
        
        # Find minimum time and corresponding algorithm
        p2_times = {
            ("MODI", "CPU"): modi_cpu_time,
            ("MODI", "GPU"): modi_gpu_time,
            ("SSM", "CPU"): ssm_cpu_time,
            ("SSM", "GPU"): ssm_gpu_time
        }
        
        p2_alg, p2_hw = min(p2_times, key=p2_times.get)
        
        # Return the best combination and estimated times
        return {
            "phase1": {"alg": p1_alg, "hw": p1_hw},
            "phase2": {"alg": p2_alg, "hw": p2_hw},
            "estimated_times": {
                "phase1": self.predict_runtime(f"{p1_alg}_{p1_hw}", supply_size),
                "phase2": self.predict_runtime(f"{p2_alg}_{p2_hw}", supply_size),
                "worst_case": max(
                    self.predict_runtime("VAM_CPU", supply_size),
                    self.predict_runtime("VAM_GPU", supply_size),
                    self.predict_runtime("LCM_CPU", supply_size),
                    self.predict_runtime("LCM_GPU", supply_size)
                ) + max(
                    self.predict_runtime("MODI_CPU", supply_size),
                    self.predict_runtime("MODI_GPU", supply_size),
                    self.predict_runtime("SSM_CPU", supply_size),
                    self.predict_runtime("SSM_GPU", supply_size)
                )
            }
        }
    
    def analyze_performance(self):
        try:
            supply_size = int(self.supply_var.get())
            demand_size = int(self.demand_var.get())
            
            if supply_size <= 0 or demand_size <= 0:
                raise ValueError("Supply and demand sizes must be positive integers")
            
            # Clear previous plot
            for widget in self.perf_frame.winfo_children():
                widget.destroy()
            
            # Create a new plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Size range for plots
            sizes = np.linspace(max(10, supply_size//2), min(20000, supply_size*2), 100)
            
            # Phase 1 algorithms
            ax1.semilogy(sizes, [self.predict_runtime("VAM_CPU", s) for s in sizes], label="VAM (CPU)")
            ax1.semilogy(sizes, [self.predict_runtime("VAM_GPU", s) for s in sizes], label="VAM (GPU)")
            ax1.semilogy(sizes, [self.predict_runtime("LCM_CPU", s) for s in sizes], label="LCM (CPU)")
            ax1.semilogy(sizes, [self.predict_runtime("LCM_GPU", s) for s in sizes], label="LCM (GPU)")
            
            ax1.set_title("Phase 1 Algorithms")
            ax1.set_xlabel("Problem Size")
            ax1.set_ylabel("Estimated Runtime (s)")
            ax1.legend()
            ax1.grid(True)
            
            # Phase 2 algorithms
            ax2.semilogy(sizes, [self.predict_runtime("MODI_CPU", s) for s in sizes], label="MODI (CPU)")
            ax2.semilogy(sizes, [self.predict_runtime("MODI_GPU", s) for s in sizes], label="MODI (GPU)")
            ax2.semilogy(sizes, [self.predict_runtime("SSM_CPU", s) for s in sizes], label="SSM (CPU)")
            ax2.semilogy(sizes, [self.predict_runtime("SSM_GPU", s) for s in sizes], label="SSM (GPU)")
            
            ax2.set_title("Phase 2 Algorithms")
            ax2.set_xlabel("Problem Size")
            ax2.set_ylabel("Estimated Runtime (s)")
            ax2.legend()
            ax2.grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            # Add current size marker
            ax1.axvline(x=supply_size, color='r', linestyle='--', label=f"Current Size ({supply_size})")
            ax2.axvline(x=supply_size, color='r', linestyle='--', label=f"Current Size ({supply_size})")
            
            # Add best choices at current size
            best = self.determine_best_algorithms(supply_size, demand_size)
            
            # Create text for best choices
            best_text = (
                f"Best Choices for Size {supply_size}:\n"
                f"Phase 1: {best['phase1']['alg']} on {best['phase1']['hw']} "
                f"(~{best['estimated_times']['phase1']:.6f} s)\n"
                f"Phase 2: {best['phase2']['alg']} on {best['phase2']['hw']} "
                f"(~{best['estimated_times']['phase2']:.6f} s)\n"
                f"Total: ~{best['estimated_times']['phase1'] + best['estimated_times']['phase2']:.6f} s\n"
                f"Worst Case: ~{best['estimated_times']['worst_case']:.6f} s"
            )
            
            # Text box for best choices
            text_frame = ttk.LabelFrame(self.perf_frame, text="Performance Summary")
            text_frame.pack(fill=tk.X, padx=5, pady=5)
            
            text_box = tk.Text(text_frame, height=6, width=60, wrap=tk.WORD, font=('', 12))
            text_box.pack(padx=5, pady=5)
            text_box.insert(tk.END, best_text)
            text_box.config(state=tk.DISABLED)
            
            # Add the plot to the GUI
            canvas = FigureCanvasTkAgg(fig, master=self.perf_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Update the notebook to show the performance tab
            self.notebook.select(self.perf_frame)
            
        except ValueError as e:
            # Create centered error message
            msg = tk.Toplevel(self.root)
            msg.title("Input Error")
            msg.geometry("300x150")
            self.center_window(msg)
            
            ttk.Label(msg, text=str(e), font=('', 12)).pack(pady=20)
            ttk.Button(msg, text="OK", command=msg.destroy, style='Large.TButton').pack()
            
            msg.transient(self.root)
            msg.grab_set()
            self.root.wait_window(msg)
    
    def get_binary_mode(self, p1_alg, p1_hw, p2_alg, p2_hw):
        # Convert algorithm and hardware choices to binary mode
        # First digit: 0=VAM, 1=LCM for phase 1
        d1 = '1' if p1_alg == 'LCM' else '0'
        
        # Second digit: 0=CPU, 1=GPU for phase 1
        d2 = '1' if p1_hw == 'GPU' else '0'
        
        # Third digit: 0=MODI, 1=SSM for phase 2
        d3 = '1' if p2_alg == 'SSM' else '0'
        
        # Fourth digit: 0=CPU, 1=GPU for phase 2
        d4 = '1' if p2_hw == 'GPU' else '0'
        
        return d1 + d2 + d3 + d4
    
    def run_solver(self):
        try:
            supply_size = int(self.supply_var.get())
            demand_size = int(self.demand_var.get())
            seed = int(self.seed_var.get())
            
            if supply_size <= 0 or demand_size <= 0:
                raise ValueError("Supply and demand sizes must be positive integers")
            
            # Determine algorithm selection
            if self.selection_mode.get() == "auto":
                best = self.determine_best_algorithms(supply_size, demand_size)
                p1_alg = best['phase1']['alg']
                p1_hw = best['phase1']['hw']
                p2_alg = best['phase2']['alg']
                p2_hw = best['phase2']['hw']
            else:
                p1_alg = self.p1_alg_var.get()
                p1_hw = self.p1_hw_var.get()
                p2_alg = self.p2_alg_var.get()
                p2_hw = self.p2_hw_var.get()
            
            # Get binary mode
            binary_mode = self.get_binary_mode(p1_alg, p1_hw, p2_alg, p2_hw)
            
            # Clear output
            self.output_text.delete(1.0, tk.END)
            
            # Build command and input
            cmd = ["./TPSolver"]
            
            # Format input as expected by the C program
            input_text = f"{seed}\n{supply_size}\n{demand_size}\n{binary_mode}\n"
            
            # Log the command
            self.output_text.insert(tk.END, f"Running: TPSolver with parameters:\n")
            self.output_text.insert(tk.END, f"  Seed: {seed}\n")
            self.output_text.insert(tk.END, f"  Supply Nodes: {supply_size}\n")
            self.output_text.insert(tk.END, f"  Demand Nodes: {demand_size}\n")
            self.output_text.insert(tk.END, f"  Binary Mode: {binary_mode} ")
            self.output_text.insert(tk.END, f"(Phase 1: {p1_alg} on {p1_hw}, Phase 2: {p2_alg} on {p2_hw})\n\n")
            
            # Run the command
            try:
                process = subprocess.Popen(
                    cmd, 
                    stdin=subprocess.PIPE, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = process.communicate(input=input_text)
                
                # Display output
                if stdout:
                    self.output_text.insert(tk.END, "Output:\n" + stdout + "\n")
                
                if stderr:
                    self.output_text.insert(tk.END, "Errors:\n" + stderr + "\n")
                
                # Select the output tab
                self.notebook.select(0)  # Index 0 is the output tab
                
            except Exception as e:
                self.output_text.insert(tk.END, f"Error running TPSolver: {str(e)}\n")
                
        except ValueError as e:
            # Create centered error message
            msg = tk.Toplevel(self.root)
            msg.title("Input Error")
            msg.geometry("300x150")
            self.center_window(msg)
            
            ttk.Label(msg, text=str(e), font=('', 12)).pack(pady=20)
            ttk.Button(msg, text="OK", command=msg.destroy, style='Large.TButton').pack()
            
            msg.transient(self.root)
            msg.grab_set()
            self.root.wait_window(msg)

if __name__ == "__main__":
    root = tk.Tk()
    app = TPSolverGUI(root)
    root.mainloop()