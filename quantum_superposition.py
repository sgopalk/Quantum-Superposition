#Simulation of the Schrödinger equation to demonstrate quantum superposition and state collapse
#Developed by Dr. Gopal Kashyap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import streamlit as st
import streamlit.components.v1 as components
import tempfile
import base64
import os
from scipy.special import hermite
from scipy.special import factorial
# Constants
hbar = 1.0
m = 1.0
omega=1.0
alpha=1.0
L=1.0
#For particle in a 1D Box

# Define eigenfunction
def psi_n(n, x, L):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

# Define energy
def energy_n(n, L):
    return (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)

# Define superposed wavefunction
def psi_superposed(x, t, L, ns, cs):
    psi = np.zeros_like(x, dtype=complex)
    for n, c in zip(ns, cs):
        psi += c * psi_n(n, x, L) * np.exp(-1j * energy_n(n,L) * t / hbar)
    return psi
    
#For Harmonic oscillator potential    
def psi_ho(n, x):
    # Hermite polynomial
    Hn = hermite(n)(x)
    # Normalization
    norm = 1.0/np.sqrt((2.0**n) * factorial(n) * np.sqrt(np.pi))
    psi = norm * np.exp(-x**2/2) * Hn
    return psi

# Define energy
def energy_ho(n):
    return (n+1/2)*hbar*omega

# Define superposed wavefunction
def psi_superposed_ho(x, t, ns, cs):
    psi = np.zeros_like(x, dtype=complex)
    for n, c in zip(ns, cs):
        psi += c * psi_ho(n, x) * np.exp(-1j * energy_ho(n) * t / hbar)
    return psi

if "playing" not in st.session_state:
    st.session_state.playing = False
if "frame" not in st.session_state:
    st.session_state.frame = 0
    
# Build animation

# Assuming these are defined
# from your_module import psi_superposed, psi_n

def generate_animation_base64(L, ns, cs, speed=0.001, interval=100):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import tempfile, os, base64

    def psi_n(n, x, L):
        return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

    def energy_n(n, L):
        return (n ** 2) * (np.pi ** 2) / (2 * L ** 2)  # assume ħ = 1, m = 1

    def psi_superposed(x, t, L, ns, cs):
        total = np.zeros_like(x, dtype=complex)
        for n, c in zip(ns, cs):
            E_n = energy_n(n, L)
            total += c * psi_n(n, x, L) * np.exp(-1j * E_n * t )
        return total

    x = np.linspace(0, L, 1000)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Left: Re(Psi) ---
    line_total_re, = ax1.plot([], [], color='red', lw=2, label=r'$Re[\Psi(x,t)]$')
    line_individuals = [ax1.plot([], [], '--', lw=1.2, label=fr'$Re[\Psi_{{{n}}}(x,t)]$')[0] for n in ns]
    ax1.set_xlim(0, L)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Real part of $\Psi$')
    ax1.legend()

# --- Right: |Psi|² ---
    line_prob, = ax2.plot([], [], color='red', label=r'$|\Psi(x,t)|^2$')
    line_individuals_prob = [ax2.plot([], [], '--', label=fr'$|\Psi_{{{n}}}|^2$')[0] for n in ns]
    ax2.set_xlim(0, L)
    ax2.set_ylim(0, 6)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Probability Density $|\Psi|^2$')
    ax2.legend()

    def init():
        line_total_re.set_data([], [])
        line_prob.set_data([], [])
        for line in line_individuals:
            line.set_data([], [])
        for line in line_individuals_prob:
            line.set_data([], [])
            
        return [line_total_re, line_prob] + line_individuals + line_individuals_prob

    def update(frame):
        t = frame * speed
        psi = psi_superposed(x, t, L, ns, cs)
        line_total_re.set_data(x, np.real(psi))
        line_prob.set_data(x, np.abs(psi)**2)
        
        for line, n, c in zip(line_individuals, ns, cs):
            psi_n_val =  psi_n(n, x, L)* np.exp(-1j * energy_n(n, L) * t )
            line.set_data(x, np.real(psi_n_val))
            
        for line, n, c in zip(line_individuals_prob, ns, cs):
            psi_n_val =  psi_n(n, x, L)* np.exp(-1j * energy_n(n, L) * t )
            line.set_data(x, (np.abs( psi_n_val) ** 2))

        return [line_total_re, line_prob] + line_individuals + line_individuals_prob

    ani = FuncAnimation(fig, update, frames=500, init_func=init, blit=True, interval=interval)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer=PillowWriter(fps=30))
    plt.close(fig)

    with open(tmpfile.name, "rb") as f:
        data_url = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmpfile.name)

    return f'<img src="data:image/gif;base64,{data_url}" alt="quantum animation">'

#Animation for Harmonic oscillator

def generate_animation_base64_ho(ns, cs, speed=0.05, interval=100, hbar=1.0, omega=1.0):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    import tempfile, os, base64
    from scipy.special import hermite, factorial

    # --- Harmonic oscillator eigenfunction ---
    def psi_ho(n, x):
        Hn = hermite(n)(x)
        norm = 1.0 / np.sqrt((2.0**n) * factorial(n) * np.sqrt(np.pi))
        psi = norm * np.exp(-x**2 / 2) * Hn
        return psi

    # --- Energy levels ---
    def energy_ho(n):
        return (n + 0.5) * hbar * omega

    # --- Superposition ---
    def psi_superposed_ho(x, t, ns, cs):
        psi = np.zeros_like(x, dtype=complex)
        for n, c in zip(ns, cs):
            psi += c * psi_ho(n, x) * np.exp(-1j * energy_ho(n) * t / hbar)
        return psi

    x = np.linspace(-5, 5, 800)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: Re(Psi) ---
    line_total_re, = ax1.plot([], [], color='red', lw=2, label=r'$Re[\Psi(x,t)]$')
    line_individuals = [ax1.plot([], [], '--', lw=1.2, label=fr'$Re[\psi_{{{n}}}(x)]$')[0] for n in ns]
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(-1, 1)  # adjust later if needed
    ax1.set_xlabel('x')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Real part of $\Psi(x,t)$')
    ax1.legend()

    # --- Right: |Psi|² ---
    line_prob, = ax2.plot([], [], color='red', lw=2, label=r'$|\Psi(x,t)|^2$')
    line_individuals_prob = [ax2.plot([], [], '--', lw=1.2, label=fr'$|\psi_{{{n}}}(x)|^2$')[0] for n in ns]
    ax2.set_xlim(x.min(), x.max())
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Probability Density $|\Psi(x,t)|^2$')
    ax2.legend()

    # --- Init ---
    def init():
        line_total_re.set_data([], [])
        line_prob.set_data([], [])
        for line in line_individuals:
            line.set_data([], [])
        for line in line_individuals_prob:
            line.set_data([], [])
        return [line_total_re, line_prob] + line_individuals + line_individuals_prob

    # --- Update ---
    def update(frame):
        t = frame * speed
        psi = psi_superposed_ho(x, t, ns, cs)

        # total
        line_total_re.set_data(x, np.real(psi))
        line_prob.set_data(x, np.abs(psi) ** 2)

        # eigenfunctions
        for line, n in zip(line_individuals, ns):
            line.set_data(x, np.real(psi_ho(n, x)* np.exp(-1j * energy_ho(n) * t / hbar)))
        for line, n in zip(line_individuals_prob, ns):
            line.set_data(x, np.abs(psi_ho(n, x))**2)

        return [line_total_re, line_prob] + line_individuals + line_individuals_prob

    ani = FuncAnimation(fig, update, frames=500, init_func=init, blit=True, interval=interval)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer=PillowWriter(fps=30))
    plt.close(fig)

    with open(tmpfile.name, "rb") as f:
        data_url = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmpfile.name)

    return f'<img src="data:image/gif;base64,{data_url}" alt="quantum animation">'


# Streamlit UI
st.title("Simulation of the Schrödinger equation to demonstrate quantum superposition and state collapse")

n_vals = []
c_vals = []
st.sidebar.header("Select the Potential")
potential = st.sidebar.selectbox("Potential", ("Infinite Square Well", "Harmonic Oscillator"))
st.sidebar.header("Quantum States and Coefficients")
# Input number of basis states
st.sidebar.markdown("### Number of eigen states in the superposition")
N = st.sidebar.number_input("N", min_value=1, max_value=10, value=1, step=1, key="num_states")

# Input fields for each state
for i in range(N):
    st.sidebar.markdown(f"#### State {i + 1}")
    col1, col2, col3 = st.sidebar.columns(3)
    if potential == "Infinite Square Well":
        n = col1.number_input(f"$n_{{{i+1}}}$", min_value=1, value=i + 1, step=1, key=f"n{i}")
    else:
        n = col1.number_input(f"$n_{{{i+1}}}$", min_value=0, value=i + 1, step=1, key=f"n{i}")
    c_real = col2.number_input(f"Re($c_{i+1}$)", value=1.0 if i == 0 else 0.0, step=0.1, format="%.3f", key=f"re{i}")
    c_imag = col3.number_input(f"Im($c_{i+1}$)", value=0.0, step=0.1, format="%.3f", key=f"im{i}")

    n_vals.append(n)
    c_vals.append(complex(c_real, c_imag))


# Show normalization constant input
st.sidebar.markdown("### Normalization Factor")
#row1 = st.sidebar.rows(1)
norm_factor = st.sidebar.number_input(f"norm_factor", value=1.0,format="%.3f" )
if potential == "Infinite Square Well": 
    st.sidebar.markdown("### Length of Box")
    L= st.sidebar.number_input(f"L",min_value=0.5,max_value=10.0,value=1.0,step=0.5 )#1  # Or get this from user input if needed

# Normalize the coefficients
c_vals = [c / norm_factor for c in c_vals]

# Check normalization: sum of |c|^2
norm_check = np.sum(np.abs(c)**2 for c in c_vals)

if abs(np.sqrt(norm_check) - 1.0) < 1e-3:
    st.success(f"✅ Normalized: $\sum |c_i|^2 = {norm_check:.2f}$")
else:
    st.error(f"❌ Not normalized: $\sum |c_i|^2 = {norm_check:.3f}. Please adjust.")
#st.info("Amplitudes auto-normalized to ensure c₁² + c₂² = 1.")
if "running" not in st.session_state:
    st.session_state["running"] = False
if "snapshot_requested" not in st.session_state:
    st.session_state["snapshot_requested"] = False
# Start and stop buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("▶️ Start Animation"):
        st.session_state["running"] = True
with col2:
    if st.button("⏹️ Stop Animation"):
        st.session_state["running"] = False
        
if potential == "Infinite Square Well":
    if abs(np.sqrt(norm_check)- 1.0) < 1e-3:
        if st.session_state["running"]:
    # Show animation
            animation_html = generate_animation_base64(L, n_vals, c_vals)
            st.markdown(animation_html, unsafe_allow_html=True)

# Either manually capture snapshot or always show one when not running
        frame_number = st.slider("Select frame number for snapshot", min_value=0, max_value=300, value=100)
        dt = 0.009
        t_snapshot = frame_number * dt

# Snapshot plot
        if not st.session_state["running"] or st.session_state["snapshot_requested"]:
            x = np.linspace(0, L, 500)
            psi = psi_superposed(x, t_snapshot, L, n_vals, c_vals)
            prob_density = np.abs(psi)**2

#        fig, ax = plt.subplots(figsize=(8, 4))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(x, np.real(psi), 'r-', label=r'$\text{Re}[\Psi(x,t)]$')

    # Dynamically plot each component probability density
            colors = ['b--', 'g--', 'y--', 'm--', 'c--']
            for i, (n, c) in enumerate(zip(n_vals, c_vals)):
                component_density = np.real( psi_n(n, x, L)* np.exp(-1j * energy_n(n, L) * t_snapshot ))
                ax1.plot(x, component_density, colors[i % len(colors)], label=fr'$\Psi_{{{n}}}(x)$')

            ax1.legend()
            ax1.set_title(f"Snapshot at frame {frame_number}, time t = {t_snapshot:.3f} s")
        
            ax2.plot(x, prob_density, 'r-', label=r'$|\Psi(x,t)|^2$')

    # Dynamically plot each component probability density
            colors = ['b--', 'g--', 'y--', 'm--', 'c--']
            for i, (n, c) in enumerate(zip(n_vals, c_vals)):
                component_density = np.abs( psi_n(n, x, L)) ** 2
                ax2.plot(x, component_density, colors[i % len(colors)], label=fr'$|\Psi_{{{n}}}(x)|^2$')

            ax2.legend()
            ax2.set_title(f"Snapshot at frame {frame_number}, time t = {t_snapshot:.3f} s")
    
            st.pyplot(fig)

            st.session_state["snapshot_requested"] = False


# Analytical Expression
    st.markdown("### Normalized eigenfunctions of particle in 1D Box")

    st.markdown(r"$$\Psi_n(x) = \sqrt{\frac{2}{L}} \sin\left(\frac{n\pi x}{L}\right)$$", unsafe_allow_html=True)

    
    st.markdown("### Normalized Superposed State in 1D Box")
    expr = " + ".join([
    f"({np.round(c.real, 3)}{f'{np.round(c.imag, 3):+}i' if c.imag else ''})\\Psi_{{{n}}}(x)e^{{-iE_{{{n}}}t/\\hbar}}"
        for n, c in zip(n_vals, c_vals)
])
    st.markdown(f"$$\\Psi(x, t) = {expr}$$", unsafe_allow_html=True)


# Energy Measurement / Collapse
    if abs(np.sqrt(norm_check)- 1.0) < 1e-3:
        if st.button("🔘️Measure Energy "):
            with st.spinner("Collapsing..."):
                probs = np.abs(np.array(c_vals))**2
                probs = probs / np.sum(probs)  # normalize

                outcome = np.random.choice(n_vals, p=probs) 
                outcome_index = n_vals.index(outcome)

# Get the probability of collapsing to that state
                collapse_prob = probs[outcome_index]

# Set the plot title
                x = np.linspace(0, L, 500)
                collapsed_state = psi_n(outcome, x, L)
                probability = np.abs(collapsed_state) ** 2

#            fig, ax = plt.subplots(figsize=(8, 5))
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.plot(x, collapsed_state, label=rf'$\text{{Re}}(\Psi_{{{outcome}}}(x))$', color='red')
                ax1.set_xlim(0, L)
                ax1.set_ylim(np.min(collapsed_state)*1.2,np.max(collapsed_state)*1.2)#0, np.max(probability)*1.2)
                ax1.set_title(fr'Wavefunction collapsed to $\Psi_{{{outcome}}}$ and energy $E_{{{outcome}}}$')
                ax1.set_xlabel('x')
                ax1.set_ylabel('Amplitude')
                ax1.legend()
            
                ax2.plot(x, probability, label=rf'$|\Psi_{{{outcome}}}|^2 $', color='red')
                ax2.set_xlim(0, L)
                ax2.set_ylim(0, np.max(probability)*1.2)
#            ax2.set_title(fr'Wavefunction collapsed to $\Psi_{{{outcome}}}$ with probability {collapse_prob:.3f}')
                ax2.set_xlabel('x')
                ax2.set_ylabel('Probability Density')
                ax2.legend()
                st.pyplot(fig)
            
# Position Measurement / Collapse
    if abs(np.sqrt(norm_check) - 1.0) < 1e-3:
        if st.button("🔘️Measure Position"):
            with st.spinner("Collapsing..."):
                x = np.linspace(0, L, 500)
                t = 0.009  # Time of measurement

            # Evaluate probability density from superposed state
                psi_vals = psi_superposed(x, t, L, n_vals, c_vals)
                prob_density = np.abs(psi_vals)**2
                prob_density /= np.trapz(prob_density, x)  # Normalize

            # Sample measurement outcome x0
                x0 = np.random.choice(x, p=prob_density/np.sum(prob_density))
                sigma = 0.005 * L

            # Collapse: Gaussian centered at x0
                collapsed_state = np.exp(-(x - x0)**2 / (2 * sigma**2))
                collapsed_state /= np.sqrt(np.trapz(np.abs(collapsed_state)**2, x))

            # Plot collapsed wavefunction
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x, collapsed_state, label=fr'$\Psi(x)$ after measurement', color='red')
                ax.set_xlim(0, L)
                ax.set_ylim(np.min(collapsed_state)*1.2, np.max(collapsed_state)*1.2)
                ax.set_title(fr'Wavefunction collapsed at $x = {x0:.2f}$')
                ax.set_xlabel('x')
                ax.set_ylabel('Amplitude')
                ax.legend()
                st.pyplot(fig)
elif potential == "Harmonic Oscillator":
    if abs(np.sqrt(norm_check)- 1.0) < 1e-3:
        if st.session_state["running"]:
    # Show animation
            animation_html = generate_animation_base64_ho(n_vals, c_vals)
            st.markdown(animation_html, unsafe_allow_html=True)

# Either manually capture snapshot or always show one when not running
        frame_number = st.slider("Select frame number for snapshot", min_value=0, max_value=300, value=100)
        dt = 0.009
        t_snapshot = frame_number * dt

# Snapshot plot
        if not st.session_state["running"] or st.session_state["snapshot_requested"]:
            x = np.linspace(-5, 5, 500)
            psi = psi_superposed_ho(x, t_snapshot, n_vals, c_vals)
            prob_density = np.abs(psi)**2

#        fig, ax = plt.subplots(figsize=(8, 4))
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.plot(x, np.real(psi), 'r-', label=r'$\text{Re}[\Psi(x,t)]$')

    # Dynamically plot each component probability density
            colors = ['b--', 'g--', 'y--', 'm--', 'c--']
            for i, (n, c) in enumerate(zip(n_vals, c_vals)):
                component_density = np.real( psi_ho(n, x)* np.exp(-1j * energy_ho(n) * t_snapshot ))
                ax1.plot(x, component_density, colors[i % len(colors)], label=fr'$\Psi_{{{n}}}(x)$')

            ax1.legend()
            ax1.set_title(f"Snapshot at frame {frame_number}, time t = {t_snapshot:.3f} s")
        
            ax2.plot(x, prob_density, 'r-', label=r'$|\Psi(x,t)|^2$')

    # Dynamically plot each component probability density
            colors = ['b--', 'g--', 'y--', 'm--', 'c--']
            for i, (n, c) in enumerate(zip(n_vals, c_vals)):
                component_density = np.abs( psi_ho(n, x)) ** 2
                ax2.plot(x, component_density, colors[i % len(colors)], label=fr'$|\Psi_{{{n}}}(x)|^2$')

            ax2.legend()
            ax2.set_title(f"Snapshot at frame {frame_number}, time t = {t_snapshot:.3f} s")
    
            st.pyplot(fig)

            st.session_state["snapshot_requested"] = False


# Analytical Expression
    st.markdown("### Normalized harmonic oscillator eigenfunctions")

    st.markdown(r"$$\Psi_n(x) = \sqrt{\frac{\alpha}{\sqrt{\pi}2^n n!}} e^{-\alpha^2 x^2/2}H_n(\alpha x)$$", unsafe_allow_html=True)
    
    st.markdown("### Normalized Superposed State of Harmonic Oscillator")
    expr = " + ".join([
    f"({np.round(c.real, 3)}{f'{np.round(c.imag, 3):+}i' if c.imag else ''})\\Psi_{{{n}}}(x)e^{{-iE_{{{n}}}t/\\hbar}}"
        for n, c in zip(n_vals, c_vals)
])
    st.markdown(f"$$\\Psi(x, t) = {expr}$$", unsafe_allow_html=True)


# Energy Measurement / Collapse
    if abs(np.sqrt(norm_check)- 1.0) < 1e-3:
        if st.button("🔘️Measure Energy "):
            with st.spinner("Collapsing..."):
                probs = np.abs(np.array(c_vals))**2
                probs = probs / np.sum(probs)  # normalize

                outcome = np.random.choice(n_vals, p=probs) 
                outcome_index = n_vals.index(outcome)

# Get the probability of collapsing to that state
                collapse_prob = probs[outcome_index]

# Set the plot title
                x = np.linspace(-4, 4, 500)
                collapsed_state = psi_ho(outcome, x)
                probability = np.abs(collapsed_state) ** 2

#            fig, ax = plt.subplots(figsize=(8, 5))
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.plot(x, collapsed_state, label=rf'$\text{{Re}}(\Psi_{{{outcome}}}(x))$', color='red')
                ax1.set_xlim(-4,4)
                ax1.set_ylim(np.min(collapsed_state)*1.2,np.max(collapsed_state)*1.2)#0, np.max(probability)*1.2)
                ax1.set_title(fr'Wavefunction collapsed to $\Psi_{{{outcome}}}$ and energy $E_{{{outcome}}}$')
                ax1.set_xlabel('x')
                ax1.set_ylabel('Amplitude')
                ax1.legend()
            
                ax2.plot(x, probability, label=rf'$|\Psi_{{{outcome}}}|^2 $', color='red')
                ax2.set_xlim(-4, 4)
                ax2.set_ylim(0, np.max(probability)*1.2)
#            ax2.set_title(fr'Wavefunction collapsed to $\Psi_{{{outcome}}}$ with probability {collapse_prob:.3f}')
                ax2.set_xlabel('x')
                ax2.set_ylabel('Probability Density')
                ax2.legend()
                st.pyplot(fig)
            
# Position Measurement / Collapse
    if abs(np.sqrt(norm_check) - 1.0) < 1e-3:
        if st.button("🔘️Measure Position"):
            with st.spinner("Collapsing..."):
                x = np.linspace(-4, 4, 500)
                t = 0.009  # Time of measurement

            # Evaluate probability density from superposed state
                psi_vals = psi_superposed_ho(x, t, n_vals, c_vals)
                prob_density = np.abs(psi_vals)**2
                prob_density /= np.trapz(prob_density, x)  # Normalize

            # Sample measurement outcome x0
                x0 = np.random.choice(x, p=prob_density/np.sum(prob_density))
                sigma = 0.02 

            # Collapse: Gaussian centered at x0
                collapsed_state = np.exp(-(x - x0)**2 / (2 * sigma**2))
                collapsed_state /= np.sqrt(np.trapz(np.abs(collapsed_state)**2, x))

            # Plot collapsed wavefunction
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(x, collapsed_state, label=fr'$\Psi(x)$ after measurement', color='red')
                ax.set_xlim(-4, 4)
                ax.set_ylim(np.min(collapsed_state)*1.2, np.max(collapsed_state)*1.2)
                ax.set_title(fr'Wavefunction collapsed at $x = {x0:.2f}$')
                ax.set_xlabel('x')
                ax.set_ylabel('Amplitude')
                ax.legend()
                st.pyplot(fig)
        
