import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import streamlit as st
import streamlit.components.v1 as components
import tempfile
import os
import base64

# Constants
hbar = 1.0
m = 1.0
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
    
if "playing" not in st.session_state:
    st.session_state.playing = False
if "frame" not in st.session_state:
    st.session_state.frame = 0
    
# Build animation

# from your_module import psi_superposed, psi_n

def generate_animation_base64(L, ns, cs, speed=0.009, interval=100):
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
            total += c * psi_n(n, x, L) * np.exp(-1j * E_n * t + 1j)
        return total

    x = np.linspace(0, L, 1000)
    fig, ax = plt.subplots(figsize=(8, 5))
    line_total, = ax.plot([], [], '-', color='red', label=r'$|\Psi_{\text{total}}|^2$')
    line_individuals = [ax.plot([], [], '--', label=fr'$|\Psi_{{{n}}}|^2$')[0] for n in ns]

    ax.set_xlim(0, L)
    ax.set_ylim(0, 6)
    ax.set_title('Quantum Particle in 1D Box')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density')
    ax.legend()

    def init():
        line_total.set_data([], [])
        for line in line_individuals:
            line.set_data([], [])
        return [line_total] + line_individuals

    def update(frame):
        t = frame * speed
        psi = psi_superposed(x, t, L, ns, cs)
        prob_density = np.abs(psi) ** 2
        line_total.set_data(x, prob_density)

        for line, n, c in zip(line_individuals, ns, cs):
            line.set_data(x, (np.abs(c * psi_n(n, x, L)) ** 2))

        return [line_total] + line_individuals

    ani = FuncAnimation(fig, update, frames=100, interval=interval, init_func=init, blit=True)

    tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.gif')
    ani.save(tmpfile.name, writer=PillowWriter(fps=1000 // interval))
    plt.close(fig)

    with open(tmpfile.name, "rb") as f:
        data_url = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmpfile.name)

    return f'<img src="data:image/gif;base64,{data_url}" alt="quantum animation">'



# Streamlit UI
st.title("Quantum Superposition in a 1D Box")


n_vals = []
c_vals = []

st.sidebar.header("Quantum States and Coefficients")

# Input number of basis states
st.sidebar.markdown("### Number of energy eigenstates in the superposition")
N = st.sidebar.number_input("N", min_value=1, max_value=10, value=3, step=1, key="num_states")

# Input fields for each state
for i in range(N):
    st.sidebar.markdown(f"#### State {i + 1}")
    col1, col2, col3 = st.sidebar.columns(3)

    n = col1.number_input(f"$n_{{{i+1}}}$", min_value=1, value=i + 1, step=1, key=f"n{i}")
    c_real = col2.number_input(f"Re($c_{i+1}$)", value=1.0 if i == 0 else 0.0, step=0.1, format="%.3f", key=f"re{i}")
    c_imag = col3.number_input(f"Im($c_{i+1}$)", value=0.0, step=0.1, format="%.3f", key=f"im{i}")

    n_vals.append(n)
    c_vals.append(complex(c_real, c_imag))



# Show normalization constant input
st.sidebar.markdown("### Normalization Factor")
#row1 = st.sidebar.rows(1)
norm_factor = st.sidebar.number_input(f"norm_factor", value=1.0,format="%.3f" )
st.sidebar.markdown("### Length of Box")
L= st.sidebar.number_input(f"L",value=1,step=1 )#1  # Or get this from user input if needed

# Normalize the coefficients
c_vals = [c / norm_factor for c in c_vals]

# Check normalization: sum of |c|^2
norm_check = np.sum(np.abs(c)**2 for c in c_vals)

if abs(np.sqrt(norm_check) - 1.0) < 1e-3:
    st.success(f"✅ Normalized: $\sum |c_i|^2 = {norm_check:.3f}$")
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
#with col3:
#    if st.button("📸 Capture Snapshot"):
#        st.session_state["snapshot_requested"] = True
# Animation or snapshot
if abs(np.sqrt(norm_check)- 1.0) < 1e-3:
    if st.session_state["running"]:
    # Show animation
        animation_html = generate_animation_base64(L, n_vals, c_vals)
        st.markdown(animation_html, unsafe_allow_html=True)

# Either manually capture snapshot or always show one when not running
    frame_number = st.slider("Select frame number for snapshot", min_value=0, max_value=200, value=100)
    dt = 0.009
    t_snapshot = frame_number * dt

# Snapshot plot
    if not st.session_state["running"] or st.session_state["snapshot_requested"]:
        x = np.linspace(0, L, 1000)
        psi = psi_superposed(x, t_snapshot, L, n_vals, c_vals)
        prob_density = np.abs(psi)**2

        fig, ax = plt.subplots()
        ax.plot(x, prob_density, 'r-', label=r'$|\Psi_{\text{total}}|^2$')

    # Dynamically plot each component probability density
        colors = ['b--', 'g--', 'y--', 'm--', 'c--']
        for i, (n, c) in enumerate(zip(n_vals, c_vals)):
            component_density = np.abs(c * psi_n(n, x, L))**2
            ax.plot(x, component_density, colors[i % len(colors)], label=fr'$|\Psi_{{{n}}}|^2$')

        ax.legend()
        ax.set_title(f"Snapshot at frame {frame_number}, time t = {t_snapshot:.3f} s")
        st.pyplot(fig)

        st.session_state["snapshot_requested"] = False


# Analytical Expression
st.markdown("### Normalized Superposed State in 1D Box")
expr = " + ".join([
f"({np.round(c.real, 3)}{f'{np.round(c.imag, 3):+}i' if c.imag else ''})\\Psi_{{{n}}}(x)e^{{-iE_{{{n}}}t/\\hbar}}"
    for n, c in zip(n_vals, c_vals)
])
st.markdown(f"$$\\Psi(x, t) = {expr}$$", unsafe_allow_html=True)


# Measurement / Collapse
if abs(np.sqrt(norm_check)- 1.0) < 1e-3:
    if st.button("🔘️Make observation "):
        with st.spinner("Collapsing..."):
            probs = np.abs(np.array(c_vals))**2
            probs = probs / np.sum(probs)  # normalize

            outcome = np.random.choice(n_vals, p=probs)
            outcome_index = n_vals.index(outcome)

# Get the probability of collapsing to that state
            collapse_prob = probs[outcome_index]

# Set the plot title
            x = np.linspace(0, L, 1000)
            collapsed_state = psi_n(outcome, x, L)
            probability = np.abs(collapsed_state) ** 2

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, probability, label=rf'$|\Psi_{{{outcome}}}|^2 \text{{ collapsed}}$', color='red')
            ax.set_xlim(0, L)
            ax.set_ylim(0, np.max(probability)*1.2)
            ax.set_title(fr'Collapsed to $\Psi_{{{outcome}}}$ with probability {collapse_prob:.3f}')
            ax.set_xlabel('x')
            ax.set_ylabel('Probability Density')
            ax.legend()
            st.pyplot(fig)


