import streamlit as st
import numpy as np
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Marine Sensor Network Simulator+", layout="wide")

st.title("🌊 Advanced Marine Sensor Network Simulator")
st.caption("FI9065 — Sensor Networks for Marine Applications (Upgraded Version)")

st.sidebar.header("Simulation Controls")

num_nodes = st.sidebar.slider("Number of sensor nodes", 4, 20, 10)
area_size = st.sidebar.slider("Ocean area size (meters)", 200, 3000, 1200)

network_type = st.sidebar.selectbox(
    "Communication Type",
    ["RF Telemetry", "Cabled / Fiber", "Acoustic"]
)

enable_cluster = st.sidebar.checkbox("Enable Clustering (energy efficient)")
enable_sofar = False
if network_type == "Acoustic":
    enable_sofar = st.sidebar.checkbox("Enable SOFAR Channel Effect (long range)")

np.random.seed(3)
nodes = np.random.randint(50, area_size-50, (num_nodes, 2))
base_station = np.array([area_size//2, 20])

def distance(a, b):
    return math.dist(a, b)

# ---------------------------
# CLUSTERING (manual algorithm — no ML libs)
# ---------------------------
clusters = {}
cluster_centers = []

if enable_cluster:
    k = max(2, num_nodes // 4)

    # choose first k nodes as centers
    cluster_centers = nodes[:k]

    for i in range(k):
        clusters[i] = []

    for node in nodes:
        dists = [distance(node, c) for c in cluster_centers]
        cid = np.argmin(dists)
        clusters[cid].append(node)

# ---------------------------
# ROUTING: nearest neighbor (tree)
# ---------------------------
def build_route(nodes, start):
    remaining = nodes.tolist()
    route = [start.tolist()]
    current = start

    while remaining:
        nearest = min(remaining, key=lambda p: distance(current, p))
        route.append(nearest)
        remaining.remove(nearest)
        current = np.array(nearest)

    return np.array(route)

route = build_route(nodes, base_station)

# ---------------------------
# Communication model
# ---------------------------
results = []
total_energy = 0
total_delay = 0
sync_overhead = 0

for node in nodes:
    d = distance(node, base_station)

    if network_type == "RF Telemetry":
        speed = 3e8
        energy = d * 0.06
        sync = 0.00002

    elif network_type == "Cabled / Fiber":
        speed = 2e8
        energy = d * 0.015
        sync = 0.00001

    else:
        speed = 1500
        energy = d * 0.12
        sync = 0.002

        if enable_sofar:
            d = d * 0.8

    delay = d / speed
    total_energy += energy
    total_delay += delay
    sync_overhead += sync

    results.append((round(d,2), round(delay,6), round(energy,3)))

# ---------------------------
# PLOT MAP
# ---------------------------
fig, ax = plt.subplots()

ax.scatter(nodes[:,0], nodes[:,1], label="Sensor Nodes")
ax.scatter(base_station[0], base_station[1], color="red", label="Base Station")

# route visualization
ax.plot(route[:,0], route[:,1], linestyle="--")

if enable_cluster:
    for i, c in enumerate(cluster_centers):
        ax.scatter(c[0], c[1], marker="x", s=120)
        for n in clusters[i]:
            ax.plot([c[0], n[0]], [c[1], n[1]], alpha=0.4)

ax.set_title("Underwater Network Layout")
ax.set_xlim(0, area_size)
ax.set_ylim(0, area_size)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Depth / Span (m)")
ax.legend()

st.pyplot(fig)

# ---------------------------
# TABLE
# ---------------------------
st.subheader("📊 Node Metrics")
st.write("| Node | Distance (m) | Delay (sec) | Energy Units |")
st.write("|------|--------------|-------------|--------------|")

for i, r in enumerate(results):
    st.write(f"| {i+1} | {r[0]} | {r[1]} | {r[2]} |")

st.success(f"Total Delay: {round(total_delay,6)} sec")
st.info(f"Total Energy: {round(total_energy,3)} units")
st.warning(f"Synchronization Overhead: {round(sync_overhead,6)} sec")

st.divider()

st.subheader("🧠 Interpretation")

if network_type == "RF Telemetry":
    st.write("""
- Very low delay  
- Large energy consumption underwater  
- Good only for surface & short marine zones
""")

elif network_type == "Cabled / Fiber":
    st.write("""
- Reliable and low-energy  
- High installation cost  
- Ideal for permanent observatories
""")

else:
    st.write("""
- Works deep underwater  
- Slow due to low sound speed  
- SOFAR channel improves long-range transmission
""")

if enable_cluster:
    st.info("Clustering reduces communication hops and saves energy.")

st.caption("Pure simulation. No datasets. No ML models. Built only from theory.")
