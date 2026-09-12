import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display

R = np.random.default_rng(7351)

def get_graph(n, p, s=127):
    while True:
        G = nx.erdos_renyi_graph(n, p, seed=s)
        if nx.is_connected(G):
            return G
        s += 1

def segment(a, b, k):
    t = np.linspace(0, 1, k, endpoint=False)
    return np.asarray(a) + t[:, None] * (np.asarray(b) - np.asarray(a))

def build(strokes):
    return np.vstack([segment(a, b, k) for a, b, k in strokes])

# 20 target points for every letter
SHAPES = {
    "L": [
        ((0,2),(0,0),10), ((0,0),(1,0),10)
    ],
    "O": [
        ((0,2),(1,2),5), ((1,2),(1,0),5),
        ((1,0),(0,0),5), ((0,0),(0,2),5)
    ],
    "K": [
        ((0,0),(0,2),8), ((0,1),(1,2),6),
        ((0,1),(1,0),6)
    ],
    "E": [
        ((0,2),(1,2),5), ((0,2),(0,0),5),
        ((0,1),(.8,1),5), ((0,0),(1,0),5)
    ],
    "S": [
        ((1,2),(0,2),4), ((0,2),(0,1),4),
        ((0,1),(1,1),4), ((1,1),(1,0),4),
        ((1,0),(0,0),4)
    ],
    "H": [
        ((0,0),(0,2),7), ((1,0),(1,2),7),
        ((0,1),(1,1),6)
    ]
}

def simulate():
    n = 20
    p = 0.22
    dt = 0.055
    gain = 0.35
    iterations = 90

    G = get_graph(n, p)

    A = nx.to_numpy_array(G)
    L = np.diag(A.sum(1)) - A

    goals = {c: build(v) for c, v in SHAPES.items()}

    X = R.uniform(-1.5, 1.5, (n, 2))
    history = []

    for c in "LOKESH":
        target = goals[c]
        target -= target.mean(axis=0)

        for _ in range(iterations):
            U = -L @ (X - target) + gain * (target - X)
            X += dt * U
            history.append((c, X.copy()))

    # Communication graph
    plt.figure(figsize=(7, 5))
    pos = nx.spring_layout(G, seed=92)

    nx.draw(
        G, pos,
        with_labels=True,
        node_color="black",
        edge_color="grey",
        node_size=450,
        font_color="white"
    )

    plt.title("Connected Communication Graph")
    plt.savefig(
        "lokesh_graph.png",
        dpi=200,
        bbox_inches="tight"
    )
    plt.show()

    # Animation
    fig, ax = plt.subplots(figsize=(6, 6))

    dots = ax.scatter(
        [], [], s=65,
        color="black",
        zorder=2
    )

    links = [
        ax.plot(
            [], [],
            color="lightblue",
            linewidth=0.7,
            zorder=1
        )[0]
        for _ in G.edges()
    ]

    heading = ax.set_title("")

    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.5, 1.7)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")

    def animate(i):
        char, pts = history[i]

        dots.set_offsets(pts)
        heading.set_text(f"Formation Control: {char}")

        for line, (u, v) in zip(links, G.edges()):
            line.set_data(
                pts[[u, v], 0],
                pts[[u, v], 1]
            )

        return [dots, heading, *links]

    ani = FuncAnimation(
        fig,
        animate,
        frames=range(0, len(history), 3),
        interval=50,
        blit=False
    )

    filename = "lokesh_animation.gif"

    ani.save(
        filename,
        writer=PillowWriter(fps=20)
    )

    plt.close(fig)

    print("GIF saved at:")
    import os
    print(os.path.abspath(filename))

    display(Image(filename=filename))


simulate()
