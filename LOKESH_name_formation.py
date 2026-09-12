import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display


rng = np.random.default_rng(11004)


# GENERATE A CONNECTED ERDOS-RENYI GRAPH
def connected_graph(n, p, seed=41):
    """
    Generate an Erdos-Renyi graph repeatedly until
    a connected graph is obtained.
    """

    current_seed = seed

    while True:
        graph = nx.erdos_renyi_graph(n,p,seed=current_seed)
        if nx.is_connected(graph):
            return graph
        current_seed += 1


# GENERATE POINTS ALONG A LINE
def line_points(start, end, number):
    """
    Return equally spaced 2-D points along a line segment.
    """

    t = np.linspace(0.0,1.0,number,endpoint=False)
    start = np.asarray(start,dtype=float)
    end = np.asarray(end,dtype=float)
    return start + t[:, None] * (end - start)

# CREATE LETTER FROM STROKES
def make_letter(strokes, number_of_agents=20):
    """
    Convert line-segment strokes into exactly
    number_of_agents target positions.
    """

    points = np.vstack(
        [
            line_points(start, end, count)
            for start, end, count in strokes
        ]
    )

    return points


# LETTER STROKES FOR "LOKESH"
LETTER_STROKES = {

  "L": [
        ((0, 2), (0, 0), 10),
        ((0, 0), (1, 0), 10),
    ],


    "O": [
        ((0, 2), (1, 2), 5),
        ((1, 2), (1, 0), 5),
        ((1, 0), (0, 0), 5),
        ((0, 0), (0, 2), 5),
    ],


    "K": [
        ((0, 0), (0, 2), 8),
        ((0, 1), (1, 2), 6),
        ((0, 1), (1, 0), 6),
    ],

    "E": [
        ((0, 2), (1, 2), 5),
        ((0, 2), (0, 0), 5),
        ((0, 1), (0.8, 1), 5),
        ((0, 0), (1, 0), 5),
    ],

    "S": [
        ((1, 2), (0, 2), 4),
        ((0, 2), (0, 1), 4),
        ((0, 1), (1, 1), 4),
        ((1, 1), (1, 0), 4),
        ((1, 0), (0, 0), 4),
    ],

    "H": [
        ((0, 0), (0, 2), 7),
        ((1, 0), (1, 2), 7),
        ((0, 1), (1, 1), 6),
    ],
}


# MAIN SIMULATION
def run_simulation():

    # PARAMETERS
    number_of_agents = 20

    # Probability of an edge between two agents
    edge_probability = 0.22

    # Simulation time step
    time_step = 0.055

    # Attraction / anchoring gain
    anchoring_gain = 0.35

    # Number of iterations spent forming each letter
    steps_per_letter = 90


    # CREATE CONNECTED GRAPH
    graph = connected_graph(
        number_of_agents,
        edge_probability,
        seed=41
    )


    # ADJACENCY MATRIX
    adjacency = nx.to_numpy_array(
        graph,
        nodelist=range(number_of_agents)
    )

    # DEGREE MATRIX
    degree_matrix = np.diag(
        adjacency.sum(axis=1)
    )


    # GRAPH LAPLACIAN L = D - A
    laplacian = degree_matrix - adjacency

  # CREATE TARGET POSITIONS FOR EACH LETTER
    targets = {}

    for letter, strokes in LETTER_STROKES.items():

        targets[letter] = make_letter(strokes,number_of_agents)


    # RANDOM INITIAL POSITIONS
    positions = rng.uniform(
        -1.5,
        1.5,
        size=(number_of_agents, 2)
    )

    # STORE FRAMES FOR ANIMATION
    frames = []
  
    # FORM "LOKESH" LETTER BY LETTER
    for letter in "LOKESH":

        # Desired positions for current letter
        desired = targets[letter].copy()

        # CENTER LETTER AROUND ORIGIN
        desired = desired - desired.mean(axis=0)

        # FORMATION CONTROL
        for _ in range(steps_per_letter):

            # Consensus / formation control  u = -L(x-r)
            control = -laplacian @ (positions - desired)

            # Attraction toward desired positions gamma(r-x)
            control += anchoring_gain*(desired - positions)

            # Update positions
            positions = (positions + time_step * control)

            # Save current positions
            frames.append((letter,positions.copy()))

    # DISPLAY COMMUNICATION GRAPH
    plt.figure(figsize=(7, 5))

    graph_layout = nx.spring_layout(graph,seed=4)
    
    nx.draw(graph,graph_layout,with_labels=True,node_color="skyblue",edge_color="gray",node_size=450)
    plt.title("Connected Erdos-Renyi Graph: N=20, p=0.22")


    # Do NOT use tight_layout()
    # It can produce warnings with NetworkX axes.


    plt.savefig("lokesh_communication_graph.png",dpi=200, bbox_inches="tight")


    plt.show()


    # CREATE ANIMATION FIGURE
    figure, axis = plt.subplots(figsize=(6, 6))

    # AGENTS
    scatter = axis.scatter([],[],s=65,color="royalblue",zorder=2)

    # COMMUNICATION GRAPH EDGES
    edge_lines = []
    for _ in graph.edges():
        line = axis.plot([],[],color="lightgray",linewidth=0.7, zorder=1)[0]
        edge_lines.append(line)


    # GRAPH SETTINGS
    title = axis.set_title("")
    axis.set_xlim(-1.8,1.8)
    axis.set_ylim(-1.5,1.7)
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.set_xlabel("x position")
    axis.set_ylabel("y position")

  # UPDATE FUNCTION FOR ANIMATION
    def update(frame_index):
        # Get letter and current agent positions
        letter, current_positions = frames[frame_index]
        # Update agent positions
        scatter.set_offsets(current_positions)
        # Update title
        title.set_text(f"Formation Control: {letter}")
        # Update communication edges
        for line, (i, j) in zip(edge_lines,graph.edges()):
            line.set_data(current_positions[[i, j], 0],current_positions[[i, j], 1])
        return [scatter,title,*edge_lines]

    # CREATE ANIMATION
    animation = FuncAnimation(figure,update,frames=range(0,len(frames),3),interval=50,blit=False)
    # SAVE AS GIF
    gif_name = "lokesh_formation.gif"
    animation.save(gif_name,writer=PillowWriter(fps=20))

    # Close matplotlib figure
    plt.close(figure)
    # DISPLAY GIF IN JUPYTER NOTEBOOK

    display(Image(filename=gif_name))

# RUN THE COMPLETE SIMULATION
run_simulation()
