import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import Image, display
import os
class Formation:
    def __init__(self):
        self.agents=20
        self.prob=0.22
        self.dt=0.055
        self.k=0.35
        self.cycles=90
        self.random=np.random.default_rng(8246)
        self.shapes=self.define_shapes()
        self.graph=self.create_network()
        self.laplacian=self.get_laplacian()
    def define_shapes(self):
        return {
            "L":[((0,2),(0,0),10),((0,0),(1,0),10)],
            "O":[((0,2),(1,2),5),((1,2),(1,0),5),((1,0),(0,0),5),((0,0),(0,2),5)],
            "K":[((0,0),(0,2),8),((0,1),(1,2),6),((0,1),(1,0),6)],
            "E":[((0,2),(1,2),5),((0,2),(0,0),5),((0,1),(0.8,1),5),((0,0),(1,0),5)],
            "S":[((1,2),(0,2),4),((0,2),(0,1),4),((0,1),(1,1),4),((1,1),(1,0),4),((1,0),(0,0),4)],
            "H":[((0,0),(0,2),7),((1,0),(1,2),7),((0,1),(1,1),6)]
        }
    def create_network(self):
        seed=153
        while True:
            network=nx.erdos_renyi_graph(self.agents,self.prob,seed=seed)
            if nx.is_connected(network): return network
            seed+=1
    def get_laplacian(self):
        matrix=nx.to_numpy_array(self.graph)
        degree=np.diag(np.sum(matrix,axis=1))
        return degree-matrix
    @staticmethod
    def make_segment(start,finish,amount):
        values=np.linspace(0,1,amount,endpoint=False)
        start=np.array(start,dtype=float)
        finish=np.array(finish,dtype=float)
        return start+values[:,None]*(finish-start)
    def target_coordinates(self,strokes):
        pieces=[]
        for start,finish,amount in strokes: 
            pieces.append(self.make_segment(start,finish,amount))
        return np.vstack(pieces)
    def prepare_targets(self):
        targets={}
        for character in self.shapes:
            points=self.target_coordinates(self.shapes[character])
            points-=points.mean(axis=0)
            targets[character]=points
        return targets
    def run(self):
        targets=self.prepare_targets()
        current=self.random.uniform(-1.5,1.5,size=(self.agents,2))
        record=[]
        for character in "LOKESH":
            destination=targets[character]
            for step in range(self.cycles):
                difference=current-destination
                velocity=-self.laplacian@difference+self.k*(destination-current)
                current=current+self.dt*velocity
                record.append((character,current.copy()))
        return record
    def show_network(self):
        plt.figure(figsize=(7,5))
        layout=nx.spring_layout(self.graph,seed=318)
        nx.draw(self.graph,layout,with_labels=True,node_color="black",edge_color="skyblue",node_size=450,font_color="white")
        plt.title("Connected Erdos-Renyi Network")
        plt.savefig("lokesh_graph.png",dpi=200,bbox_inches="tight")
        plt.show()
    def animate(self,data):
        fig,ax=plt.subplots(figsize=(6,6))
        points=ax.scatter([],[],s=65,color="black")
        connections=[]
        for _ in self.graph.edges():
            connections.append(ax.plot([],[],color="skyblue",linewidth=0.7)[0])
        heading=ax.set_title("")
        ax.set_xlim(-1.8,1.8)
        ax.set_ylim(-1.5,1.7)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("x position")
        ax.set_ylabel("y position")
        def redraw(frame):
            character,location=data[frame]
            points.set_offsets(location)
            heading.set_text("Formation Control : "+character)
            for line,(a,b) in zip(connections,self.graph.edges()):
                line.set_data(location[[a,b],0],location[[a,b],1])
            return [points,heading,*connections]
        movie=FuncAnimation(fig,redraw,frames=range(0,len(data),3),interval=50,blit=False)
        filename="lokesh_animation.gif"
        movie.save(filename,writer=PillowWriter(fps=20))
        plt.close(fig)
        print("Animation saved to:")
        print(os.path.abspath(filename))
        display(Image(filename=filename))
model=Formation()
trajectory=model.run()
model.show_network()
model.animate(trajectory)
