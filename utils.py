# -----------------------------------------------------------------------------
# rng related

# class that mimics the random interface in Python, fully deterministic,
# and in a way that we also control fully, and can also use in C, etc.
class RNG:
    def __init__(self, seed):
        self.state = seed

    def random_u32(self):
        # xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
        # doing & 0xFFFFFFFF is the same as cast to uint32 in C
        self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
        self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
        return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

    def random(self):
        # random float32 in [0, 1)
        return (self.random_u32() >> 8) / 16777216.0

    def uniform(self, a=0.0, b=1.0):
        # random float32 in [a, b)
        return a + (b-a) * self.random()

# -----------------------------------------------------------------------------
# data related

# Generates the Yin Yang dataset.
# Thank you https://github.com/lkriener/yin_yang_data_set
def gen_data_yinyang(random: RNG, n=1000, r_small=0.1, r_big=0.5):
    pts = []

    def dist_to_right_dot(x, y):
        return ((x - 1.5 * r_big)**2 + (y - r_big)**2) ** 0.5

    def dist_to_left_dot(x, y):
        return ((x - 0.5 * r_big)**2 + (y - r_big)**2) ** 0.5

    def which_class(x, y):
        d_right = dist_to_right_dot(x, y)
        d_left = dist_to_left_dot(x, y)
        criterion1 = d_right <= r_small
        criterion2 = d_left > r_small and d_left <= 0.5 * r_big
        criterion3 = y > r_big and d_right > 0.5 * r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < r_small or d_left < r_small

        if is_circles:
            return 2
        return 0 if is_yin else 1

    def get_sample(goal_class=None):
        while True:
            x = random.uniform(0, 2 * r_big)
            y = random.uniform(0, 2 * r_big)
            if ((x - r_big)**2 + (y - r_big)**2) ** 0.5 > r_big:
                continue
            c = which_class(x, y)
            if goal_class is None or c == goal_class:
                scaled_x = (x / r_big - 1) * 2
                scaled_y = (y / r_big - 1) * 2
                return [scaled_x, scaled_y, c]

    for i in range(n):
        goal_class = i % 3
        x, y, c = get_sample(goal_class)
        pts.append([[x, y], c])

    tr = pts[:int(0.8 * n)]
    val = pts[int(0.8 * n):int(0.9 * n)]
    te = pts[int(0.9 * n):]
    return tr, val, te

# -----------------------------------------------------------------------------
# visualization related

def vis_color(nodes, color):
    # colors a set of nodes (for visualization)
    # allowing single node color setting too, for example for the loss node
    if hasattr(nodes, "__len__"):
        for n in nodes:
            setattr(n, '_vis_color', color)
    else:
        setattr(nodes, '_vis_color', color)

#apply vis to self andchild nodes once

def vis_color_selfandChild(root,color):
    setattr(root, '_vis_color', color)
    for child in root._prev:
         setattr(child, '_vis_color', color)

#new visualization function for loss function

def vis_color_entropy(root,color):
    # traverse the tree till you get the identifying function in this case its exp
    nodes, edges = [], []
    def build(v):
        if v not in nodes:
            nodes.append(v)
            setattr(v, '_vis_color', color)
            for child in v._prev:
                if v._op == 'exp':
                    delattr(v,'_vis_color')
                    setattr(v, '_vis_color', 'orange')
                    break
                build(child)
    build(root)
def trace(root):
    # traces the full graph of nodes and edges starting from the root
    nodes, edges = [], []
    def build(v):
        if v not in nodes:
            nodes.append(v)
            for child in v._prev:
                if (child, v) not in edges:
                    edges.append((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR', outfile='graph'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    # brew install graphviz
    # pip install graphviz
    from graphviz import Digraph
    assert rankdir in ['LR','TB']
    nodes, edges = trace(root)
    #increased rank step to beautify the cluster of lines on step 1 of the neural net
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir, 'nodesep': '0.1', 'ranksep': '0.8','newrank':'true'})


    #creating cluster ie. vizgraph sections to cluster inputs,params. intermediate nodes and loss
    with dot.subgraph(name='cluster0') as c:  
        c.attr(label='Input',fontsize="30")
        c.attr(rank='min')
        c.attr(style='filled', color='lightgreen')
  
    with dot.subgraph(name='cluster2') as c:
        c.attr(label='Loss',fontsize="30") 
        c.attr(style='filled', color='lightblue')
    with dot.subgraph(name='cluster3') as p:
        p.attr(label='Parameter Nodes',fontsize="30")
        p.attr(style='filled', color='lightyellow')
        dot.unflatten(stagger =3)
        c.attr(rank='same')

    with dot.subgraph(name='cluster4') as intn:
        intn.attr(label='Intermediate Nodes',fontsize="30") 
        intn.attr(style='filled', color='grey91')

    for n in nodes:
        fillcolor = n._vis_color if hasattr(n, '_vis_color') else "white"

        #add nodes to respective clusters based on the vis_color value
        if hasattr(n, '_vis_color'):
            if n._vis_color == 'lightblue':
                with dot.subgraph(name='cluster0') as c:
                    c.node(name=str(id(n)), label="data: %.4f\ngrad: %.4f" % (n.data, n.grad), shape='box', style='filled', fillcolor=fillcolor, width='0.1', height='0.1', fontsize='10')  
            elif n._vis_color == 'lightgreen' or n._vis_color == 'blue':
                with dot.subgraph(name='cluster3') as p:
                        p.node(name=str(id(n)), label="data: %.4f\ngrad: %.4f" % (n.data, n.grad), shape='box', style='filled', fillcolor=fillcolor,pos='0, 0!', width='0.1', height='0.1', fontsize='10')  
            elif n._vis_color == 'grey91':
                with dot.subgraph(name='cluster2') as c:
                    c.node(name=str(id(n)), label="data: %.4f\ngrad: %.4f" % (n.data, n.grad), shape='box', style='filled', fillcolor=fillcolor, width='0.1', height='0.1', fontsize='10')  
            else :
                with dot.subgraph(name='cluster4') as intn:
                        intn.node(name=str(id(n)), label="data: %.4f\ngrad: %.4f" % (n.data, n.grad), shape='box', style='filled', fillcolor=fillcolor, width='0.1', height='0.1', fontsize='10')  
        else:
             with dot.subgraph(name='cluster4') as intn:
                        intn.node(name=str(id(n)), label="data: %.4f\ngrad: %.4f" % (n.data, n.grad), shape='box', style='filled', fillcolor=fillcolor, width='0.1', height='0.1', fontsize='10')  
          

        #dot.node(name=str(id(n)), label="data: %.4f\ngrad: %.4f" % (n.data, n.grad), shape='box', style='filled', fillcolor=fillcolor, width='0.1', height='0.1', fontsize='10')
        if n._op:
            if hasattr(n, '_vis_color'):
                if n._vis_color == 'grey91':
                    with dot.subgraph(name='cluster2') as intn:
                            intn.node(name=str(id(n)) + n._op, label=n._op, width='0.1', height='0.1', fontsize='10')
                    dot.edge(str(id(n)) + n._op, str(id(n)), minlen='1')
                else:
                    with dot.subgraph(name='cluster4') as intn:
                            intn.node(name=str(id(n)) + n._op, label=n._op, width='0.1', height='0.1', fontsize='10')
                    dot.edge(str(id(n)) + n._op, str(id(n)), minlen='1')
            else:
                with dot.subgraph(name='cluster4') as intn:
                            intn.node(name=str(id(n)) + n._op, label=n._op, width='0.1', height='0.1', fontsize='10')
                dot.edge(str(id(n)) + n._op, str(id(n)), minlen='1')

    for n1, n2 in edges:
        #dot.edge(str(id(n1)), str(id(n2)) + n2._op, minlen='1')
        if hasattr(n1, '_vis_color'):
            if n1._vis_color == 'lightgreen' or n1._vis_color == 'blue':
                dot.edge(str(id(n1)), str(id(n2)) + n2._op, minlen='1')
                #dot.edge('dummy0',str(id(n1)),)
            else:
                dot.edge(str(id(n1)), str(id(n2)) + n2._op, minlen='1')
        else:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op, minlen='1')
        

    print("found a total of ", len(nodes), "nodes and", len(edges), "edges")
    print("saving graph to", outfile + "." + format)
    dot.render(outfile, format=format)
