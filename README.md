yabn
====

Yet Another Bayes Net

YABN is a small script intended to allow for the construction of linear Bayesian networks, and sampling from them. Currently, the only emission distribution implemented is the NormalDistribution. This has been merged with my larger project, <a href="https://github.com/jmschrei/pomegranate">pomegranate</a>, which is a more extensive graphical models library.

<h2>Usage</h2>

A `network` is the primary object, comprised of `nodes` which each have an emission `distribution`. See the following example:

```
from yabn import *

n1 = Node( NormalDistribution( 5, 2 ), name="n1" )
n2 = Node( NormalDistribution( 23, 1 ), name="n2" )

network = Network( name="test" )
network.add_nodes( [n1, n2] )
network.add_transition( n1, n2, 0.5 )
network.bake()

print network.sample()
```

In this case, two nodes were originally created. The first node had an emission distribution of ~Normal( 5, 2 ), and the second had an emission distribution of ~Normal( 23, 1 ). The nodes are named appropriately, so that the results can be easily interpreted. Next, a network object is made, which represents the entire Bayesian network. Nodes need to be added to the model, either individually or as a list. Then transitions need to be added one at a time, with the weight of the edge being the last parameter. Lastly, the network needs to be ~baked~, which means you're done adding to the graph and want to convert it to the sparse matrix representation for efficient and speedy calculations. After baking it, the network can be sampled, which means that samples are drawn from the emission distributions of nodes, and values are pushed downstream where edges exist. For example, if n1 sampled a 5, and node n2 sampled a 23, since there is an edge with weight 0.5, node n2 would actually emit 25.5, since that's 23+(5*0.5).  
