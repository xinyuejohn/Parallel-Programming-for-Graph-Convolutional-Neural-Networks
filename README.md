# Parallel-Programming-for-Graph-Convolutional-Neural-Networks
Using openMP, MPI and SIMD intrinsics to accelerate the straight-forward sequential implementation of the Graph Convolutional Neural Networks.

# Sequential Optimization

  * **Problem Specification**

  We decided to optimize with respect to the following section of the code:

```
1        for(int c_out = 0; c_out < node->dim_hidden; ++c_out){
2            for(int c_in = 0; c_in < node->dim_features; ++c_in){
3                node->tmp_hidden[c_out] += node->x[c_in] * model.weight_1[c_in * node->dim_hidden + c_out];
4            }
5        }
```

  In line 3, 'model.weight_1[c_in * node->dim_hidden + c_out]' will result in massive cache miss.
  Given that the outer part of the loop is 'c_out', every time a weight is loaded, weights that are located near the specified weight will be loaded. For example, c_in * node->dim_hidden + c_out + 1, c_in * node->dim_hidden + c_out + 2. But the execution of the next weight would not be those weights loaded in cache. It would instead be '(c_in_original+1) * node->dim_hidden + c_out', which is way beyond the cache range and will result in inefficiency.

  * **Solution: Loop Interchange**

```
1        for(int c_in = 0; c_in < node->dim_features; ++c_in){
2            for(int c_out = 0; c_out < node->dim_hidden; ++c_out){
3                node->tmp_hidden[c_out] += node->x[c_in] * model.weight_1[c_in * node->dim_hidden + c_out];
4            }
5        }
```

  The inner and outer loop are interchanged(in line 1 and line 2), in this way, the data locality problem can be solved.

  **This Optimization is added in the previous submitted 'gcn_omp.cpp'(can be seen in 'gcn_omp.cpp' in the current folder) and leads to speed up of 83-84 combined with openMP**


# OpenMP Optimization
 * **Implementation**
 using the following clause to parallelize 
```
#pragma omp parallel for 
#pragma omp parallel for private()
#pragma omp parallel for private() reduction()

```

# MPI Optimization

  * **Goal**
  
  Distribute the workload of 'first_layer_transform' and 'second_layer_transform' to evenly (ideally, if the remainder is 0, otherwise the last node will be assigned slightly more workload).

  * **Implementation**

    * Specify problem in rank 0 node
    * Broadcast the model and its relevant information from rank 0 to all the other nodes
    * Create a Contiguous memory buffer to store the input 'x' of nodes in all nodes
    * Compute the workload of each node w.r.t. the buffer(for 'x')
    * Read all input from rank 0 and send the needed input for the workload to each node. The nodes who received the input will store them in their own buffers in the specified location
    * Allocate Contiguous memory for 'logits', 'temp_hidden', 'hidden', 'temp_logits' in each node.


    **NOTE**: Contiguous memory allocation for 'x', 'logits', 'temp_hidden', 'hidden', 'temp_logits' are used to simplify the exchange of information between nodes via mpi. Otherwise, if they are initialized not in a contiguous way, an additional loop will be needed to send and receive data.

    * Each node computes the result w.r.t. its assigned nodes separately and send the results back to the rank 0 node, where the output will be synchronized.

    ***First layer transform and aggregation ends, second layer transform starts***

    * Send the needed input for the workload to each node. The nodes who received the input will store them in the specified location. 
    * Each node computes the result w.r.t. its assigned nodes separately and send the results back to the rank 0 node, where the output will be synchronized.

    ***Second layer transform and aggregation ends***

    * Compute the accuracy on rank 0 node


# Hybrid Programing

## Final Implementation

#### Note

The increased speed up shown in this part is the _combined result_ of the previous implementations. For example, the speed up resulting from parallelizing reading the input loop part is combined with the previous parallelization of those functions.

* Changed from ``MPI_Init(&argc, &argv)`` to ``MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &prov);``
* Parallelized the ``first_layer_transform``, ``second_layer_transform``, ``first_layer_aggregate``, ``second_layer_aggregate``, ``create_graph`` and the function in calculating accuracy in rank 0
  * Increased speed up to 20-23
* Parallelized the reading input loop in rank 0 
  * Increased speed up to around 42
* Parallelized the sending input ``x `` to other processes part 
  * Increased speed up to 55-58

## Other Experiments

* Parallelized the rest of the loops (e.g. the loops in transferring intermediate results) and didn't lead to speed up or may even result in slight decrease in speedup
* Given that the parallelization of sending input ``x`` to other processes part can lead to a combined effect of increase in speedup (at least +13), we tried replacing the previous sending `x` via a loop of `MPI_Send` to `MPI_Scatterv`. The result is that the combined effect of  ``MPI_Scatterv`` has only around 20, hence we stick with our current implementation. 










