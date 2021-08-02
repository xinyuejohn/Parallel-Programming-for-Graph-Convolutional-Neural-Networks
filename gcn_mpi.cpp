#include "Model.hpp"
#include "Node.hpp"
#include <mpi.h>
#include <mpi.h>

#define DEBUG 0

/***************************************************************************************/
void first_layer_transform(Node** nodes, int num_nodes, Model &model){
    Node* node;
    #pragma omp parallel for 
    for(int n = 0; n < num_nodes; ++n){
        node = nodes[n];
        for(int c_in = 0; c_in < node->dim_features; ++c_in){
            for(int c_out = 0; c_out < node->dim_hidden; ++c_out){
                node->tmp_hidden[c_out] += node->x[c_in] * model.weight_1[c_in * node->dim_hidden + c_out];
            }
        }   
    }
}
/***************************************************************************************/


/***************************************************************************************/
void first_layer_aggregate(Node** nodes, int num_nodes, Model &model){
    // aggregate
    float* message;
    float norm;
    Node* node;

    #pragma omp parallel for private(norm, message)
    for(int n = 0; n < num_nodes; ++n){
        node = nodes[n];
        // aggregate from each neighbor
        for(int neighbor : node->neighbors){
            message = nodes[neighbor]->tmp_hidden;
            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);
            // aggregate normalized message
            for(int c = 0; c < node->dim_hidden; ++c){
                node->hidden[c] += message[c] / norm;
            }
        }
        // add bias
        for(int c = 0; c < node->dim_hidden; ++c){
            node->hidden[c] += model.bias_1[c];
        }
        // apply relu
        for(int c = 0; c < node->dim_hidden; ++c){
            node->hidden[c] = node->hidden[c] < 0.0 ? 0.0 : node->hidden[c];
        }
    }
}
/***************************************************************************************/


/***************************************************************************************/
// computation in second layer
void second_layer_transform(Node** nodes, int num_nodes, Model &model){
    // transform
    Node* node;
    #pragma omp parallel for 
    for(int n = 0; n < num_nodes; ++n){
        node = nodes[n];
        for(int c_out = 0; c_out < node->num_classes; ++c_out){
            for(int c_in = 0; c_in < node->dim_hidden; ++c_in){
                node->tmp_logits[c_out] += node->hidden[c_in] * model.weight_2[c_in * node->num_classes + c_out];
            }
        } 
    }
}
/***************************************************************************************/


/***************************************************************************************/
void second_layer_aggregate(Node** nodes, int num_nodes, Model &model){
    // aggregate
    Node* node;
    float* message;
    float norm;
    // for each node
    #pragma omp parallel for private(norm, message)
    for(int n = 0; n < num_nodes; ++n){
        node = nodes[n];
        // aggregate from each neighbor
        for(int neighbor : node->neighbors){
            message = nodes[neighbor]->tmp_logits;
            // normalization w.r.t. degrees of node and neighbor
            norm = 1.0 / sqrt(node->degree * nodes[neighbor]->degree);

            // aggregate normalized message
            for(int c = 0; c < node->num_classes; ++c){
                node->logits[c] += message[c] / norm;
            }
        }

        // add bias
        for(int c = 0; c < node->num_classes; ++c){
            node->logits[c] += model.bias_2[c];
        }
    }        
}
/***************************************************************************************/


/***************************************************************************************/
void create_graph(Node** nodes, Model &model){
    // set neighbor relations
    int source, target;

    #pragma omp parallel for private(source, target)
    for(int e = 0; e < model.num_edges; ++e){
        source = model.edges[e];
        target = model.edges[model.num_edges + e];
        // self-loops twice in edges, so ignore for now
        // and add later
        if (source != target){
            nodes[source]->neighbors.push_back(target);
        }
    }

    // add self-loops
    #pragma omp parallel for 
    for(int n = 0; n < model.num_nodes; ++n){
        Node *node = nodes[n];
        node->neighbors.push_back(node->ID);
        node->degree = node->neighbors.size();
    }
}
/***************************************************************************************/


/***************************************************************************************/
int main(int argc, char** argv) {
    int seed = -1;
    int init_no = -1;
    std::string dataset("");

    // init MPI
    int size, rank;
    // MPI_Status s;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #if DEBUG
        double time_val = MPI_Wtime();
        printf("MPI Process %d of %d (value=%f)\n", rank, size, time_val);
    #endif

    // specify problem
    #if DEBUG
        // for measuring your local runtime
        auto tick = std::chrono::high_resolution_clock::now();
        if(rank == 0){
            Model::specify_problem(argc, argv, dataset, &init_no, &seed);
        }
        MPI_Bcast(&init_no, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #else
        if(rank == 0){
            Model::specify_problem(dataset, &init_no, &seed);
        }
        MPI_Bcast(&init_no, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #endif

    // load model specifications and model weights
    Model model(dataset, init_no, seed);

    if(rank == 0){
        model.load_model();        
    }
    // broadcast meta data first
    // signature of Bcast: int MPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
    MPI_Bcast(&model.num_classes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&model.dim_features, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&model.dim_hidden, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&model.num_nodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&model.num_edges, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // of type float
    int weight_1_count = model.dim_features * model.dim_hidden;
    int weight_2_count = model.dim_hidden * model.num_classes;
    int bias_1_count = model.dim_hidden;
    int bias_2_count = model.num_classes;
    // of type int
    int labels_count = model.num_nodes;
    int edges_count = model.num_edges * 2;
    if(rank > 0){
        int weight_bias_count = weight_1_count + weight_2_count + bias_1_count + bias_2_count;
        int labels_edges_count = labels_count + edges_count;
        float* weight_bias_buffer = (float*)calloc(weight_bias_count, sizeof(float));
        int* labels_edges_buffer = (int*)calloc(labels_edges_count, sizeof(int));
        model.weight_1 = weight_bias_buffer;
        model.weight_2 = model.weight_1 + weight_1_count;
        model.bias_1 = model.weight_2 + weight_2_count;
        model.bias_2 = model.bias_1 + bias_1_count;
        model.labels = labels_edges_buffer;
        model.edges = labels_edges_buffer + labels_count;
    }
    MPI_Bcast(model.weight_1, weight_1_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(model.weight_2, weight_2_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(model.bias_1, bias_1_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(model.bias_2, bias_2_count, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(model.labels, labels_count, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(model.edges, edges_count, MPI_INT, 0, MPI_COMM_WORLD);
    

    // create graph (i.e. load data into each node and load edge structure)
    int dim_features = model.dim_features;
    int dim_hidden = model.dim_hidden;
    int num_classes = model.num_classes;
    int num_nodes = model.num_nodes;
    
    Node** nodes = (Node**)malloc(num_nodes * sizeof(Node*));
    if(nodes == nullptr){
        exit(1);
    }
    // initialize nodes
    char* x_buffer = (char*)calloc(dim_features, sizeof(float) * num_nodes);
    int read_size = dim_features * sizeof(float);
// perform actual computation in network
    int nodes_start_pos = num_nodes / size * rank;
    int nodes_local_num = num_nodes / size;
    if(rank == size - 1){
        nodes_local_num += num_nodes % size;
    }
    // if rank==0, initialize nodes by reading from files
    if(rank == 0){
        // bulk read files
        for(int n = 0; n < num_nodes; ++n){
            std::stringstream ss; ss << model.base_dir << "X/x_" << n << ".bin";
            std::string filename = ss.str();
            std::ifstream binary(filename, std::ios::binary);
            if (binary.fail()){
                std::cerr << "File " << filename << " does not exist, aborting!" << std::endl;
                exit(1);
            }
            binary.read(x_buffer + n * read_size, read_size);
        }
        for(int i = 1; i < size; i++){
            #if DEBUG
                std::cout << "Sending x_buffer from rank 0 to " << i << std::endl;
                // std::cout << "x_buffer[0] + x_buffer[10] + x_buffer[100]: " << (int)x_buffer[0] + (int)x_buffer[10] + (int)x_buffer[100] << std::endl;
            #endif
            int nodes_start_pos_i = num_nodes / size * i;
            int nodes_local_num_i = num_nodes / size;
            if(i == size - 1){
                nodes_local_num_i += num_nodes % size;
            }    
            MPI_Send(x_buffer + nodes_start_pos_i * read_size, nodes_local_num_i * read_size, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        }
    }
    // if rank>0, receive nodes and fill in nodes list
    else{
        MPI_Status s1;
        MPI_Recv (x_buffer + nodes_start_pos * read_size, nodes_local_num * read_size, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &s1);
        #if DEBUG
            std::cout << "Recv result of process " << rank << ": " << s1.MPI_ERROR << std::endl;
            // std::cout << "x_buffer[0] + x_buffer[10] + x_buffer[100]: " << (int)x_buffer[0] + (int)x_buffer[10] + (int)x_buffer[100] << std::endl;
        #endif
    }


    float* x_space = (float*) x_buffer;
    float* tmp_hidden_space = (float*)calloc(dim_hidden, sizeof(float) * num_nodes); 
    float* hidden_space = (float*)calloc(dim_hidden, sizeof(float) * num_nodes);          // holds hidden representation (i.e. after transformation and aggregation)
    float* tmp_logits_space = (float*)calloc(num_classes, sizeof(float) * num_nodes);      // holds copy of logits (for aggregation)
    float* logits_space = (float*)calloc(num_classes, sizeof(float) * num_nodes); 
    
    for(int n = 0; n < num_nodes; ++n){
        nodes[n] = new Node(n, model, 0);
        nodes[n]->x = x_space + n * dim_features;
        nodes[n]->tmp_hidden = tmp_hidden_space + n * dim_hidden;
        nodes[n]->hidden = hidden_space + n * dim_hidden;
        nodes[n]->tmp_logits = tmp_logits_space + n * num_classes;
        nodes[n]->logits = logits_space + n * num_classes;
    }
    
    create_graph(nodes, model);

    
    first_layer_transform(nodes + nodes_start_pos, nodes_local_num, model);
    // transfer nodes->tmp_hidden to rank 0
    if(rank == 0){
        for(int i = 1; i < size; i++){
            int nodes_start_pos_i = num_nodes / size * i;
            int nodes_local_num_i = num_nodes / size;
            if(i == size - 1){
                nodes_local_num_i += num_nodes % size;
            }
            MPI_Status s;
            MPI_Recv (tmp_hidden_space + nodes_start_pos_i * dim_hidden, // out buffer
                nodes_local_num_i * dim_hidden, // count
                MPI_FLOAT, // data type
                i, // sender's rank
                1, // msg tag
                MPI_COMM_WORLD, // communicator
                &s); // status information
        }
        first_layer_aggregate(nodes, model.num_nodes, model);
    }
    else{
        MPI_Send(tmp_hidden_space + nodes_start_pos * dim_hidden, nodes_local_num * dim_hidden, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }
    // synchronize nodes->hidden across processes
    if(rank == 0){
        for(int i = 1; i < size; i++){
            // only send part of nodes, cause only them are used in the second transformation
            int nodes_start_pos_i = num_nodes / size * i;
            int nodes_local_num_i = num_nodes / size;
            if(i == size - 1){
                nodes_local_num_i += num_nodes % size;
            }
            MPI_Send(hidden_space + nodes_start_pos_i * dim_hidden, nodes_local_num_i * dim_hidden, MPI_FLOAT, i, 2, MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Status s;
        MPI_Recv (hidden_space + nodes_start_pos * dim_hidden, // out buffer
                nodes_local_num * dim_hidden, // count
                MPI_FLOAT, // data type
                0, // sender's rank
                2, // msg tag
                MPI_COMM_WORLD, // communicator
                &s); // status information
    }

    second_layer_transform(nodes + nodes_start_pos, nodes_local_num, model);
    // transfer nodes->tmp_logits to process 0
    if(rank == 0){
        for(int i = 1; i < size; i++){
            int nodes_start_pos_i = num_nodes / size * i;
            int nodes_local_num_i = num_nodes / size;
            if(i == size - 1){
                nodes_local_num_i += num_nodes % size;
            }
            MPI_Status s;
            MPI_Recv (tmp_logits_space + nodes_start_pos_i * num_classes, // out buffer
                nodes_local_num_i * num_classes, // count
                MPI_FLOAT, // data type
                i, // sender's rank
                3, // msg tag
                MPI_COMM_WORLD, // communicator
                &s); // status information
        }
        second_layer_aggregate(nodes, model.num_nodes, model);
    }
    else{
        MPI_Send(tmp_logits_space + nodes_start_pos * num_classes, nodes_local_num * num_classes, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
    }

    if(rank == 0){
        // compute accuracy
        float acc = 0.0;
        int pred, correct;
        // #pragma omp parallel for private(pred, correct) reduction(+:acc)
        for(int n = 0; n < model.num_nodes; ++n){
            pred = nodes[n]->get_prediction();
            correct = pred == model.labels[n] ? 1 : 0;
            acc = acc + (float)correct;
        }
        
        acc = acc / model.num_nodes;
        std::cout << "accuracy " << acc << std::endl;
        std::cout << "DONE" << std::endl;
    }

    #if DEBUG
        // for measuring your local runtime
        auto tock = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = tock - tick;
        std::cout << "elapsed time " << elapsed_time.count() << " second" << std::endl;
    #endif

    // clean-up 
    free(nodes);
    free(x_space);
    free(tmp_hidden_space);
    free(hidden_space);
    free(tmp_logits_space);
    free(logits_space);
    if(rank == 0){
        model.free_model();
    }
    else{
        // they are the start of float buffer and the start of int buffer
        free(model.weight_1);
        free(model.labels);
    }

    (void)argc;
    (void)argv;

    MPI_Finalize();

    return 0;
}
/***************************************************************************************/
