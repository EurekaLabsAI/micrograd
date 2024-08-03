#include "utils.h"
#include <stdio.h>
#include <math.h>
#include <stdbool.h>


RNG rng = {42};

// ------------------------------------------------------------
// Value

typedef struct Value{
    double data;
    double grad;
    double m; // used for adamw
    double v; // used for adamw
    char *_op;
    struct Value **children;
    void (*_backward)(struct Value*);
} Value;

// Helper functions related to value struct
void print_value(Value *a) {
    printf("Value(data=%f, grad=%f)\n", a->data, a->grad);
}

Value* new_value(double data, Value **children, char *_op) {
    Value *val = malloc(sizeof(Value));
    val->data = data;
    val->grad = 0.0;
    val->m = 0.0;
    val->v = 0.0;
    val->_op = _op;
    val->children = children;
    val->_backward = NULL;
    return val;
}

// Backward function for the respective numeric operators
void add_backward(Value *out) {
    Value *first_val = out->children[0];
    Value *second_val = out->children[1];
    first_val->grad += out->grad;
    second_val->grad += out->grad;
}

void multiply_backward(Value *out) {
    Value *first_val = out->children[0];
    Value *second_val = out->children[1];
    first_val->grad += second_val->data * out->grad;
    second_val->grad += first_val->data * out->grad;
}

void power_backward(Value *out) {
    Value *first = out->children[0];
    Value *n = out->children[1];                 
    first->grad += n->data * pow(first->data, n->data - 1) * out->grad;
} 

void log_backward(Value *out) {
    Value *first = out->children[0];
    first->grad += (1 / (first->data)) * out->grad;
}

void true_div_backward(Value *out) {
    Value *first = out->children[0];
    Value *second = out->children[1];
    first->grad += (1 / second->data) * out->grad;
    second->grad += first->data * (-1/ pow(second->data, 2)) * out->grad;
}

void tanh_backward(Value *out) {
    Value *first= out->children[0];
    first->grad += exp(first->data) * out->grad;
}

void exp_backward(Value *out) {
    Value *first = out->children[0];
    first->grad += exp(first->data) * out->grad;
}

// Forward functions for numerical math operations
Value* multiply(Value *first_val, Value *second_val) {
    Value **children = calloc(2, sizeof(Value));
    children[0] = first_val;
    children[1] = second_val;
    Value *out = new_value(first_val->data * second_val->data, children, "*");
    out->_backward = multiply_backward;
    return out;
}

Value* add(Value *first_val, Value *second_val) {
    Value **children = calloc(2, sizeof(Value));
    children[0] = first_val;
    children[1] = second_val;
    Value *out = new_value(first_val->data + second_val->data, children, "+");
    out->_backward = add_backward;
    return out;
}

Value* negative(Value *first_val) {
    Value *neg_one = new_value(-1, NULL, "");
    return multiply(neg_one, first_val);
}

Value* subtract(Value *first_val, Value *second_val) {
    return add(first_val, negative(second_val));
}


Value* power(Value *first_val, Value *second_val) {
    Value **children = calloc(2, sizeof(Value));
    children[0] = first_val;
    children[1] = second_val;
    char *_op = "**";
    Value *out = new_value(pow(first_val->data, second_val->data), children, _op);
    out->_backward = power_backward;
    return out;
}

Value* log_value(Value *first_val) {
    Value **children = calloc(1, sizeof(Value));
    children[0] = first_val;
    Value *out = new_value(log(first_val->data), children, "log");
    out->_backward = log_backward;
    return out;
}

Value* exp_value(Value *first_val) {
    Value **children = malloc(sizeof(Value));
    children[0] = first_val;
    Value *out = new_value(exp(first_val->data), children, "exp");
    out->_backward = exp_backward;
    return out;
}

Value* true_div(Value *first_value, Value *second_value) {
    Value **children = calloc(2, sizeof(Value));
    children[0] = first_value;
    children[1] = second_value;
    Value *out = new_value(first_value->data / second_value->data, children, "/");
    out->_backward = true_div_backward;
    return out;
}

Value* tanh_act(Value *first_val) {
    Value **children = calloc(1, sizeof(Value));
    children[0] = first_val;
    Value *out = new_value(tanh(first_val->data), children, "tanh");
    out->_backward = tanh_backward;
    return out;
}

// LinkedList struct for handling multiple Values
typedef struct ValueListNode{
    Value *element;
    struct ValueListNode *next;
} ValueListNode;

typedef struct ValueList{
    ValueListNode *head;
    ValueListNode *tail;
} ValueList;

// Helper functions related to linked list values

void reverse_value_list(ValueList *list) {
    ValueListNode *prev = NULL;
    if(list->head == NULL) {
        return;
    }
    ValueListNode *oldhead = list->head;
    while(list->head != NULL) {
        ValueListNode *tnext = list->head->next;
        list->head->next = prev;
        prev = list->head;
        list->head = tnext;
    }
    list->head = prev;
    list->tail = oldhead;
}


bool is_present(Value *out, ValueList *list) {
    if(list->head == NULL) {
        return false;
    }
    ValueListNode *current = list->head;
    while(current != NULL) {
        if(current->element == out) {
            return true;
        }
        current = current->next;
    }
    return false;
}

ValueListNode* new_value_list_node(Value *node) {
    ValueListNode *new_node = malloc(sizeof(ValueListNode));
    new_node->element = node;
    new_node->next = NULL;
    return new_node;
}

int get_size(ValueList *list) {
    int size = 0;
    ValueListNode *head = list->head;
    while(head != NULL) {
        size++;
        head = head->next;
    }
    return size;
}
void merge_value_list(ValueList *list1, ValueList *list2) {
    if(list1->head == NULL) {
        list1->head = list2->head;
        list1->tail = list2->tail;
    }
    list1->tail->next = list2->head;
}

void insert_value_list(ValueList *list, Value *node) {
    if(list->head == NULL) {
        list->head = new_value_list_node(node);
        list->tail = list->head;
        return;
    }
    list->tail->next = new_value_list_node(node);
    list->tail = list->tail->next;
    return;
}

void print_value_list(ValueList *list) {
    ValueListNode *head = list->head;
    printf("[");
    while(head != NULL) {
        printf("{%f %s %f}", head->element->data, head->element->_op, head->element->grad);
        head = head->next;
    }
    puts("]");
}

ValueList* new_list() {
    ValueList *list = malloc(sizeof(ValueList));
    list->head = NULL;
    list->tail = NULL;
    return list;
}

// Helper function to build the computation graph
void build_topo(Value *out, ValueList *topological_order, ValueList *value_list) {
    if(out != NULL && !is_present(out, value_list)) {
        insert_value_list(value_list, out);
        for(int k = 0; out->children != NULL && out->children[k] != NULL; k++) {
            build_topo(out->children[k], topological_order, value_list);
        }
        insert_value_list(topological_order, out);
    }
}

// Perform backward pass from a specific value
void backward(Value *out) {
    ValueList *topological_order = new_list();
    ValueList *value_list = new_list();

    // Build the topological sorted order
    build_topo(out, topological_order, value_list);

    // Reverse the topological sorted order
    reverse_value_list(topological_order);

    // Start backpropagation from out value
    out->grad = 1.0;
    ValueListNode *current_value_node = topological_order->head;
    do {
        if(current_value_node->element->_backward != NULL) {
            current_value_node->element->_backward(current_value_node->element);
        }
        current_value_node = current_value_node->next;
    }while(current_value_node != NULL);
}

// ------------------------------------------------------------
// Multi Layered Perceptron (MLP) network

typedef struct Neuron{
    Value **weight;
    Value *bias;
    int nin;
    bool non_linear;
} Neuron;

Neuron* new_neuron(int nin, bool non_linear) {
    Neuron *neuron = malloc(sizeof(Neuron));
    neuron->weight = calloc(nin, sizeof(Value));
    neuron->bias = new_value(0.0, NULL, " ");
    neuron->nin = nin;
    neuron->non_linear = non_linear;

    for(int k = 0; k < nin; k++) {
        neuron->weight[k] = new_value(rng_uniform(&rng, -1, 1) / sqrt(nin),
        NULL, 
        " ");
    }
    return neuron;
}
Value* forward_neuron(Neuron *neuron, Value **in) {
    Value *act = NULL;
    act = multiply(neuron->weight[0], in[0]);
    for(int i = 1; i < neuron->nin; i++) {
        act = add(act, multiply(neuron->weight[i], in[i]));
    }
    act = add(act, neuron->bias);
    if(neuron->non_linear) {
        act = tanh_act(act);
    }
    return act;
}

Value** neuron_parameters(Neuron *neuron) {
    Value **parameters = calloc(neuron->nin + 1, sizeof(Value));
    for(int i = 0; i < neuron->nin; i++) {
        parameters[i] = neuron->weight[i];
    }
    parameters[neuron->nin] = neuron->bias;
    return parameters;
}

void print_neuron(Neuron *neuron) {
    char *act_type = "Linear";
    if(neuron->non_linear) {
        act_type = "Tanh";
    }
    printf("%sNeuron%d ", act_type, neuron->nin);
}

typedef struct Layer {
    Neuron *neurons;
    int nout;
} Layer;


Layer* new_layer(int nin, int nout, bool non_linear) {
    Neuron *neurons = calloc(nout, sizeof(Neuron));
    for(int i = 0; i < nout; i++) {
        neurons[i] = *new_neuron(nin, non_linear);
    }
    Layer *layer = malloc(sizeof(Layer));
    layer->neurons = neurons;
    layer->nout = nout;
    return layer;
}

Value** forward_layer(Layer *layer, Value **input) {
    Value **output = calloc(layer->nout, sizeof(Value));
    for(int i = 0; i < layer->nout; i++) {
        output[i] = forward_neuron(&layer->neurons[i], input);
    }
    return output;
}


ValueList* layer_parameters(Layer *layer) {
    ValueList *parameters = new_list();
    for(int i = 0; i < layer->nout; i++) {
         Value **n_parameters = neuron_parameters(&layer->neurons[i]);
         for(int j = 0; j < layer->neurons[i].nin + 1; j++) {
            insert_value_list(parameters, 
            n_parameters[j]);
         }
    }
    return parameters;
}

void print_layer(Layer *layer) {
    printf("[");
    for(int i = 0; i < layer->nout; i++) {
        print_neuron(&layer->neurons[i]);
    }
    printf("]");
}

typedef struct MLP {
    int layer_count;
    Layer **layers;
} MLP;

MLP* new_mlp(int nin, int nouts[], int nout_dims) {
    int sz[nout_dims + 1];
    sz[0] = nin;
    for(int k = 0; k < nout_dims; k++) {
        sz[k+1] = nouts[k];
    }
    Layer **layers = calloc(nout_dims - 1, sizeof(Layer));
    for(int k = 0; k < nout_dims; k++) {
        layers[k] = new_layer(sz[k], sz[k+1], (k != nout_dims - 1));
    }
    MLP *mlp = malloc(sizeof(MLP));
    mlp->layer_count = nout_dims;
    mlp->layers = layers;
    return mlp;
}

ValueList* model_parameters(MLP *model) {
    ValueList *model_params = new_list();
    for(int i = 0; i < model->layer_count; i++) {
        merge_value_list(model_params, layer_parameters(model->layers[i]));
    }
    return model_params;
}

Value** forward_model(MLP *model, Value **input) {
    for(int i = 0; i < model->layer_count; i++) {
        input = forward_layer(model->layers[i], input);
    }
    return input;
}

// Helper functions for calculating cross entropy loss
Value* get_max(Value **elements, int size) {
    Value *maximum = elements[0];
    for(int i = 1; i < size; i++) {
        if(elements[i]->data > maximum->data) {
            maximum = elements[i];
        }
    }
    return maximum;
}

Value* sum(Value **elements, int size) {
    Value *sum = new_value(elements[0]->data, NULL, " ");
    for(int i = 1; i < size; i++) {
        sum = add(sum, elements[i]);
    }
    return sum;
}
Value* cross_entropy(Value **logits, Value *target, int size) {
    Value *maximum = get_max(logits, size);
    for(int i = 0; i < size; i++) {
        logits[i] = subtract(logits[i], maximum);
    }
    Value **logits_exp = calloc(size, sizeof(Value));
    for(int i = 0; i < size; i++) {
        logits_exp[i] = exp_value(logits[i]);
    }
    Value *denom = sum(logits_exp, size);
    Value **probs = calloc(size, sizeof(Value));
    for(int i = 0; i < size; i++) {
        probs[i] = true_div(logits_exp[i], denom);
    }
    Value *logp = log_value(probs[(int) target->data]);
    return negative(logp);
}

// Helper function to calculation evalution loss
double eval_split(MLP *model, DataPoint **split, int size, int max_target_size) {
    Value *loss = new_value(0, NULL, " ");
    for(int i = 0; i < size; i++) {
        Value **input = calloc(2, sizeof(Value));
        input[0] = new_value(split[i]->x, NULL, " ");
        input[1] = new_value(split[i]->y, NULL, " ");
        //Get the logits for the input
        Value **logits = forward_model(model, input);

        //calculate loss
        Value *current_loss = cross_entropy(logits, 
                                       new_value(split[i]->label, NULL, " "), 
                                       max_target_size);
        loss = add(loss, current_loss);
    }
    return loss->data;
}

 
// Make parameters gradient as zero before starting next iteration
void zero_grad(ValueList *params) {
    ValueListNode *param_head = params->head;
    while(param_head != NULL) {
        param_head->element->grad = 0.0;
        param_head = param_head->next;
    }
}

// Update parameters using adamw optimizer
void update_params_with_grad_adamw(ValueList *params, double learning_rate, 
double beta1, double beta2, double weight_decay, int step) {
    ValueListNode *param_head = params->head;
    while(param_head != NULL) {
        param_head->element->m = beta1 * param_head->element->m + (1 - beta1) * param_head->element->grad;
        param_head->element->v = beta2 * param_head->element->v + (1 - beta2) * pow(param_head->element->grad, 2);
        double m_hat = param_head->element->m / (1 - pow(beta1, step + 1));
        double v_hat = param_head->element->v / (1 - pow(beta2, step + 1));
        param_head->element->data -= learning_rate * (m_hat / (sqrt(v_hat) + 1e-8) + weight_decay * param_head->element->data);
        param_head = param_head->next;
    }
}
// Update parameters using gradient descent
void update_params_with_grad(ValueList *params, double learning_rate) {
    ValueListNode *param_head = params->head;
    while(param_head != NULL) {
        param_head->element->data -= learning_rate * param_head->element->grad;
        param_head = param_head->next;
    }
}

// Training and validation
int main() {
   int target_size = 3, input_size = 2;
   int nouts[] = {16, target_size}; 
   //Init model - 16 neurons, 3 output neurons (logits)
   MLP *mlp = new_mlp(input_size, nouts, 2);

   // Generate a random dataset with 100 2D datapoints in 3 classes
   DataSet *dataset = generate(100, &rng);

    // Hyper-parameters
   double learning_rate = 1e-1;
   double beta1 = 0.9, beta2 = 0.95;
   double weight_decay = 1e-4;

   int iterations = 100;

    for(int step = 0; step < iterations; step++) {
        if(step % 10 == 0) {
            double validation_loss = eval_split(mlp, 
            dataset->val, 
            dataset->val_size, 
            3);
            printf("step %d, validation loss %f\n", step, validation_loss);
        }
        Value *loss = new_value(0, NULL, "loss");
        for(int i = 0; i < dataset->tr_size; i++) {
            Value **input = calloc(2, sizeof(Value));
            input[0] = new_value(dataset->tr[i]->x, NULL, "x");
            input[1] = new_value(dataset->tr[i]->y, NULL, "y");
            Value **logits = forward_model(mlp, input);
            loss = add(loss, cross_entropy(logits, 
            new_value(dataset->tr[i]->label, NULL, " "), target_size));
        }
        // Average the loss
        loss = multiply(loss, true_div(new_value(1.0, NULL, " "), 
                                    new_value(dataset->tr_size,NULL, " ")));
        // Performing backward on loss
        backward(loss);

        ValueList *model_params = model_parameters(mlp);
        
        // Update the params with gradient computed
        update_params_with_grad_adamw(model_params, learning_rate, beta1, beta2, weight_decay, step);
        //update_params_with_grad(model_params, learning_rate);
        //Make the grad of parameters as zero before next iteration
        zero_grad(model_params);
        
        printf("step %d, train loss %f\n", step, loss->data);
   }
   return 0;
}