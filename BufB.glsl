/* 
 * NeuroST - https://github.com/defgsus/neurost
 * (c) 0x7e0, Stefan Berke
 * License Creative Commons Attribution 3.0 Unported 
 * 
 */ 

/* * * Forward pass, error derivative, backward pass * * *
 
 Description at https://github.com/defgsus/neurost/blob/master/idea.txt 
*/



// ------- config --------

#define NUM_LAYER      	3
#define NUM_INPUT		(16*16)
#define NUM_OUTPUT		10
#define CELL_SIZE		3
#define DO_TRAIN		1

#define NUM_CELLS_0     NUM_INPUT
#define NUM_CELLS_1		40
#define NUM_CELLS_2		10
#define NUM_CELLS_3		NUM_OUTPUT

// http://www.musicdsp.org/showone.php?id=238
float Tanh(in float x) { return clamp(x * (27. + x * x) / (27. + 9. * x * x), -1., 1.); }

// activation function
float activation(in float x) { return Tanh(x); }
float derivative(in float x) { return 1. - x * x; }

// ----- end config ------



// ------------ auto config ------------

#if CELL_SIZE == 4
	#define TYPE 			vec4
	#define VEC4_TO_TYPE(v) (v)
	#define TYPE_TO_VEC4(v) (v)
#elif CELL_SIZE == 3
	#define TYPE 			vec3
	#define VEC4_TO_TYPE(v) (v).xyz
	#define TYPE_TO_VEC4(v) vec4(v, 1.)
#elif CELL_SIZE == 2
	#define TYPE 			vec2
	#define VEC4_TO_TYPE(v) (v).xy
	#define TYPE_TO_VEC4(v) vec4(v, 1., 1.)
#else
#	define TYPE 			float
#	define VEC4_TO_TYPE(v) (v).x
#	define TYPE_TO_VEC4(v) vec4(v, v, v, 1.)
#endif

int num_cells[4];
int state_y[4];
int error_y[4];
int weight_y[3];

void _initLayer()
{
    num_cells[0] = NUM_CELLS_0;
    num_cells[1] = NUM_CELLS_1;
    num_cells[2] = NUM_CELLS_2;
    num_cells[3] = NUM_CELLS_3;
    
    state_y[0] = NUM_LAYER - 1;
    state_y[1] = NUM_LAYER - 2;
    state_y[2] = NUM_LAYER - 3;
    state_y[3] = NUM_LAYER - 4;
    
    error_y[0] = NUM_LAYER - 1 + 20;
    error_y[1] = NUM_LAYER - 2 + 20;
    error_y[2] = NUM_LAYER - 3 + 20;
    error_y[3] = NUM_LAYER - 4 + 20;
    
    int y = 0;
    weight_y[0] = y; y += NUM_CELLS_1;
    weight_y[1] = y; y += NUM_CELLS_2;
    weight_y[2] = y; y += NUM_CELLS_3;
}

// automatic type overloads for activation()
vec2 activation(in vec2 x) {
    return vec2(activation(x.x), activation(x.y)); }
vec3 activation(in vec3 x) {
    return vec3(activation(x.x), activation(x.y), activation(x.z)); }
vec4 activation(in vec4 x) {
    return vec4(activation(x.x), activation(x.y), activation(x.z), activation(x.w)); }
vec2 derivative(in vec2 x) {
    return vec2(derivative(x.x), derivative(x.y)); }
vec3 derivative(in vec3 x) {
    return vec3(derivative(x.x), derivative(x.y), derivative(x.z)); }
vec4 derivative(in vec4 x) {
    return vec4(derivative(x.x), derivative(x.y), derivative(x.z), derivative(x.w)); }

#define NUM_HIDDEN_LAYER (NUM_LAYER - 2)
#if DO_TRAIN != 0
	#define NUM_FRAME_HOLD (NUM_LAYER*2)
#else
	#define NUM_FRAME_HOLD (NUM_LAYER)
#endif

// ------------- end auto config -------------


// ---------- states & values ---------

TYPE texLookup(in sampler2D sam, in ivec2 pix)
{
    return VEC4_TO_TYPE(
        texture2D(sam, (vec2(pix) + .5) / iResolution.xy) 
    	);
}


TYPE externalInputState(in int cellIdx)
{
    ivec2 ip = ivec2(int(mod(float(cellIdx), 16.)),
                     cellIdx / 16);
    return texLookup(iChannel0, ip + ivec2(0, 1));
}

TYPE expectedOutputState(in int cellIdx)
{
    return texLookup(iChannel0, ivec2(cellIdx, 0));
}

// input to each layer
TYPE cellState(in int layer, in int cellIdx)
{
    return texLookup(iChannel1, 
                     ivec2(cellIdx, NUM_LAYER-1-layer));
}

// error at each layer
TYPE cellError(in int layer, in int cellIdx)
{
    return texLookup(iChannel1,
                     ivec2(cellIdx, NUM_LAYER-1+20-layer));
}

// weight from inCell (layer-1) to outCell (layer)
TYPE weight(in int layer, in int inCell, in int outCell)
{
    if (layer == 1)
	    return texLookup(iChannel2, 
                     ivec2(inCell, 
                           outCell));
    else if (layer == 2)
	    return texLookup(iChannel2, 
                     ivec2(inCell, 
                           outCell + NUM_CELLS_0));
    else if (layer == 3)
	    return texLookup(iChannel2, 
                     ivec2(inCell, 
                           outCell + NUM_CELLS_0 + NUM_CELLS_1));
    else 
        return TYPE(0.); 
}

// -------- end states & values -------




// forward propagation of layer-1 to layer
TYPE fprop(in int layer, in int outCell)
{
    TYPE sum = TYPE(0.);

    if (layer == 1)
    {
        for (int i = 0; i < NUM_CELLS_0/2; ++i)
            sum += weight(layer, i, outCell) * cellState(layer-1, i);
    }
    else if (layer == 2)
    {
        for (int i = 0; i < NUM_CELLS_1; ++i)
            sum += weight(layer, i, outCell) * cellState(layer-1, i);
    }
	else if (layer == 3)
    {
        for (int i = 0; i < NUM_CELLS_2; ++i)
            sum += weight(layer, i, outCell) * cellState(layer-1, i);
    }

    return activation(sum);
}


// error back propagation from layer to layer-1
TYPE bprop(in int layer, in int inCell)
{
    TYPE sum = TYPE(0.);

    if (layer == 1)
    {
        for (int i = 0; i < NUM_CELLS_1; ++i)
            sum += weight(layer, inCell, i) * cellError(layer, i);
    }
    else if (layer == 2)
    {
        for (int i = 0; i < NUM_CELLS_2; ++i)
            sum += weight(layer, inCell, i) * cellError(layer, i);
    }
	else if (layer == 3)
    {
        for (int i = 0; i < NUM_CELLS_3; ++i)
            sum += weight(layer, inCell, i) * cellError(layer, i);
    }

    return activation(sum);
}



void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    _initLayer();
    
    // previous pixel
    fragColor = texture2D(iChannel1, fragCoord.xy / iResolution.xy);

    int frame = int(mod(float(iFrame), float(NUM_FRAME_HOLD)));
    int curCell = int(fragCoord.x);
    int curY = int(fragCoord.y);
	
    // copy external input
	if (frame == 0)
    {
        if (curY == state_y[0] && curCell < NUM_INPUT)
        	fragColor = TYPE_TO_VEC4(
            	externalInputState(curCell) );
    }
    
    // forward propagation
	if (frame == 1)
    {
        if (curY == state_y[1] && curCell < NUM_CELLS_1)
			fragColor = TYPE_TO_VEC4( fprop(1, curCell) );
    }
#if NUM_LAYER > 2
    if (frame == 2)
    {
        if (curY == state_y[2] && curCell < NUM_CELLS_2)
			fragColor = TYPE_TO_VEC4(fprop(2, curCell) );
    }
#endif
#if NUM_LAYER > 3
	if (frame == 3)
    {
        if (curY == state_y[3] && curCell < NUM_CELLS_3)
			fragColor = TYPE_TO_VEC4(fprop(3, curCell) );
    }
#endif
    
    
#if DO_TRAIN != 0    
    // calc output error derivative
    if (frame == NUM_LAYER)
    {
        if (curY == error_y[NUM_LAYER-1] && curCell < NUM_OUTPUT)
            fragColor = TYPE_TO_VEC4(
                derivative(
                	expectedOutputState(curCell) 
                		- cellState(NUM_LAYER-1, curCell)
            	));
    }
 
    // backprop error derivative
#if NUM_LAYER > 3
    if (frame == NUM_LAYER+1)
    {
        if (curY == error_y[2] && curCell < num_cells[2])
			fragColor = TYPE_TO_VEC4( bprop(3, curCell) );
    }
#endif
#if NUM_LAYER > 2
    if (frame == NUM_LAYER+2)
    {
        if (curY == error_y[1] && curCell < num_cells[1])
			fragColor = TYPE_TO_VEC4( bprop(2, curCell) );
    }
#endif
    #if 1
    // backprop into input layer
    if (frame == NUM_LAYER+3)
    {
        if (curY == error_y[0] && curCell < num_cells[0])
			fragColor = TYPE_TO_VEC4( bprop(1, curCell) );
    }
    #endif
#endif
    
}
