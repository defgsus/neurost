/* 
 * NeuroST - https://github.com/defgsus/neurost
 * (c) 0x7e0, Stefan Berke
 * License Creative Commons Attribution 3.0 Unported 
 * 
 */ 

/* * * Gradient descent * * *
 
 Description at https://github.com/defgsus/neurost/blob/master/idea.txt 
*/

 
// ------- config --------

const float INIT_VARIANCE = 0.1;
const float LEARNRATE = 0.01;

#define NUM_LAYER      	3
#define NUM_INPUT		(16*16)
#define NUM_OUTPUT		10
#define CELL_SIZE		3
#define DO_TRAIN		1

#define NUM_CELLS_0     NUM_INPUT
#define NUM_CELLS_1		40
#define NUM_CELLS_2		10
#define NUM_CELLS_3		NUM_OUTPUT

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

// train the weight between fromCell (layer-1) and toCell (layer)
// using error derivative (layer) and input state (layer-1)
TYPE adjust_weight(in int layer, in int fromCell, in int toCell)
{
    TYPE w = weight(layer, fromCell, toCell);
    
    TYPE der = cellError(layer, toCell);
    TYPE inp = cellState(layer-1, fromCell);
    
    // partial derivative w.r.t input state
    w += LEARNRATE * der * inp;
    
    return w;
}




// hashes by Dave_Hoskins https://www.shadertoy.com/view/4djSRW
float hash(float p)
{
	vec2 p2 = fract(vec2(p * 5.3983, p * 5.4427));
    p2 += dot(p2.yx, p2.xy + vec2(21.5351, 14.3137));
	return fract(p2.x * p2.y * 95.4337);
}
float hash(vec2 p)
{
	p  = fract(p * vec2(5.3983, 5.4427));
    p += dot(p.yx, p.xy + vec2(21.5351, 14.3137));
	return fract(p.x * p.y * 95.4337);
}



void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    _initLayer();
    
    //if (fragCoord.x >= 256. || fragCoord.y >= 10.)
    //    discard;
    
    // previous pixel
    fragColor = texture2D(iChannel2, fragCoord.xy / iResolution.xy);

    // INIT WEIGHTS
    if (iFrame == 0)
    {
        float w = (hash(fragCoord + iDate.zw)-.5)*2. * INIT_VARIANCE;
        
        fragColor = vec4(vec3(w), 1.0);
    }
    // LEARN STEP
    else if (iFrame >= NUM_FRAME_HOLD)
    {
		int frame = int(mod(float(iFrame), float(NUM_FRAME_HOLD)));
        
        if (frame == NUM_FRAME_HOLD - 1)
        {
            int inCell = int(fragCoord.x);
            int outCell = int(fragCoord.y);            
			int layer = 1;
			int numIn = num_cells[0];
            int numOut = num_cells[1];
            
            // find actual layer for shader's pixel

            if (outCell >= weight_y[2])
            {
                layer = 3;
                outCell -= weight_y[2];
                numIn = num_cells[2];
                numOut = num_cells[3];
            }
			else if (outCell >= weight_y[1])
            {
                layer = 2;
                outCell -= weight_y[1];
                numIn = num_cells[1];
                numOut = num_cells[2];
            }
			
            if (inCell < numIn && outCell < numOut 
                && layer < NUM_LAYER)
            {
            	fragColor = TYPE_TO_VEC4(
                    adjust_weight(layer, inCell, outCell) 
                	);
            	
                /*
            	if (layer == 1)
                	fragColor.x = 1.;
            	if (layer == 2)
                	fragColor.y = 1.;
            	if (layer == 3)
                	fragColor.z = 1.;
            	if (layer == 4)
                	fragColor.x = 1.;
				*/
            }
        }
    }
    /*
	else
        
    // error derivative
    if (fragCoord.y < 1. && fragCoord.x < float(NUM_CLASSES))
    {
	    int outCell = int(fragCoord.x);
        // previous output from forward pass
        float outs = outputState(outCell);
        // difference to expected output
        float err = expectedOutputState(outCell) - outs;
        // error derivative 
        float der = derivative(outs) * err;
        fragColor = vec4(der, 0., 0., 1.);
    }
	else 
    {
        // BACKPROP
        
        fragCoord.y -= 1.;

        if (fragCoord.y < float(NUM_HIDDEN_1))
        {
            int inCell = int(fragCoord.x);
            int outCell = int(fragCoord.y);
            ivec2 ip = ivec2(int(mod(float(inCell), 16.)), inCell / 16);

			// partial error derivative w.r.t layer input
        	float der = errorDerivative(1, outCell) * inputState(1, inCell);
                
            float w = weight(inCell, outCell);

            // adjust weight
            w += LEARNRATE * der; 

            fragColor = vec4(vec3(w * .25 + .5), 1.);
        }
    }
	*/
}
