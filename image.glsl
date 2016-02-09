// inputs: BufA, BufB, BufC

//This is just a refactoring of bergi's digit classifier to handle written numbers
//Find the original here: https://www.shadertoy.com/view/MdV3Wh

/*This is how the interface works...
Top of the screen is the Neural Net weights laid out by digit (garbage)
Left side is the net being trained.
Right side is for you to draw a digit in the grey square (hold left mouse)
Top of the grey square shows the network's guess.
Red dot in the middle erases your drawing.
Digits in the lower left of the screen can be replaced with the current drawing by clicking them.
*/

/* Neural Net Digit classifier on Shadertoy 

   (c) 0x7e0, Stefan Berke

   License: Creative Commons ...

   Trained to output the correct class for each of 10 digits

   No hidden layer, simply
   16x16 input -> 10 output
   2560 weights, no bias

   Left image shows current training, 
	 with desired (top) and actual (bottom) net output
   
   Right image is a test,
     with the network output (bottom) 
     and indicator of the cell with the highest output 


   It seems to learn most classes in ~15 seconds and
   then degrades somehow...
   Restart to learn from scratch
*/




float inputState(in ivec2 ip)
{
    vec2 p = (vec2(ip) + vec2(0.5, 1.5)) / iChannelResolution[0].xy;
    return texture2D(iChannel0, p).x;
}

float expectedOutputState(in int op)
{
    vec2 p = vec2(float(op)+.5, .5) / iChannelResolution[0].xy;
    return texture2D(iChannel0, p).x;
}

float outputState(in int op)
{
    vec2 p = vec2(float(op)+.5, .5) / iChannelResolution[1].xy;
    return texture2D(iChannel1, p).x;
}

float inputState2(in ivec2 ip)
{
    vec2 p = (vec2(ip) + vec2(16.5, 1.5)) / iChannelResolution[0].xy;
    return texture2D(iChannel0, p).x;
}

float outputState2(in int op)
{
    vec2 p = vec2(float(op)+.5, 1.5) / iChannelResolution[1].xy;
    return texture2D(iChannel1, p).x;
}

float weight(in int inCell, in int outCell)
{
    ivec2 ip = ivec2(inCell, outCell);
    vec2 p = (vec2(ip) + .5) / iChannelResolution[2].xy; 
    return (texture2D(iChannel2, p).x - .5) * 4.;
}


vec3 classifier(in vec2 uv)
{
    uv /= 10.;
    
    vec3 col = vec3(0.);
    if (uv.x >= 0. && uv.y >= 0. && uv.x < 16. && uv.y < 18.)
    {    
        float v = 0.2 + 0.8 * inputState(ivec2(uv));
		col = vec3(v);
    
    	if (uv.y >= 16. && uv.x <= 10.)
    	{
        	float s = outputState(int(uv));
        	col = vec3(max(0., s), 0., max(0.,-s));
    	}
    	if (uv.y >= 17. && uv.x <= 10.)
    	{
        	float s = expectedOutputState(int(uv));
        	col = vec3(max(0., s), 0., max(0.,-s));
    	}
    
    }
    return col;
}
//taken from digits/sliders/kbd widgets by FabriceNeyret2  https://www.shadertoy.com/view/MdKGRw
//     ... adapted from Andre in https://www.shadertoy.com/view/MdfGzf

float segment(vec2 uv, bool On) {
	return (On) ?  (1.-smoothstep(0.08,0.09+float(On)*0.02,abs(uv.x)))*
			       (1.-smoothstep(0.46,0.47+float(On)*0.02,abs(uv.y)+abs(uv.x)))
		        : 0.;
}

float digit(vec2 uv,int num) {
    uv.x=-uv.x;
	float seg= 0.;
    seg += segment(uv.yx+vec2(-1., 0.),num!=-1 && num!=1 && num!=4                    );
	seg += segment(uv.xy+vec2(-.5,-.5),num!=-1 && num!=1 && num!=2 && num!=3 && num!=7);
	seg += segment(uv.xy+vec2( .5,-.5),num!=-1 && num!=5 && num!=6                    );
   	seg += segment(uv.yx+vec2( 0., 0.),num!=-1 && num!=0 && num!=1 && num!=7          );
	seg += segment(uv.xy+vec2(-.5, .5),num==0 || num==2 || num==6 || num==8           );
	seg += segment(uv.xy+vec2( .5, .5),num!=-1 && num!=2                              );
    seg += segment(uv.yx+vec2( 1., 0.),num!=-1 && num!=1 && num!=4 && num!=7          );	
	return seg;
}

vec3 testImage(in vec2 uv)
{
    uv /= 10.;
    
    vec3 col = vec3(0.);
    
    if (uv.x >= 0. && uv.y >= 0. && uv.x < 16. && uv.y < 18.)
    {    
    	float v = 0.2 + 0.8 * inputState2(ivec2(uv));
		col = vec3(v);
    
    	// find output cell with highest output
    	float ma = 0.;
    	int outc = 0;
    	for (int i=0; i<10; ++i)
    	{
        	float s = outputState2(i);
        	if (s > ma)
        	{
            	ma = s;
            	outc = i;
        	}
    	}

    	// draw output state
    	if (uv.y >= 16. && uv.x <= 10.)
    	{
        	float s = outputState2(int(uv));
        	col = vec3(max(0., s), 0., max(0.,-s));
    	}
    
    	// draw highest state
    	if (uv.y >= 17. && uv.x <= 10.)
        	col = vec3(outc == int(uv.x) ? 1. : 0.);
        float d=digit(uv-vec2(12.0,16.5),outc);
        if(d>0.5)col=vec3(1.0);
    }
    return col;
}

float texLookup(in sampler2D sam, in vec2 pixel)
{
    return texture2D(sam, (pixel+.5) / iResolution.xy).x;
}

vec3 signedColor(in float x)
{
    return vec3(0., max(0., x), max(0., -x));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    fragColor = vec4(0., 0., 0., 1.);
    
	vec2 uv = fragCoord.xy / iResolution.y;
    vec2 pixel = uv * 200.;
    
    if (pixel.y >= 196.)
    {
		float v = texLookup(iChannel1, (pixel - vec2(0., 196.)) / 4.);
        fragColor = vec4(v, v, v, 1.);
    }
    else if (false)//pixel.y >= 190. && pixel.y < 194.)
    {
		float v = texLookup(iChannel1, (pixel - vec2(0., 190.+20.)) / 4.);
        fragColor = vec4(signedColor(v), 1.);
    }        
    
    
    if(fragCoord.x<160. && fragCoord.y<16.){
        fragColor=texture2D(iChannel0,(fragCoord+vec2(32.,0.0))/iResolution.xy);
        return;
    }
    
    vec3 col = classifier(fragCoord.xy - 25.);
	col = max(col, testImage(fragCoord.xy - vec2(300., 25.)));
    
    // render weight matrix
    {
        float x=floor(fragCoord.x/3.-5.),y=floor(fragCoord.y/3. - 75.);
        float X=y*16.+mod(x,16.),Y=floor(x/16.);
        int inCell = int(X);//fragCoord.x/2. - 10.);
        int outCell = int(Y);//fragCoord.y/2. - 130.);
        if (inCell >= 0 && inCell < 256 && outCell >= 0 && outCell < 10)
        {
            float w = 10.*weight(inCell, outCell);
    		col = vec3(max(0., w), 0., max(0.,-w));
        }
    }
    fragColor = vec4(col, 1.);
    vec2 ms=fragCoord.xy/iResolution.xy-0.5;
    if(length(ms)<0.035)fragColor=vec4(1.0,0.0,0.0,1.0);
    
    
	fragColor += vec4(signedColor(texture2D(iChannel2, uv/3.0).x), 1.);
    //fragColor += vec4(texture2D(iChannel2, uv/2.0).xyz, 1.);
}

