// 
// NETWORK INPUT
//
// code by eiffie, FabriceNeyret2, bergi ...
//
// renders a 16x16 random digit at 0,1
// and a 10x1 expected network output at 0,0 (the class of the digit)
// user can draw an image that will be analyzed

// ------- config --------

#define NUM_LAYER      	3
#define DO_TRAIN		1

// define to add random distortions for training
#define SMEAR
// define to add random scale and offset for training
//#define SCALE_AND_DISPLACE

// ----- end config ------



// ------------ auto config ------------

#if DO_TRAIN != 0
	#define NUM_FRAME_HOLD (NUM_LAYER*2)
#else
	#define NUM_FRAME_HOLD (NUM_LAYER)
#endif

// ------------- end auto config -------------




// ------------- pseudo handwritten digits ----------------

float Tube(vec2 pa, vec2 ba){return length(pa-ba*clamp(dot(pa,ba)/dot(ba,ba),0.0,1.0));}
float Arc(in vec2 p, float s, float e, float r1, float r2) {float t=clamp(atan(p.y*r1,p.x*r2),s,e);return length(p-vec2(r1*cos(t),r2*sin(t)));}
float num(vec2 p, int n){
	vec2 a=abs(p),a4=a-0.4;
	float d;
	if(n==0)return abs(length(p)-0.4); 
	if(n==1)return max(a.x,a4.y);
	if(n==2){
		d=Arc(p-vec2(0.0,0.2),-1.57,2.4,0.4,0.2);
		d=min(d,Arc(p+vec2(0.0,0.4),1.57,3.14,0.4,0.4));
		d=min(d,max(a4.x,abs(p.y+0.4)));
		return d;
	}
	if(n==3){
		d=Arc(p-vec2(0.0,0.2),-1.57,2.4,0.4,0.2);
		d=min(d,Arc(p+vec2(0.0,0.2),-2.4,1.57,0.4,0.2));
		return d;
	}
	if(n==4){
		d=max(a4.x,a.y);
		d=min(d,max(abs(p.x-0.4),a4.y));
		d=min(d,Tube(p-vec2(-0.4,0.0),vec2(0.6,0.4)));//split the difference in 4's
		return d;
	}
	if(n==5){
		d=max(a4.x,abs(p.y-0.4));
		d=min(d,max(abs(p.x+0.4),abs(p.y-0.2)-0.2));
		d=min(d,Arc(p-vec2(-0.05,-0.15),-2.45,2.45,0.45,0.3));
		return d;
	}
	if(n==6){
		d=Arc(p-vec2(0.0,-0.2),-3.1416,3.1416,0.4,0.2);
		d=min(d,Arc(p-vec2(0.2,-0.2),1.57,3.1416,0.6,0.6));
		return d;
	}
	if(n==7){
		d=max(a4.x,abs(p.y-0.4));
		d=min(d,Tube(p-vec2(-0.4,-0.4),vec2(0.8,0.8)));
		return d;
	}
	if(n==8){
		d=Arc(p-vec2(0.0,0.2),-3.1416,3.1416,0.4,0.2);
		d=min(d,Arc(p-vec2(0.0,-0.2),-3.1416,3.1416,0.4,0.2));
		return d;
	}
	if(n==9){
		d=Arc(p-vec2(0.0,0.2),-3.1416,3.1416,0.4,0.2);
		d=min(d,Arc(p-vec2(-0.2,0.2),-1.8,0.0,0.6,0.6));
		return d;
	}
    return 1.0;
}

vec3 printDigi(int di, vec2 uv)
{
    float d = num(uv,di);
    return vec3(smoothstep(0.18,0.0,d));
}

// ----------------------------------------


// hash by Dave_Hoskins https://www.shadertoy.com/view/4djSRW
float hash(float p)
{
	vec2 p2 = fract(vec2(p * 5.3983, p * 5.4427));
    p2 += dot(p2.yx, p2.xy + vec2(21.5351, 14.3137));
	return fract(p2.x * p2.y * 95.4337);
}


float rnd=1.234;
float rand(){return fract(sin(iGlobalTime+rnd++)*3424.4234);}
vec2 smear(vec2 uv){
    float a=rand()-0.5,b=rand()-0.5,c=(rand()-0.5)*(1.0-pow(abs(a)+abs(b),0.5));
	vec2 v=uv+vec2(a,b);
	return sin(30.0*a*v.yx+2.4*sin(10.0*b*vec2(v.x+v.y,v.x-v.y)))*0.6*c;
}
vec3 digitRect(in vec2 uv, in int digit)
{
    uv = (uv / 8. - 1.02) * 0.6;
	return printDigi(digit, uv);
}
vec3 savedRect(in vec2 uv, in int digit)
{
#ifdef SMEAR
    uv+=smear(uv/8.-1.0)*4.;
#endif
#ifdef SCALE_AND_DISPLACE
    vec2 sc = 1. + vec2(rand(), rand());
    uv *= sc;
    uv -= sc * vec2(rand(), rand()) * 6.;
#endif
    vec3 col = vec3(0.);
    if (uv.x >= 0. && uv.y >= 0. && uv.x < 16. && uv.y < 16.)
    {
    	uv.x+=float(digit+2)*16.;
		col = texture2D(iChannel0, uv/iResolution.xy).rgb;
    }
    return col;        
}
vec2 ms2Dig(vec2 ms){
    ms-=vec2(300.,25.);
    return ms/10.+vec2(16.,1.);
}
void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    // previous pixel
    fragColor = texture2D(iChannel0, fragCoord.xy / iResolution.xy);

    //ui
    if(iMouse.z>0.0){//clicked
        if(iMouse.y<16. && floor(fragCoord.x/16.)==floor(iMouse.x/16.)+2.){//replace training char with handwritten version
            float x=floor(iMouse.x/16.)*16.;
            fragColor=texture2D(iChannel0,vec2(mod(fragCoord.x,16.)+16.,fragCoord.y)/iResolution.xy);
            return;
        }
        if(fragCoord.x>=32.)return;
        vec2 ms=iMouse.xy/iResolution.xy-0.5;
        if(length(ms)<0.035)fragColor.rgb=vec3(0.0); //clear handwriting
        else{
            ms=ms2Dig(iMouse.xy);//record handwriting
            float d=length(ms-fragCoord.xy);
            d=smoothstep(1.5,0.0,d);
            fragColor.rgb=max(fragColor.rgb,vec3(d));
        }
    }

    int frame = int(mod(float(iFrame), float(NUM_FRAME_HOLD)));    
    if (frame == 0)
    {
        if (fragCoord.x >= 192. || fragCoord.y >= 17.)discard;
        fragColor=texture2D(iChannel0,fragCoord/iResolution.xy);
        if(iFrame<2){
            int i=int(fragCoord.x/16.);
            fragColor.rgb=digitRect((fragCoord - vec2(float(i)*16.,0.)), i-2);
        }
        if(fragCoord.x<16.){
            // random training digit
            int digit = int(hash(float(iFrame/7)*1.11)*9.+.5);
            fragColor.rgb = savedRect(fragCoord- vec2(0., 1.) + hash(iGlobalTime), digit); 
            // expected network output
            if (fragCoord.y < 1.)
            {
                fragColor.rgb = digit == int(fragCoord.x) ? vec3(.7) : vec3(0.);
            }
            return;
        }
    }
}
