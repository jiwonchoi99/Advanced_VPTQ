import torch
import pdb

from torch.autograd import Function

def pow2(x):
    return torch.pow(torch.tensor(2.0), x)

class FloatingPoint():
    def __init__(self, exp_width, man_width):
        self.exp_width = exp_width
        self.man_width = man_width
        
        self.mw_pow = pow2(man_width)
        
        exp_max = pow2(exp_width-1)-1
        self.max = (2-self.mw_pow.reciprocal()) * pow2(exp_max).cuda() # Consider the Inf
        self.min = pow2( -(exp_max+1) ).cuda()
        
        # Mantissa Quant 
        self.res = self.mw_pow.reciprocal().cuda()

    @torch.no_grad()    
    def quantize(self, x):
        if      self.exp_width==8 and self.man_width==23 : return x.type(torch.float32)
        elif    self.exp_width==5 and self.man_width==10 : return x.type(torch.float16)
        
        device = x.device
        self.max = self.max.to(device)
        self.min = self.min.to(device)
        self.res = self.res.to(device)

        x           = torch.clamp(x, min=-self.max, max=self.max)
        man, exp    = torch.frexp(x) 
        
        '''
        man         = man * 2 - 1 
        exp         = exp - 1 
        
        man         = (man/self.res).round() * self.res
        
        return torch.ldexp(man+1, exp)
        '''
        
        
        res         = self.res / 2
        man         = (man/res).round() * res
        
        return torch.ldexp(man, exp)

class FixedPoint():
    def __init__(self, int_width, frac_width):
        self.int_width = int_width
        self.frac_width = frac_width

        self.int_exp = pow2(int_width)
        self.frac_exp = pow2(frac_width)
        self.fxp_abs_max = self.int_exp - self.frac_exp.reciprocal()
        
    @torch.no_grad()    
    def quantize(self, x):
        device = x.device
        self.int_exp = self.int_exp.to(device)
        self.frac_exp = self.frac_exp.to(device)
        self.fxp_abs_max = self.fxp_abs_max.to(device)
        
        x = torch.round(x*self.frac_exp)
        x = x / self.frac_exp
        
        return torch.clamp(x, min = -self.fxp_abs_max, max = self.fxp_abs_max)

class MinMaxFixedPoint():
    def __init__(self, bit):
        self.bit_scale = (2**(bit-1))-1
        self.quanti_range = 1
        
    @torch.no_grad()    
    def quantize(self, x):
        # Quantization at bit, Google's sol
        input_max = torch.max(torch.abs(x))
        scale = (input_max*self.quanti_range)/self.bit_scale
        output = torch.round(torch.div(x, scale))
        output = torch.clamp(output, -1*self.bit_scale, self.bit_scale)
        return output * scale

def custom_precision(x, e_i_width, m_f_width, mode):
    if('fxp' in mode)   :   return custom_fxp(x, e_i_width, m_f_width)
    elif(mode=='fp')    :   return custom_fp(x, e_i_width, m_f_width)
    else:
        print("error in custom_precision!!!")
        exit()

def custom_fxp(x, iw , fw): #16bit -> integer = 7, float = 8\\
    iw_exp =pow2(iw)
    fw_exp =pow2(fw)
    fxp_abs_max = iw_exp - recip(fw_exp)

    x = torch.round(x*fw_exp)
    x = x / fw_exp
    return torch.clamp(x, min = -fxp_abs_max, max = fxp_abs_max) # -fxp_abs_max, torch.min(fxp_abs_max, x))

def denormal(x, denormal_unit_reci):
    return (torch.round(x.type(torch.float64) * denormal_unit_reci)/denormal_unit_reci).type(x.type())

@torch.no_grad()
def fp(x, exp=6, man=9):
    if      exp==5 and man==10 : return x.type(torch.float16)
    elif    exp==8 and man==23 : return x.type(torch.float32)
    else                       : return custom_fp(x, exp, man)

""" Floating Point Function with Subnormal 
@torch.no_grad()
def custom_fp(x, ew, mw):
    mw_pow  = pow2( mw )
    
    e_max   = pow2( ew -1 )-1   # exp range = 128
    fp_max  = pow2( e_max )  # max range = 2^128 -> 1 1111_1111 0x7
    fp_max  = ( 2 - mw_pow.reciprocal() ) * fp_max

    fp_min = pow2( -e_max+1 )      
    du_reci= pow2( mw + e_max.type(torch.float64) - 1  ).type(torch.float64) 

    def log2(x_in):
    	#return torch.div(torch.log(x_in), torch.log(torch.tensor(2.0, device=ga.device_gpu)))
    	return torch.log2(x_in)
    
    #clamp
    x=torch.clamp( x, min=-fp_max, max=fp_max)

    # lower than range  
    mask=(x>0) & (x<fp_min)
    x[mask]=denormal(x[mask], du_reci)
    mask=(x<0) & (x>-fp_min)
    x[mask]=-denormal(-x[mask], du_reci) 
    
    mask=(x==0)
    # mantissa adjust
    sign = torch.sign(x)
    x=x.abs()
    exp = torch.floor(log2(x))
    man = torch.round(mw_pow.type(torch.float64) * torch.mul(x.type(torch.float64) , pow2(-exp.type(torch.float64)))).type( x.type())
    
    x=sign*man/mw_pow *  pow2(exp)

    x[mask]=0
    return x
"""

# Floating Point Function w/o Subnormal 
@torch.no_grad()
def custom_fp(x, ew, mw):
    mw_pow  = pow2( mw )
    
    e_max   = pow2( ew -1 )-1   # ex> 5bit ==> 15
    fp_max  = pow2( e_max )     # max range = 2^128 -> 1 1111_1111 0x7
    fp_max  = ( 2 - mw_pow.reciprocal() ) * fp_max

    fp_min = pow2( - (e_max+1) )    # ex> 5bit ==> 2^-16
    #du_reci= pow2( mw + e_max.type(torch.float64) - 1  ).type(torch.float64) 

    def log2(x_in):
    	#return torch.div(torch.log(x_in), torch.log(torch.tensor(2.0, device=ga.device_gpu)))
    	return torch.log2(x_in)
    
    #clamp
    x=torch.clamp( x, min=-fp_max, max=fp_max)
    
    mask=(x<fp_min) & (x>-fp_min)
    sign = torch.sign(x)
    x=x.abs()
    exp = torch.floor(log2(x))
    man = torch.round(mw_pow.type(torch.float64) * torch.mul(x.type(torch.float64) , pow2(-exp.type(torch.float64)))).type( x.type())
    
    x=sign*man/mw_pow *  pow2(exp)

    x[mask]=0
    return x

def recip(x):
    return torch.tensor( x, device=ga.device_gpu).reciprocal()
