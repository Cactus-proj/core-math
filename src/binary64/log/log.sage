#load("../../generic/support/common.sage")
from functools import cmp_to_key

def cmp(x,y):
   if x[2] < y[2]:
      return int(-1)
   if x[2] > y[2]:
      return int(1)
   # now x[2] = y[2]
   if x[1] < y[1]:
      return int(-1)
   if x[1] > y[1]:
      return int(1)
   if x[0] < y[0]:
      return int(-1)
   if x[0] > y[0]:
      return int(1)
   return int(0)

# l = statall("/tmp/log")
def statall(f):
   f = open(f,"r")
   l = []
   while true:
      s = f.readline()
      if s=='':
         break
      s = s.split(" ")
      assert len(s) == 3
      t0 = ZZ(s[0])
      t1 = ZZ(s[1])
      e = ZZ(s[2])
      l.append((t0,t1,e))
   f.close()
   l.sort(key=cmp_to_key(cmp))
   l2 = []
   for t0,t1,e in l:
      if l2==[]:
         l2 = [((t0,e),(t1,e))]
      else:
         t1old,e1old = l2[-1][1]
         if t1old*2^e1old > t0*2^e:
            print ((t1old,e1old), (t0, e))
         assert t1old*2^e1old <= t0*2^e, "t1old*2^e1old <= t0*2^e"
         if t1old*2^e1old == t0*2^e:
            l2[-1] = (l2[-1][0],(t1,e))
         else:
            l2.append(((t0,e),(t1,e)))
   l = l2
   return l

def fast_two_sum(a,b):
   x = a+b
   z = x-a
   y = b-z
   return x, y

def fast_two_sum_a(a,b):
   x = a+b
   z = x-a
   y = z-b
   return x, -y

# p=6 rnd='RNDU' a=0x1p+24 b=0x1p+5
# checkall2(8)
# 8 RNDU 0x1p+33 0x1p+7 0.495881782945736
# checkall2(9)
# 9 RNDU 0x1p+37 0x1p+8 0.497932879377432
def checkall(p,rnd='RNDN'):
   R = RealField(p,rnd=rnd)
   maxerr = 0
   u = RR(2^-p)
   for e in range(3*p+3):
      for A in range(2^(p-1),2^p):
         a = R(A*2^e)
         for B in range(2^(p-1),2^p):
            b = R(B)
            x, y = fast_two_sum(a,b)
            X = x.exact_rational()
            Y = y.exact_rational()
            if X+Y != A*2^e+B:
               err = abs((X+Y-(A*2^e+B))/X)
               # err = abs((X+Y-(A*2^e+B))/(A*2^e+B))
               if err > maxerr:
                  maxerr = err
                  print (p,rnd,get_hex(a),get_hex(b),err/(4*u^2))

def checkall2(p):
   for r in 'NZUD':
      checkall(p,'RND'+r)

# given a list l of worst cases for log, output the largest one such that
# log1p(x) rounds to a different value than log(x) for any rounding mode
def compare_log1p(l):
   xmax = -infinity
   for x in l:
      if x <= xmax:
         continue
      for r in 'NZUD':
         R = RealField(53,rnd='RND'+r)
         y = log(R(x))
         z = R(n(log(1+x.exact_rational()),200))
         if y!=z:
            xmax = x
            print (get_hex(x))
            break

# ensures x*inv-1 is an exact multiple of denom(inv)
def znorm(x,inv):
   z = x.exact_rational()*inv.exact_rational()-1
   w = 1/inv.exact_rational().denom()
   u = (x.ulp()*w).exact_rational()
   assert z/u in ZZ
   return ZZ(z/u)

def best_inv(xmin,xmax,p):
   Ru = RealField(p,rnd='RNDU')
   rmax = Ru(1/xmin.exact_rational())
   # since we round up, rmax >= 1/xmin, thus x*rmax-1>=0 for xmin <= x <= xmax
   Rd = RealField(p,rnd='RNDD')
   rmin = Rd(1/xmax.exact_rational())
   # since we round down, rmin <= 1/xmax, thus x*rmin-1<=0 for xmin <= x <= xmax
   r = rmin
   bestz = infinity
   while r <= rmax:
      # compute bound for |x*r-1|
      R = r.exact_rational()
      zmin = abs(xmin.exact_rational()*R-1)
      zmax = abs(xmax.exact_rational()*R-1)
      z = n(max(zmin,zmax))
      # ensure x*r-1 is exact for xmin <= x <= xmax
      assert xmin.ulp()==xmax.ulp(), "xmin.ulp()==xmax.ulp()"
      u = xmin.ulp()/R.denom()
      # x*r is an integer multiple of u
      if z/u<=2^53: # x*r-1 is exact
         if z<bestz:
            bestz = z
            bestr = r
      r = r.nextabove()
   return bestr, bestz

# generate table _INVERSE
def inverse():
   k = 9
   R = RealField(10,rnd='RNDU')
   l = []
   maxz = 0
   x0 = RR("0x1.6a09e667f3bcdp-1",16)
   x1 = 2*x0
   imin = ZZ(floor(x0*2^k))
   imax = ZZ(floor(x1*2^k))
   for i in range(imin,imax+1):
      xmin = RR(i*2^-k)
      if xmin < x0:
         xmin = x0
      xmax = RR((i+1)*2^-k).nextbelow()
      if xmax >= x1:
         xmax = x1.nextbelow()
      r,z = best_inv(xmin,xmax,10)
      if z>maxz:
         maxz = z
         print (i,maxz)
      l.append(r)
   return l
